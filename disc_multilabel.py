"""Train and evaluate four independent defect scores on complete disc photographs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as vision

from disc import CLASSES, SPLITS, digest, images_in, read_csv, source_path, square, write_csv

LABELS = tuple(CLASSES)
BATCH_SIZE = 16
THRESHOLD = 0.5
SEED = 42


def dataset(manifest, root):
    """Read whole-image labels and reject groups or image content crossing splits."""
    rows = read_csv(manifest)
    groups, hashes, paths = {}, {}, set()
    for row in rows:
        kind, group, split = row["data_kind"], row["group_id"], row["split"]
        path = source_path(root, row["path"])
        if split not in SPLITS or kind not in {"real", "synthetic"} or not group:
            raise ValueError(f"Invalid split or group for {path}")
        if path in paths or digest(path) != row["sha256"]:
            raise ValueError(f"Duplicate path or changed photo: {path}")
        paths.add(path)
        if groups.setdefault((kind, group), split) != split or hashes.setdefault(row["sha256"], split) != split:
            raise ValueError(f"Photo group or content crosses splits: {path}")
        if any(row[label] not in {"0", "1"} for label in LABELS):
            raise ValueError(f"Expected a 0/1 value for each defect: {path}")
        row["path"] = path
    return rows


def encoder(backbone, weights, dino_source, device):
    """Load the frozen features of the original classification backbone."""
    if backbone == "dinov2":
        if dino_source is None:
            raise ValueError("DINOv2 requires --dino-source pointing to the pinned official source")
        model = torch.hub.load(str(dino_source.resolve()), "dinov2_vits14", source="local", pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    else:
        from ultralytics import YOLO

        yolo = YOLO(str(weights))
        if yolo.task != "classify" or yolo.model.model[-1].linear.out_features != 1000:
            raise ValueError("Expected the original ImageNet YOLO11n-cls weights")
        model = yolo.model
        model.model[-1].linear = nn.Identity()
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def features(model, backbone, paths, device):
    """Encode complete, square-padded discs in the backbone's input format."""
    size = 350 if backbone == "dinov2" else 352
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = [square(p).resize((size, size), Image.Resampling.BICUBIC) for p in paths[start : start + BATCH_SIZE]]
        batch = torch.stack([vision.to_tensor(im) for im in images]).to(device)
        if backbone == "dinov2":
            batch = vision.normalize(batch, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            encoded = model(batch)
        else:
            _, encoded = model(batch)
        batches.append(encoded.float())
    return torch.cat(batches)


def train(args):
    """Fit a new linear head to the fixed features, with identical step budgets across runs."""
    rows = [r for r in dataset(args.data, args.root) if r["split"] == "train"]
    if not args.mixed:
        rows = [r for r in rows if r["data_kind"] == "real"]
    real, mothers = [], defaultdict(list)
    for i, row in enumerate(rows):
        if row["data_kind"] == "real":
            real.append(i)
        else:
            mothers[row["group_id"]].append(i)
    if not real or (args.mixed and not mothers) or args.steps < 1:
        raise ValueError("Training requires real images, positive steps, and synthetic mothers for --mixed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    model = encoder(args.backbone, args.weights, args.dino_source, args.device)
    x = features(model, args.backbone, [r["path"] for r in rows], args.device)
    y = torch.tensor([[int(r[c]) for c in LABELS] for r in rows], dtype=torch.float32, device=args.device)
    torch.manual_seed(SEED)
    head = nn.Linear(x.shape[1], len(LABELS)).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    rng = torch.Generator().manual_seed(SEED)
    real = torch.tensor(real)
    groups = list(mothers.values())
    for _ in range(args.steps):
        indexes = real[torch.randint(len(real), (10 if args.mixed else BATCH_SIZE,), generator=rng)].tolist()
        if args.mixed:
            for g in torch.randint(len(groups), (6,), generator=rng).tolist():
                indexes.append(groups[g][torch.randint(len(groups[g]), (), generator=rng).item()])
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(x[indexes]), y[indexes])
        loss.backward()
        optimizer.step()
    with args.output.open("xb") as handle:
        torch.save(
            {
                "backbone": args.backbone,
                "weights_sha256": digest(args.weights),
                "data_sha256": digest(args.data),
                "classes": LABELS,
                "threshold": THRESHOLD,
                "head": head.cpu().state_dict(),
                "steps": args.steps,
                "seed": SEED,
                "mixed": args.mixed,
            },
            handle,
        )
    print(f"Saved {args.output} after {args.steps} steps (final batch loss {loss.item():.4f})")


@torch.no_grad()
def scores(args, paths, checkpoint):
    """Load a saved head and return an independent sigmoid score for each defect."""
    if tuple(checkpoint["classes"]) != LABELS or checkpoint["weights_sha256"] != digest(args.weights):
        raise ValueError("Checkpoint expects a different class order or pretrained weight file")
    model = encoder(checkpoint["backbone"], args.weights, args.dino_source, args.device)
    x = features(model, checkpoint["backbone"], paths, args.device)
    head = nn.Linear(x.shape[1], len(LABELS)).to(args.device)
    head.load_state_dict(checkpoint["head"])
    probabilities = head(x).sigmoid().cpu()
    return probabilities, probabilities >= checkpoint["threshold"]


def predictions(paths, probabilities, decisions):
    """Create one row per image, allowing zero, one, or multiple class names."""
    return [
        {
            "path": str(path),
            "classes": json.dumps([CLASSES[c] for c, keep in zip(LABELS, flags) if keep], ensure_ascii=False),
            **{f"score_{c}": round(score, 6) for c, score in zip(LABELS, probs)},
        }
        for path, probs, flags in zip(paths, probabilities.tolist(), decisions.tolist())
    ]


def evaluate(args):
    """Report real and synthetic results separately, with equal weight per physical disc or mother."""
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if checkpoint["data_sha256"] != digest(args.data):
        raise ValueError("Evaluate against the same frozen dataset manifest used for training")
    rows = [r for r in dataset(args.data, args.root) if r["split"] == args.split]
    if not rows:
        raise ValueError(f"No images in split {args.split}")
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [r["path"] for r in rows]
    probabilities, decisions = scores(args, paths, checkpoint)
    target = torch.tensor([[int(r[c]) for c in LABELS] for r in rows], dtype=torch.bool)
    exact = (decisions == target).all(dim=1)
    result = predictions(paths, probabilities, decisions)
    summary = {}
    for kind in ("real", "synthetic"):
        group = defaultdict(list)
        for i, row in enumerate(rows):
            if row["data_kind"] == kind:
                group[row["group_id"]].append(i)
        if not group:
            continue
        indices = [i for members in group.values() for i in members]
        truth, pred = target[indices], decisions[indices]
        tp = (truth & pred).sum(dim=0)
        summary[kind] = {
            "images": len(indices),
            "groups": len(group),
            "exact_images": int(exact[indices].sum()),
            "exact_rate_by_group": sum(exact[ids].float().mean().item() for ids in group.values()) / len(group),
            "per_class": {
                label: {
                    "precision": tp[i].item() / pred[:, i].sum().item() if pred[:, i].any() else None,
                    "recall": tp[i].item() / truth[:, i].sum().item() if truth[:, i].any() else None,
                }
                for i, label in enumerate(LABELS)
            },
        }
    for i, (row, prediction) in enumerate(zip(rows, result)):
        prediction.update(
            group_id=row["group_id"],
            data_kind=row["data_kind"],
            truth=json.dumps([CLASSES[c] for c in LABELS if row[c] == "1"], ensure_ascii=False),
            exact=int(exact[i]),
        )
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "predictions.csv", result, list(result[0]))
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def predict(args):
    """Write the classes and four scores for a photograph or directory."""
    paths = images_in(args.input)
    if not paths:
        raise ValueError(f"No supported photographs: {args.input}")
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    rows = predictions(paths, *scores(args, paths, checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows, list(rows[0]))


def main():
    """Expose a separate entry point for the multi-label experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, handler in (("train", train), ("evaluate", evaluate), ("predict", predict)):
        sub = commands.add_parser(command)
        sub.set_defaults(handler=handler)
        sub.add_argument("--weights", type=Path, required=True, help="Original pretrained backbone weights")
        sub.add_argument("--dino-source", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", default="cuda:0")
        if command != "predict":
            sub.add_argument("--data", type=Path, required=True, help="Four-label CSV manifest")
            sub.add_argument("--root", type=Path, default=Path.cwd(), help="Root of image paths in the manifest")
        if command != "train":
            sub.add_argument("--model", type=Path, required=True, help="Saved four-label classification head")
    commands.choices["train"].add_argument("--backbone", choices=("dinov2", "yolo11n"), required=True)
    commands.choices["train"].add_argument("--steps", type=int, default=600)
    commands.choices["train"].add_argument("--mixed", action="store_true")
    commands.choices["evaluate"].add_argument("--split", choices=("val", "test"), default="test")
    commands.choices["predict"].add_argument("input", type=Path)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
