"""Manage disc-defect classification with physical-disc and explicit known-disc capture-group experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.classify import ClassificationTrainer

CLASSES = {
    "black_spot": "内部黑色缺陷",
    "notch": "边缘缺陷",
    "polyline": "内部黑色折线",
    "white_spot": "内部白色缺陷",
}
FIELDS = ["path", "class_name", "batch", "disc_id", "lighting_id", "split", "sha256"]
SPLITS = ("train", "val", "test")
CONDITIONS = {
    "original": None,
    "brightness_0.7": ("brightness", 0.7),
    "brightness_1.3": ("brightness", 1.3),
    "contrast_0.7": ("contrast", 0.7),
    "contrast_1.3": ("contrast", 1.3),
}


class RotationTrainer(ClassificationTrainer):
    """Add full-frame rotation to training while retaining native validation and inference transforms."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Prepend rotation before the native full-area resize, leaving validation unchanged."""
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train" and self.args.degrees:
            from torchvision.transforms import InterpolationMode, RandomRotation

            dataset.torch_transforms.transforms.insert(
                0,
                RandomRotation(
                    self.args.degrees,
                    interpolation=InterpolationMode.BILINEAR,
                    expand=True,
                    fill=114,
                ),
            )
        return dataset


class SyntheticTrainer(RotationTrainer):
    """Control synthetic exposure without treating rotated files as independent real discs."""

    def __init__(self, *args, synthetic_fraction, **kwargs):
        """Keep the requested sampling fraction outside Ultralytics configuration arguments."""
        self.synthetic_fraction = synthetic_fraction
        super().__init__(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        """Use weighted sampling for training and preserve the native validation loader."""
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode != "train":
            return loader
        if rank != -1 or not 0 < self.synthetic_fraction < 1:
            raise ValueError("Synthetic sampling requires one device and a fraction strictly between 0 and 1")
        import torch
        from torch.utils.data import WeightedRandomSampler

        from ultralytics.data.build import InfiniteDataLoader

        root = Path(self.args.data)
        rows = [r for r in read_csv(root / "manifest.csv") if r["split"] == "train"]
        synthetic = {str((root / r["prepared_path"]).resolve()) for r in rows if r.get("data_kind") == "synthetic"}
        real = len(loader.dataset) - len(synthetic)
        weight = self.synthetic_fraction * real / ((1 - self.synthetic_fraction) * len(synthetic))
        weights = [weight if str(Path(s[0]).resolve()) in synthetic else 1.0 for s in loader.dataset.samples]
        samples = round(real / (1 - self.synthetic_fraction))
        recipe = {
            "synthetic_fraction": self.synthetic_fraction,
            "real_images": real,
            "synthetic_images": len(synthetic),
            "samples_per_epoch": samples,
            "synthetic_file_weight": weight,
            "replacement": True,
            "sampler_seed": self.args.seed,
        }
        (self.save_dir / "sampling.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        sampler = WeightedRandomSampler(weights, samples, generator=torch.Generator().manual_seed(self.args.seed))
        return InfiniteDataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            collate_fn=loader.collate_fn,
            generator=loader.generator,
        )


def digest(path):
    """Hash a file to detect changes to already registered photographs."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    """Read an Excel-compatible registration or dataset snapshot."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    """Write a UTF-8 CSV with an explicit header, including when there are no rows."""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def images_in(root):
    """List supported photographs in a file or directory."""
    root = Path(root)
    return sorted(
        p for p in (root.rglob("*") if root.is_dir() else [root]) if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS
    )


def square(path):
    """Apply EXIF orientation and pad the entire RGB photograph to a square."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        side = max(image.size)
        return ImageOps.pad(image, (side, side), color=(114, 114, 114))


def source_path(source, relative):
    """Keep manifest paths inside their declared source directory."""
    path = (source / relative).resolve()
    path.relative_to(source.resolve())
    return path


def sheet(items, destination):
    """Render five labeled photographs per row without changing source images."""
    canvas = Image.new("RGB", (1100, ((len(items) + 4) // 5) * 248), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        x, y = (i % 5) * 220, (i // 5) * 248
        draw.multiline_text((x + 4, y + 3), label, fill="black")
        canvas.paste(ImageOps.contain(image, (212, 212)), (x + 4, y + 32))
    canvas.save(destination)


def index(args):
    """Register new paths with blank physical IDs and preserve existing annotations."""
    if not args.source.is_dir():
        raise NotADirectoryError(f"Original photograph directory does not exist: {args.source}")
    args.workspace.mkdir(parents=True, exist_ok=True)
    manifest = args.workspace / "manifest.csv"
    rows = read_csv(manifest) if manifest.exists() else []
    known = {row["path"] for row in rows}
    if len(known) != len(rows):
        raise ValueError("manifest.csv contains duplicate paths")
    initial = not manifest.exists()
    for path in images_in(args.source):
        relative = path.relative_to(args.source).as_posix()
        if relative in known:
            continue
        cls = next((name for name in CLASSES if path.stem.startswith(name + "_")), "")
        number = path.stem[len(cls) + 1 :] if cls else ""
        original = initial and path.parent == args.source and number.isdigit() and 1 <= int(number) <= 30
        batch = f"{(int(number) - 1) // 5 + 1:02d}" if original else ""
        rows.append(
            {
                "path": relative,
                "class_name": cls,
                "batch": batch,
                "disc_id": "",
                "lighting_id": "L0" if original else "",
                "split": "",
                "sha256": digest(path),
            }
        )
    write_csv(manifest, rows, FIELDS)
    pending = [row for row in rows if not row["disc_id"] or not row["lighting_id"] or not row["class_name"]]
    write_csv(args.workspace / "pending.csv", pending, FIELDS)
    previews = args.workspace / "previews"
    previews.mkdir(exist_ok=True)
    enhancement = []
    for cls in CLASSES:
        selected = sorted(
            (r for r in rows if r["class_name"] == cls and r["batch"]),
            key=lambda r: r["path"],
        )
        if not selected:
            continue
        sheet(
            [
                (
                    f"batch {r['batch']} / {r['disc_id'] or 'UNCONFIRMED'}\n{Path(r['path']).name}",
                    square(source_path(args.source, r["path"])),
                )
                for r in selected
            ],
            previews / f"{cls}.png",
        )
        image = square(source_path(args.source, selected[0]["path"]))
        for name, condition in CONDITIONS.items():
            enhanced = enhance(image, condition)
            enhancement.append((f"{cls}\n{name}", enhanced))
    if enhancement:
        sheet(enhancement, previews / "lighting.png")
    print(f"Registered {len(rows)} paths; pending physical/light labels: {len(pending)}.\nEdit {manifest}")


def registered(args):
    """Validate identity, source integrity, and existing immutable split assignments."""
    rows = read_csv(args.workspace / "manifest.csv")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image paths in manifest")
    ready = [r for r in rows if r["class_name"] and r["disc_id"] and r["lighting_id"]]
    if not ready:
        raise ValueError("No confirmed discs. Fill disc_id in manifest.csv after checking the previews.")
    groups = defaultdict(list)
    for row in ready:
        if row["class_name"] not in CLASSES:
            raise ValueError(f"Unknown class: {row['class_name']}")
        path = source_path(args.source, row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError(f"Registered source changed; preserve the old file and register a new path: {path}")
        square(path)  # Decode every confirmed photograph before creating a dataset version.
        groups[row["disc_id"]].append(row)
    ledger_path = args.workspace / "disc_splits.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    if not ledger:
        counts = Counter(r["class_name"] for r in ready)
        if len(ready) != 120 or any(counts[c] != 30 for c in CLASSES) or len(groups) != 12:
            raise ValueError("First preparation requires all 120 original images and 12 confirmed discs.")
        if any(len(g) != 10 or {r["lighting_id"] for r in g} != {"L0"} for g in groups.values()):
            raise ValueError("Initially each physical disc must have 10 L0 images; IDs must be globally unique.")
        rng = random.Random(42)
        for cls in CLASSES:
            discs = sorted(d for d, group in groups.items() if {r["class_name"] for r in group} == {cls})
            if len(discs) != 3:
                raise ValueError(f"Expected three confirmed physical discs for {cls}")
            rng.shuffle(discs)
            for disc, split in zip(discs, SPLITS):
                ledger[disc] = {"class_name": cls, "split": split}
    for disc, group in groups.items():
        classes = {r["class_name"] for r in group}
        specified = {r["split"] for r in group if r["split"]}
        if len(classes) != 1:
            raise ValueError(f"Physical disc {disc} has conflicting classes")
        if disc not in ledger:
            if len(specified) != 1 or not specified <= set(SPLITS):
                raise ValueError(f"New disc {disc}: explicitly register one train/val/test assignment")
            ledger[disc] = {
                "class_name": next(iter(classes)),
                "split": next(iter(specified)),
            }
        assignment = ledger[disc]
        if classes != {assignment["class_name"]} or specified - {assignment["split"]}:
            raise ValueError(f"Physical disc {disc}: class/split conflicts with disc_splits.json")
        for row in group:
            row["split"] = assignment["split"]
    coverage = {(r["split"], r["class_name"]) for r in ready}
    if coverage != {(split, cls) for split in SPLITS for cls in CLASSES}:
        raise ValueError("Every split must contain all four classes")
    pending = [r for r in rows if r not in ready]
    known = {r["path"] for r in rows}
    unindexed = [
        p.relative_to(args.source).as_posix()
        for p in images_in(args.source)
        if p.relative_to(args.source).as_posix() not in known
    ]
    write_csv(
        args.workspace / "pending.csv",
        pending + [{**dict.fromkeys(FIELDS, ""), "path": path} for path in unindexed],
        FIELDS,
    )
    print(f"Excluded pending registrations: {len(pending)}; unindexed photographs: {len(unindexed)}")
    return rows, ready, ledger


def prepare(args):
    """Create a new, lossless dataset snapshot and persist physical-disc assignments."""
    destination = args.workspace / "datasets" / args.version
    if destination.exists():
        raise FileExistsError(f"Choose a new --version; dataset already exists: {destination}")
    rows, ready, ledger = registered(args)
    destination.mkdir(parents=True)
    snapshot = []
    for row in ready:
        relative = source_path(args.source, row["path"]).relative_to(args.source)
        target = Path(row["split"]) / row["class_name"] / relative.with_name(relative.name + ".png")
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        square(source_path(args.source, row["path"])).save(output)
        snapshot.append(
            {
                **row,
                "prepared_path": target.as_posix(),
                "prepared_sha256": digest(output),
            }
        )
    write_csv(
        destination / "manifest.csv",
        snapshot,
        [*FIELDS, "prepared_path", "prepared_sha256"],
    )
    write_csv(args.workspace / "manifest.csv", rows, FIELDS)
    (args.workspace / "disc_splits.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"Prepared {destination}\nImages: {dict(Counter(r['split'] for r in ready))}")


def snapshot(data, experimental=False, known_discs=False):
    """Reject identity leakage, unexpected images, and changes to a prepared version."""
    if known_discs and not experimental:
        raise ValueError("Known-disc capture groups require the explicit experiment command")
    rows = read_csv(data / "manifest.csv")
    groups = defaultdict(set)
    real_train = {
        (r["disc_id"], r["class_name"])
        for r in rows
        if r["split"] == "train" and r.get("data_kind") != "synthetic" and r["disc_id"]
    }
    for row in rows:
        synthetic = row.get("data_kind") == "synthetic"
        external = row.get("data_kind") == "external"
        external_reference = (
            known_discs
            and row.get("source_role") == ("external_labeled_image" if external else "external_appearance_reference")
            and not row.get("source_disc_id")
            and not row["disc_id"]
            and row["split"] == "train"
            and row.get("shared_ancestry_group_id")
            and len(row.get("source_sha256", "")) == 64
            and set(row["source_sha256"]) <= set("0123456789abcdef")
        )
        if external and not external_reference:
            raise ValueError("External labeled images require train-only provenance in a known-disc experiment")
        if synthetic and (
            row["disc_id"]
            or (row["split"] != "train" and not experimental)
            or not row.get("mother_id")
            or (
                (row.get("source_disc_id"), row["class_name"]) not in real_train
                and not (
                    known_discs
                    and row.get("source_role") == "appearance_reference"
                    and any(disc == row.get("source_disc_id") for disc, _ in real_train)
                )
                and not external_reference
            )
        ):
            raise ValueError("Synthetic images require a mother ID, no physical ID, and a registered training source")
        if not row["lighting_id"] or (not synthetic and not external and not row["disc_id"]):
            raise ValueError("Prepared snapshots require confirmed physical-disc and lighting IDs")
        if row.get("shared_ancestry_group_id"):
            groups[("ancestry", row["shared_ancestry_group_id"])].add((row["split"],))
        if known_discs:
            if not row.get("capture_group"):
                raise ValueError("Known-disc experiments require a capture/lineage group for every image")
            groups[("capture", row["capture_group"])].add((row["class_name"], row["split"]))
        if synthetic or not known_discs:
            groups[("mother", row["mother_id"]) if synthetic else row["disc_id"]].add((row["class_name"], row["split"]))
        elif row["disc_id"]:
            groups[("disc_class", row["disc_id"])].add((row["class_name"],))
        groups[("content", row["prepared_sha256"])].add((row["class_name"], row["split"]))
        path = source_path(data, row["prepared_path"])
        if Path(row["prepared_path"]).parts[:2] != (row["split"], row["class_name"]):
            raise ValueError(f"Snapshot path/class mismatch: {row['prepared_path']}")
        if digest(path) != row["prepared_sha256"]:
            raise ValueError(f"Prepared photograph changed: {path}")
    if any(len(assignments) != 1 for assignments in groups.values()):
        raise ValueError("A physical disc, capture group, mother or identical image has conflicting assignments")
    expected = {r["prepared_path"] for r in rows}
    actual = {p.relative_to(data).as_posix() for split in SPLITS for p in images_in(data / split)}
    if len(expected) != len(rows) or actual != expected:
        raise ValueError("Prepared image inventory differs from its snapshot manifest")
    if {(r["split"], r["class_name"]) for r in rows} != {(s, c) for s in SPLITS for c in CLASSES}:
        raise ValueError("Prepared dataset must contain all four classes in every split")
    return rows


def train(args):
    """Train the existing YOLO classifier using a confirmed dataset snapshot."""
    from ultralytics import YOLO

    rows = snapshot(args.data, args.command == "experiment", getattr(args, "known_discs", False))
    run = args.workspace / "runs" / args.run
    if run.exists():
        raise FileExistsError(f"Choose a new --run: {run}")
    model = YOLO(args.weights)
    if model.task != "classify":
        raise ValueError("Training requires -cls classification weights")
    model.train(
        trainer=partial(SyntheticTrainer, synthetic_fraction=args.synthetic_fraction)
        if args.command != "experiment" and any(r.get("data_kind") == "synthetic" for r in rows)
        else RotationTrainer
        if args.rotation
        else None,
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16,
        device=args.device,
        workers=0,
        patience=20,
        seed=42,
        degrees=180.0 if args.rotation else 0.0,
        scale=0.0,
        auto_augment=None,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        amp=False,
        project=str(run.parent),
        name=run.name,
        exist_ok=False,
    )
    shutil.copy2(args.data / "manifest.csv", model.trainer.save_dir / "manifest.csv")
    if args.command == "experiment":
        (model.trainer.save_dir / "selection.json").write_text(
            json.dumps(
                {
                    "selection_source": "mixed_validation_only",
                    "weights_sha256": digest(model.trainer.best),
                }
            ),
            encoding="utf-8",
        )
    print(f"Best weights: {model.trainer.best}")


def enhance(image, condition):
    """Apply deterministic synthetic brightness or contrast for previews and stress tests."""
    if condition is None:
        return image
    kind, factor = condition
    operation = ImageEnhance.Brightness if kind == "brightness" else ImageEnhance.Contrast
    return operation(image).enhance(factor)


def infer(model, images, device):
    """Use model-provided class IDs and the same full-image preprocessing for every prediction."""
    if model.task != "classify" or set(model.names.values()) != set(CLASSES):
        raise ValueError("Expected a trained classifier with exactly the four registered defect names")
    results = model.predict(images, batch=16, device=device, verbose=False)
    return [(r.names[r.probs.top1], float(r.probs.top1conf)) for r in results]


def evaluate(args):
    """Evaluate held-out discs or explicitly requested capture groups with optional synthetic lighting stress."""
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix

    experimental = args.command == "experiment"
    known_discs = getattr(args, "known_discs", False)
    rows = [r for r in snapshot(args.data, experimental, known_discs) if r["split"] == args.split]
    output = args.workspace / "reports" / args.run
    output.mkdir(parents=True)
    shutil.copy2(args.data / "manifest.csv", output / "manifest.csv")
    predictions, metrics = [], []
    conditions = CONDITIONS if args.stress else {"original": None}
    summary = {
        "split": args.split,
        "images": len(rows),
        "independent_discs": len({r["disc_id"] for r in rows if r["disc_id"]}),
        "synthetic_mothers": len({r["mother_id"] for r in rows if r.get("mother_id")}),
        "experimental": experimental,
        "evaluation_protocol": "known_disc_capture_blocks" if known_discs else "physical_disc_holdout",
        "dataset_manifest_sha256": digest(args.data / "manifest.csv"),
        "models": [],
        "notice": (
            "Known discs occur in training and evaluation; capture blocks are held out, not physical objects. "
            if known_discs
            else ""
        )
        + "Rotated photos are correlated. Synthetic stress does not establish real cross-light accuracy.",
    }
    for number, weights in enumerate(args.models, 1):
        provenance = weights.parent.parent / "manifest.csv"
        catalog = weights.parent / "model_catalog.json"
        if catalog.exists():
            record = next(
                m for m in json.loads(catalog.read_text(encoding="utf-8"))["models"] if m["file"] == weights.name
            )
            if digest(weights) != record["sha256"]:
                raise ValueError(f"Weights differ from the model catalog: {weights}")
            provenance = weights.parent.parent / record["training_manifest"]
        if provenance.exists():
            training = [r for r in read_csv(provenance) if r["split"] == "train"]
            unit = "capture_group" if known_discs else "disc_id"
            seen = {r[unit] for r in training if r.get(unit)}
            if seen & {r[unit] for r in rows if r.get(unit)}:
                raise ValueError(f"Evaluation {unit} values were used to train {weights}")
            if {r["prepared_sha256"] for r in training} & {r["prepared_sha256"] for r in rows}:
                raise ValueError(f"Evaluation image contents were used to train {weights}")
            for unit in ("mother_id", "shared_ancestry_group_id"):
                seen = {r[unit] for r in training if r.get(unit)}
                if seen & {r[unit] for r in rows if r.get(unit)}:
                    raise ValueError(f"Evaluation {unit} values were used to train {weights}")
        model = YOLO(str(weights))
        model_id = str(weights.resolve())
        model_dir = output / f"model_{number}"
        model_dir.mkdir()
        base, model_metrics = {}, []
        for name, condition in conditions.items():
            entries = []
            confusion = ConfusionMatrix(names=model.names, task="classify")
            indices = {label: i for i, label in model.names.items()}
            for start in range(0, len(rows), 16):
                batch = rows[start : start + 16]
                images = [enhance(square(args.data / r["prepared_path"]), condition) for r in batch]
                for row, (predicted, confidence) in zip(batch, infer(model, images, args.device)):
                    entry = {
                        "model": model_id,
                        "condition": name,
                        "path": row["path"],
                        "disc_id": row["disc_id"],
                        "data_kind": row.get("data_kind") or "real",
                        "mother_id": row.get("mother_id", ""),
                        "capture_group": row.get("capture_group", ""),
                        "lighting_id": row["lighting_id"],
                        "true_class": row["class_name"],
                        "predicted_class": predicted,
                        "predicted_zh": CLASSES[predicted],
                        "confidence": confidence,
                        "correct": int(predicted == row["class_name"]),
                    }
                    entries.append(entry)
                    confusion.matrix[indices[predicted], indices[row["class_name"]]] += 1
            predictions.extend(entries)
            groups = defaultdict(list)
            for entry in entries:
                groups[("overall", "all")].append(entry)
                for scope in ("true_class", "lighting_id", "disc_id"):
                    if entry[scope]:
                        groups[(scope, entry[scope])].append(entry)
                groups[("lighting_class", f"{entry['lighting_id']}/{entry['true_class']}")].append(entry)
                if experimental:
                    groups[("data_kind", entry["data_kind"])].append(entry)
                    groups[("domain_class", f"{entry['data_kind']}/{entry['true_class']}")].append(entry)
                    if entry["mother_id"]:
                        groups[("mother_id", entry["mother_id"])].append(entry)
            for (scope, group), values in groups.items():
                correct = sum(v["correct"] for v in values)
                accuracy = correct / len(values)
                if name == "original":
                    base[(scope, group)] = accuracy
                metric = {
                    "model": model_id,
                    "condition": name,
                    "scope": scope,
                    "group": group,
                    "images": len(values),
                    "independent_discs": len({v["disc_id"] for v in values if v["disc_id"]}),
                    "correct": correct,
                    "top1": accuracy,
                    "delta_from_original": accuracy - base[(scope, group)],
                }
                model_metrics.append(metric)
            folder = model_dir / name
            folder.mkdir()
            confusion.plot(normalize=False, save_dir=folder)
        real = [
            m
            for m in model_metrics
            if m["condition"] == "original" and m["scope"] == "lighting_id" and m["independent_discs"]
        ]
        summary["models"].append(
            {
                "weights": model_id,
                "worst_real_lighting": min(real, key=lambda m: m["top1"]),
            }
        )
        metrics.extend(model_metrics)
    write_csv(output / "predictions.csv", predictions, list(predictions[0]))
    write_csv(
        output / "errors.csv",
        [p for p in predictions if not p["correct"]],
        list(predictions[0]),
    )
    write_csv(output / "metrics.csv", metrics, list(metrics[0]))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reports: {output}; {len(rows)} images from {summary['independent_discs']} independent discs")


def predict(args):
    """Predict one photograph or a directory and export Chinese and English labels."""
    from ultralytics import YOLO

    paths = images_in(args.input)
    if not paths:
        raise ValueError(f"No supported photographs: {args.input}")
    if args.output.exists():
        raise FileExistsError(args.output)
    model = YOLO(str(args.model))
    rows = []
    for start in range(0, len(paths), 16):
        batch = paths[start : start + 16]
        for path, (cls, confidence) in zip(batch, infer(model, [square(p) for p in batch], args.device)):
            rows.append(
                {
                    "path": str(path.resolve()),
                    "class_name": cls,
                    "class_zh": CLASSES[cls],
                    "confidence": confidence,
                }
            )
            print(f"{path.name}: {CLASSES[cls]} ({confidence:.2%})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows, list(rows[0]))


def main():
    """Expose preparation, training, evaluation, and prediction as one standalone command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset"),
        help="Original photographs (never modified)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("runs/disc"),
        help="Registration and experiment outputs",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Create/update editable registration and grouped previews")
    command = commands.add_parser("prepare", help="Validate identities and create a new dataset version")
    command.add_argument("--version", required=True)
    command = commands.add_parser("train", help="Train only from a confirmed dataset version")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--weights", default="yolo11n-cls.pt")
    command.add_argument("--run", required=True)
    command.add_argument("--epochs", type=int, default=100)
    command.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Training input size; saved in weights for inference",
    )
    command.add_argument(
        "--synthetic-fraction",
        type=float,
        default=0.25,
        help="Expected synthetic share of weighted draws for mixed snapshots (single device)",
    )
    command.add_argument(
        "--rotation",
        action="store_true",
        help="Train with random full-frame rotations in [-180, 180]",
    )
    command.add_argument("--device", default="0")
    command = commands.add_parser(
        "evaluate",
        help="Report held-out accuracy and optional synthetic lighting stress",
    )
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--models", type=Path, nargs="+", required=True)
    command.add_argument("--run", required=True)
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--stress", action="store_true")
    command.add_argument("--device", default="0")
    command = commands.add_parser("experiment", help="Explicit mixed real/synthetic training and evaluation")
    command.add_argument("phase", choices=("train", "val", "test"))
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--run", required=True, help="Unique training name or evaluation report name")
    command.add_argument("--weights", default="yolo11n-cls.pt")
    command.add_argument("--models", type=Path, nargs="+", help="Required for val/test")
    command.add_argument("--epochs", type=int, default=100)
    command.add_argument("--device", default="0")
    command.add_argument("--stress", action="store_true")
    command.add_argument(
        "--known-discs",
        action="store_true",
        help="Adapt known physical discs using isolated capture/lineage groups; not unseen-object evaluation",
    )
    command = commands.add_parser("predict", help="Predict a file or directory to CSV")
    command.add_argument("input", type=Path)
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default="0")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.workspace = args.workspace.resolve()
    if args.workspace == args.source or args.source in args.workspace.parents:
        parser.error("--workspace must be outside --source so generated previews cannot become source images")
    for field in ("version", "run"):
        value = getattr(args, field, None)
        if value is not None and (Path(value).name != value or value in {".", ".."}):
            parser.error(f"--{field} must be a single directory name")
    if hasattr(args, "data"):
        args.data = args.data.resolve()
    if args.command == "experiment":
        args.imgsz, args.rotation = 352, False
        if args.phase != "train" and not args.models:
            parser.error("experiment val/test requires --models")
        args.split = args.phase
        (train if args.phase == "train" else evaluate)(args)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
