> 完整项目交接、四个模型、数据版本及继续训练步骤见[项目交接说明](项目交接说明.md)。

# 圆盘缺陷四分类

使用已有的 Ultralytics 环境和 `yolo11n-cls.pt`，识别边缘缺陷、内部白色缺陷、内部黑色缺陷、内部黑色折线。每张照片输出一种缺陷及置信度，不标注位置，不包含正常圆盘类别。工具不修改 Ultralytics 源码或原始照片。

## 环境与入口

使用已经配置好的 `yolo` Conda 环境。已在 Python 3.12、Ultralytics 8.4.163、RTX 4070 Ti SUPER 上验证。脚本依赖该环境已有的 Ultralytics、Pillow 和绘图依赖，不需要额外训练框架。

在 **Anaconda PowerShell Prompt** 中执行，修改前三个路径为实际位置：

```powershell
conda activate yolo
$discTool = "C:\path\to\AIC\disc.py"
$discSource = "C:\Laptop\mian\竞赛\ultralytics-8.4.163\dataset"
$discWorkspace = "C:\path\to\disc-experiment"
$discWeights = "C:\Laptop\mian\竞赛\ultralytics-8.4.163\yolo11n-cls.pt"
python $discTool --help
```

实验目录必须位于原始照片目录之外。后续所有命令必须使用同一个实验目录，以保留圆盘归属记录。`--version` 和 `--run` 使用新的单级目录名，已有数据集、训练目录、报告和预测 CSV 不会被覆盖。

## 1. 确认实物身份

```powershell
python $discTool --source $discSource --workspace $discWorkspace index
```

生成：

- `manifest.csv`：可编辑登记表，UTF-8 BOM 编码，可用 Excel 打开。
- `pending.csv`：尚未登记完整的图片。
- `previews/<类别>.png`：初始照片分组预览，每行是连续五张的拍摄批次。
- `previews/lighting.png`：原图及亮度、对比度变化预览。

初始数据每类 30 张，文件名为 `notch_001.bmp` 等。查看每类六行预览，人工确认哪两行对应同一个圆盘。**连续十张不能直接视作一个圆盘。**

编辑 `manifest.csv`：

| 列            | 用途                                                                  |
| ------------- | --------------------------------------------------------------------- |
| `path`        | 相对原始照片目录的路径，不要改动已有路径                              |
| `class_name`  | `notch`、`white_spot`、`black_spot`、`polyline`                       |
| `batch`       | 初始连续五张的批次号，辅助核对，不用于自动判断实物身份                |
| `disc_id`     | 全局唯一的实物 ID，例如 `notch_01`；同一实物十张照片使用相同 ID       |
| `lighting_id` | 初始照片已经标为 `L0`，后续填写实际光照条件                           |
| `split`       | 初始留空，由准备命令分配；新实物必须明确填写 `train`、`val` 或 `test` |
| `sha256`      | 自动生成的原图校验值，不要修改                                        |

初次准备要求确认全部 120 张：每类三个圆盘、每个圆盘十张 `L0` 照片。未确认身份时工具会停止，不会猜测配对。可以再次运行 `index` 刷新预览和待整理清单，已有人工标注会保留。

## 2. 生成不可覆盖的数据集版本

```powershell
python $discTool --source $discSource --workspace $discWorkspace prepare --version v1
$discData = Join-Path $discWorkspace "datasets\v1"
```

种子固定为 42，每类分别用一个实物训练、一个验证、一个测试，初始为 40/40/40 张。`disc_splits.json` 保存实物归属；不要删除、编辑它来重新分配照片。

照片按最长边补为正方形，输出为无损 PNG，并生成 `train/val/test` 下的四类目录。每个版本保存含原始路径、实物、光照、集合、生成路径及校验值的登记快照。训练和评估都会检查快照，拒绝跨集合实物、额外图片或被改动的准备数据。

## 3. 短程检查与正式训练

```powershell
python $discTool --workspace $discWorkspace train --data $discData --weights $discWeights --run smoke_v1 --epochs 2
python $discTool --workspace $discWorkspace train --data $discData --weights $discWeights --run baseline_v1
$discModel = Join-Path $discWorkspace "runs\baseline_v1\weights\best.pt"
```

先确认 `lighting.png` 中的细小缺陷仍可辨认，再运行训练。两个命令均从预训练权重开始；正式训练不沿用短程检查权重。

默认参数：100 轮、320 输入尺寸、batch 16、GPU 0、workers 0、patience 20、seed 42、scale 0、auto_augment None、erasing 0、hsv_h/hsv_s 0、hsv_v 0.3。训练使用 FP32（`amp=False`），避免 AMP 检查额外下载检测模型。可用 `--device cpu` 进行 CPU 验证。

当前分类实现的 `hsv_v=0.3` 同时随机改变亮度和对比度，倍率分别为 0.7～1.3。增强只发生在训练加载过程中，不产生新的独立实物。补为正方形并设置 `scale=0` 保留圆盘边缘；验证与预测没有随机增强。

每次训练保留最佳/最终权重、参数 YAML、训练曲线、验证结果以及数据登记快照。验证集用于选择最佳模型和提前停止。

### 输入尺寸与模型大小对照

训练可通过`--imgsz`改变输入尺寸，默认仍为320；`--weights`可指定`yolo11s-cls.pt`等分类预训练权重。评估和预测自动沿用各模型权重保存的训练尺寸，不会将352模型强制缩放为320。

对于补边后352×352的照片，可固定同一数据集和基线参数，分别测试以下配置。正式训练前给每种配置使用新名称运行`--epochs 2`短程检查；正式训练从对应预训练权重重新开始。

```powershell
python $discTool --workspace $discWorkspace train --data $discData --weights $discWeights --imgsz 352 --run size_n352
python $discTool --workspace $discWorkspace train --data $discData --weights "yolo11s-cls.pt" --imgsz 320 --run size_s320
python $discTool --workspace $discWorkspace train --data $discData --weights "yolo11s-cls.pt" --imgsz 352 --run size_s352
```

这些命令不叠加旋转，便于区分尺寸和模型容量的影响。先按验证集比较总体、各类及合成光照指标，记录模型选择，再对同一测试集一次性报告结果。提高尺寸不会产生原图中不存在的细节，较大模型也不保证准确率提高。使用新的实验名称，保留原基线和历史结果。

### 旋转增强对照

保持同一数据集、预训练权重和其他参数，在训练命令添加`--rotation`：

```powershell
python $discTool --workspace $discWorkspace train --data $discData --weights $discWeights --run smoke_rotation --epochs 2 --rotation
python $discTool --workspace $discWorkspace train --data $discData --weights $discWeights --run rotation --rotation
```

训练加载时随机旋转−180°～180°，使用双线性插值和灰色114补边；扩展画布保留完整图像，再通过原训练变换缩放到指定输入尺寸（默认320）。斜向旋转会因新增补边而缩小画面中的圆盘。旋转角度以`degrees=180`保存在参数YAML中；不添加该选项时保持原基线流程。

验证、测试和预测不旋转，沿用原处理。先检查训练图中的细线和圆盘边缘，再用相同验证/测试数据比较原模型与旋转模型。增强图不会另存为新实物样本，无需生成新的数据集版本；使用新的实验和报告名称保留原结果。

### 合成图片与真实照片混合训练

合成图片只加入新版本的训练目录，验证和测试保留真实照片。快照CSV仍使用原有字段，合成行另外填写
`data_kind=synthetic`、全局唯一的`mother_id`和`source_disc_id`；`disc_id`留空，`split=train`。
来源圆盘必须是同类别的真实训练圆盘。每张旋转图保留母图ID，不能分入验证或测试；高清母图不与其缩放版本重复加入。

在明确使用 `experiment --known-discs` 的适配实验中，未知实物身份的外部外观参考后代可按来源整组划分。清单须将
`source_role` 标为 `external_appearance_reference`，`source_disc_id` 留空，并记录有效的 `source_sha256` 和非空的
`shared_ancestry_group_id`。仍须记录母图、拍摄来源组和合成光照标记；同一母图及共同来源组不得跨集合。
这不会放宽正式实物隔离训练的来源要求，也不能将外部照片猜测登记为已有圆盘。参考照片及其后代参与训练后，
该参考不能继续作为独立外部测试证据；外部实物身份未确认时，不声称验证了未见实物。

按目录标签导入、尚未确认实物身份或采集来源的图片使用 `data_kind=external`、
`source_role=external_labeled_image`；`disc_id` 和 `source_disc_id` 留空。与外部参考相同，必须记录来源
SHA-256、共同来源组和明确的光照标记（未知时标为未登记），并且只允许通过 `experiment --known-discs`
进入适配实验。已确认的实物登记表不填入猜测的身份。外部图片单独计数，不当作已确认真实圆盘或合成图片；
旋转副本、参考照片及已记录的共同祖先必须整组管理。同一来源组只能属于一个集合，不能将衍生图片与其参考
分别用于训练和验证、测试；相同物理圆盘仍可出现在不同拍摄组中，这不属于未见实物评估。

重新划分时另建不可覆盖的快照，保留原有字段和历史归属，并将同一拍摄组、母图、参考内容校验值及共同祖先
连接成来源组，保存到 `shared_ancestry_group_id`。使用固定随机种子，以约70%/15%/15%为目标分配整个来源组，
记录实际数量；共享参考的大组可能使比例及各域类别覆盖不均衡。验证集选择权重后再评估测试集，分别报告
真实、合成与外部图片。只有新模型未参与训练的来源组才能作为该模型的留出评估；历史模型若训练过新测试图片，
不能直接用它们的准确率作公平排名。未登记的外部实物身份也不能靠旋转内容分组得到确认。

创建快照时，记录每个输出文件的`prepared_path`和`prepared_sha256`，并保留来源及角度字段。

混合快照默认按期望25%的合成比例加权抽样；可使用`--synthetic-fraction 0.5`比较50%。支持单设备训练，
不修改Ultralytics循环。每轮抽样数量约为真实训练图片数除以真实抽样比例，使用有放回抽样，实际每轮比例会波动。
同一母图若拥有相同数量的角度文件，其总抽样权重相等；旋转版本不会登记成新的实物。
`sampling.json`保存比例、抽样数量和随机种子；原有仅真实照片的训练方式保持不变。

```powershell
python $discTool --workspace $discWorkspace train --data "C:\path\to\mixed_snapshot" --weights $discWeights --imgsz 352 --synthetic-fraction 0.25 --run mixed25
```

离线旋转已经改变了画面方向和补边，本轮比较不再叠加`--rotation`。先用真实验证集选择比例和最佳模型，
记录选择后再在同一真实测试集上比较；合成图预测准确率不用于证明实际收益。

## 4. 独立评估与光照压力测试

```powershell
python $discTool --workspace $discWorkspace evaluate --data $discData --models $discModel --run test_v1 --stress
```

评估默认使用测试集。调参时可用 `--split val`，不要用测试结果反复选择增强参数。

报告位于 `reports/test_v1`：

- `metrics.csv`：原图、亮度 0.7/1.3、对比度 0.7/1.3 五种条件的 Top-1，按总体、类别、真实光照、圆盘及光照×类别分组，并列出相对原图的变化。
- `predictions.csv`、`errors.csv`：逐图预测及误判，含实物 ID、光照 ID、真实类别、预测类别、中文名称和置信度。
- 每个模型和测试条件的混淆矩阵 PNG；横轴为真实类别、纵轴为预测类别。
- `summary.json`：独立圆盘数量、数据快照校验值及表现最弱的真实光照条件。

初始测试集虽然有 40 张照片，但只有 **4 个独立圆盘**。合成明暗测试不能证明模型适应真实光源方向、阴影或反光。模型置信度也不等于实测准确率。

## 5. 单张或文件夹预测

```powershell
python $discTool predict "C:\path\to\new_photo.bmp" --model $discModel --output (Join-Path $discWorkspace "single_prediction.csv")
python $discTool predict "C:\path\to\new_photos" --model $discModel --output (Join-Path $discWorkspace "batch_predictions.csv")
```

每张照片完整补边后预测。CSV 含完整路径、英文类别、中文类别及置信度。类别编号读取模型映射，不硬编码数字。工具拒绝未训练成这四类的 ImageNet 分类权重。

## 6. 后续加入真实多光照照片

使用同一色温光源，保持相机、背景、曝光设置一致，覆盖实际会遇到的照明范围：

| 光照 ID | 条件         |
| ------- | ------------ |
| `L0`    | 现有光照     |
| `L1`    | 固定方向较暗 |
| `L2`    | 固定方向较亮 |
| `L3`    | 左侧照明     |
| `L4`    | 右侧照明     |

默认每个圆盘每个新增条件采集五张不同角度照片，四类以及训练、验证、测试圆盘均覆盖这些条件。后续照片允许其他分辨率及不同数量。

1. 使用新文件名放入原始目录或子目录，保留旧图；例如 `notch_01_L1_001.bmp`。文件名前缀可预填类别，但实物 ID 和光照必须人工登记，不能从编号猜测。
2. 运行 `index`，在登记表中新行填写 `disc_id` 和 `lighting_id`。已有圆盘的 `split` 可留空，准备时会继承原有归属。未登记完整的新增图不会进入训练；未运行索引的新增文件也会报告为待整理。
3. 运行 `prepare --version v2`。已有实物不重新随机分配，新实物必须先登记唯一 ID 和明确集合归属。不要将测试圆盘的新照片放入训练集。
4. 从相同预训练权重训练新实验 `baseline_v2`，使用 v2 的全部训练数据，保留 v1 数据和权重。
5. 用下面的命令让新旧模型在**同一个 v2 测试集**上评估：

```powershell
$discDataV2 = Join-Path $discWorkspace "datasets\v2"
$discModelV2 = Join-Path $discWorkspace "runs\baseline_v2\weights\best.pt"
python $discTool --workspace $discWorkspace evaluate --data $discDataV2 --models $discModel $discModelV2 --run compare_v2 --stress
```

补拍同一实物能扩充光照覆盖，不会增加独立圆盘数量。跨光照能力以真实条件下的分组指标衡量；本项目不预设未经实测支持的准确率保证。

## V15：加入近边缘白色缺陷后重新划分

本地实验 `v15_regrouped_all_20261023` 复用现有训练入口，纳入 V14 的 7000 张有效图片，
以及 `white_near_rim_batch_v2_20261023`、`white_near_rim_large_batch_v1_20261023` 各 240 张局部角度图。
共 7480 张：1280 张已登记真实照片、5280 张合成角度图、920 张已标注外部图片。
高清母图和预览不重复计入；外部无标签的 12 张仍排除。

固定种子 42，按已记录来源的连通组重新分配，目标比例为 70%/15%/15%，实际为
5358/1061/1061 张。大来源组不可拆分，因此实际比例和各域类别数量并不完全均衡。
同一拍摄组、母图及已记录的共同参考来源保持在同一集合；数据快照保留旧归属、来源连接及分配记录。
这仍是现有圆盘的适配实验，不能解释为未见实物测试。

在包含照片和模型的项目根目录，使用已有 YOLO 环境运行；CPU 将设备改为 `cpu`，每次使用新运行名称：

```powershell
python disc.py --workspace disc-experiment experiment train --known-discs --data disc-experiment/datasets/v15_regrouped_all_20261023 --weights yolo11n-cls.pt --run teammate_v15_smoke_001 --epochs 2 --device 0
python disc.py --workspace disc-experiment experiment train --known-discs --data disc-experiment/datasets/v15_regrouped_all_20261023 --weights yolo11n-cls.pt --run teammate_v15_full_001 --epochs 100 --device 0
python disc.py --workspace disc-experiment experiment test --known-discs --data disc-experiment/datasets/v15_regrouped_all_20261023 --models models/known_discs_regrouped_n352_v15_20261023.pt --run teammate_v15_test_001 --stress --device 0
```

正式训练从原始预训练权重开始，以验证集选择 `best.pt`；测试集不更新参数。
准备和分组配方保存在本地 `disc-experiment/checks/regroup_v15_20261023/`，不提交照片或实验产物。
新近边缘图与原有近边缘图共享祖先，必须整组处理；若该组进入训练，其识别结果仅作拟合检查。
历史 V14 模型训练过新留出集中的部分图片，同集比较须列出暴露情况，不能作为独立公平排名。
真实照片仍仅记录 L0 光照，多光照能力尚未验证。

## V16：补充验证覆盖并比较学习率

实验训练入口可显式传入 `--optimizer`、`--lr0`、`--momentum` 和 `--warmup-bias-lr`，
直接复用 Ultralytics 的同名参数。默认仍为 `auto`；自动优化器会覆盖指定的学习率，比较学习率时须选择
`AdamW` 或 `SGD`。这些选项只用于 `experiment`，正式实物隔离训练的默认参数保持不变。

本地 V16 保留 V15 的全部 5358 张训练图与 1061 张测试图，新增 20 张独立来源白色缺陷母图的
240 张局部旋转图，仅补充验证集至 1301 张，快照总计 7720 张。新增图参考旧验证集中 20 个不同来源组，
与训练来源组隔离；10 张母图为小尺寸内部缺陷，10 张为近边缘内部缺陷，后者保留明显灰色间隔。
母图及其全部角度、参考来源必须保持在验证集合；生成更多图片不等于增加真实实物。

两组都从原始预训练权重开始，使用 AdamW、`momentum=0.9`、`warmup_bias_lr=0.0`，只比较
`lr0=0.00125` 与 `lr0=0.0003`。每组的 `best.pt` 由同一验证集选择；再按验证 Top-1 选择候选，
同分时选较低学习率，固定权重校验值后才评估测试集。100 轮上限、提前停止 20 轮及其他既有参数不变。

```powershell
python disc.py --workspace disc-experiment experiment train --known-discs --data disc-experiment/datasets/v16_validation_lr_20261023 --weights yolo11n-cls.pt --run teammate_v16_lr125_001 --epochs 100 --device 0 --optimizer AdamW --lr0 0.00125 --momentum 0.9 --warmup-bias-lr 0.0
python disc.py --workspace disc-experiment experiment train --known-discs --data disc-experiment/datasets/v16_validation_lr_20261023 --weights yolo11n-cls.pt --run teammate_v16_lr030_001 --epochs 100 --device 0 --optimizer AdamW --lr0 0.0003 --momentum 0.9 --warmup-bias-lr 0.0
```

短程检查改为 `--epochs 2` 并使用新运行名称；CPU 改为 `--device cpu`。
验证补充图和提示词位于本地 `dataset/synthetic/white_validation_independent_v1_20261023/`，
数据与参数审计位于 `disc-experiment/checks/validation_lr_v16_20261023/`。
真实圆盘仍只有原有 12 个，且仅有 L0；既有测试已反复查看，此比较属于已知圆盘研究，不能声称盲测或新实物泛化。
训练、验证、测试及历史模型的实际来源暴露必须在报告中分别列出。

## 实现边界

随机按照片划分的现有工具不能保证实物隔离，也不保存追加采集的光照信息。本项目只增加一个独立脚本，复用 Ultralytics 训练、预测和混淆矩阵绘图，以及 Pillow 的补边/增强功能。登记、版本快照和分组报告集中在同一个入口，避免修改框架或创建新的训练循环。

仓库只提交源码和说明。照片、人工登记、预览、数据集、权重和实验报告保留在本地实验目录；请勿将原始照片或权重提交到 Git。

验证覆盖实物跨集合冲突、未确认身份、未登记新增图、不同光照继承归属、旧版本保护及原图像素保留。另使用独立生成的技术验证样本完成两轮真实 GPU 训练、权重重载、单张/批量预测和双模型光照报告核对；这些样本和权重不用于判断现有圆盘的识别效果。用户数据的正式训练须先完成人工实物分组。
