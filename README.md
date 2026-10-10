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

## 实现边界

随机按照片划分的现有工具不能保证实物隔离，也不保存追加采集的光照信息。本项目只增加一个独立脚本，复用 Ultralytics 训练、预测和混淆矩阵绘图，以及 Pillow 的补边/增强功能。登记、版本快照和分组报告集中在同一个入口，避免修改框架或创建新的训练循环。

仓库只提交源码和说明。照片、人工登记、预览、数据集、权重和实验报告保留在本地实验目录；请勿将原始照片或权重提交到 Git。

验证覆盖实物跨集合冲突、未确认身份、未登记新增图、不同光照继承归属、旧版本保护及原图像素保留。另使用独立生成的技术验证样本完成两轮真实 GPU 训练、权重重载、单张/批量预测和双模型光照报告核对；这些样本和权重不用于判断现有圆盘的识别效果。用户数据的正式训练须先完成人工实物分组。
