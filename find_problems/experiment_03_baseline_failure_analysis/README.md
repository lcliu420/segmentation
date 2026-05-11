# 实验三：Baseline 逐图失败分析

本实验用于训练代表性 baseline，并在 test 集上保存逐图预测结果和逐图完整指标。它不是为了做模型排行榜，而是为了回答：不同模型是否会在同一批样本上失败，这些失败是否与实验二发现的边界频率差异有关。

重要约束：`../dataset/` 严格只读。不要在 `dataset/` 内新建、删除、重命名、移动或覆盖任何文件。实验三的脚本、日志、checkpoint、预测 mask、CSV 和后续可视化都只能写入 `experiment_03_baseline_failure_analysis/` 内。

## 目标

实验三的核心产物不是单个平均 Dice，而是逐图失败分析表。

每个 baseline 都应在最终测试阶段保存：

```text
image, modal, split, model, Dice, IoU, Boundary_IoU, HD95, MAE, pred_fg_ratio, gt_fg_ratio
```

后续将这些逐图指标与实验二的边界频率指标合并：

```text
experiment_02_scale_vs_boundary_frequency/outputs/boundary_frequency_metrics.csv
```

重点验证：

- `scale` 是否不是导致模型失败的主要因素。
- `global_hfr` 是否能解释部分模态差异。
- `boundary_hfr / boundary_freq_std / boundary_freq_gap` 是否更能解释 `Boundary_IoU` 下降和 `HD95` 升高。
- 多个 baseline 是否在同一批边界频率异常样本上共同失败。

## 目录结构

```text
experiment_03_baseline_failure_analysis/
  U-Net/
  U-Net++/
  Attention U-Net/
  Swin-Unet/
  outputs/
    runs/
    prediction_metrics/
    merged_metrics/
    failure_cases/
  predictions/
    <model>/<modal>/test/
  figures/
  visual_checks/
  logs/
```

当前已经完成训练与测试的是：

```text
Swin-Unet/
U-Net/
U-Net++/
Attention U-Net/
```

这四个 baseline 均已完成 WL/NBI 两个模态的正式测试，输出已经同步到当前目录的 `outputs/` 和 `predictions/`。其他 baseline 后续如需扩展，继续按同一套输出规范补齐。

## 标准训练流程

训练阶段只记录轻量指标：

```text
train: loss
val: loss, Dice, IoU
checkpoint: best val Dice
```

不建议每个 epoch 计算：

```text
Boundary IoU / HD95 / MAE
```

尤其是 `HD95`，它更适合放在最终测试阶段逐图计算。

## 运行命令：Swin-Unet

以下命令需要在 `experiment_03_baseline_failure_analysis/Swin-Unet` 目录下运行：

```bash
cd experiment_03_baseline_failure_analysis/Swin-Unet
```

正式训练前先确认服务器上的数据划分是否为本实验要求的 `1960/420/420`：

```bash
find ../../dataset/WL/train/images -type f | wc -l
find ../../dataset/WL/val/images -type f | wc -l
find ../../dataset/WL/test/images -type f | wc -l
find ../../dataset/NBI/train/images -type f | wc -l
find ../../dataset/NBI/val/images -type f | wc -l
find ../../dataset/NBI/test/images -type f | wc -l
```

快速 smoke test：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 1 --batch_size 2 --num_workers 0 --limit_samples 2 --run_name smoke_swin_unet_WL --model_name swin_unet_smoke
```

WL 正式训练：

```bash
python train.py --modal WL --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name swin_unet_WL --model_name swin_unet
```

NBI 正式训练：

```bash
python train.py --modal NBI --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name swin_unet_NBI --model_name swin_unet
```

WL 正式测试：

```bash
python test.py --modal WL --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir ../outputs/runs/swin_unet_WL --checkpoint ../outputs/runs/swin_unet_WL/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name swin_unet_WL --model_name swin_unet
```

NBI 正式测试：

```bash
python test.py --modal NBI --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir ../outputs/runs/swin_unet_NBI --checkpoint ../outputs/runs/swin_unet_NBI/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name swin_unet_NBI --model_name swin_unet
```

## 运行命令：U-Net

以下命令需要在 `experiment_03_baseline_failure_analysis/U-Net` 目录下运行：

```bash
cd experiment_03_baseline_failure_analysis/U-Net
```

正式训练前同样需要确认服务器上的数据划分是否为本实验要求的 `1960/420/420`：

```bash
find ../../dataset/WL/train/images -type f | wc -l
find ../../dataset/WL/val/images -type f | wc -l
find ../../dataset/WL/test/images -type f | wc -l
find ../../dataset/NBI/train/images -type f | wc -l
find ../../dataset/NBI/val/images -type f | wc -l
find ../../dataset/NBI/test/images -type f | wc -l
```

快速 smoke test：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 1 --batch_size 2 --num_workers 0 --limit_samples 2 --run_name smoke_unet_WL --model_name unet_smoke
```

WL 正式训练：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name unet_WL --model_name unet
```

NBI 正式训练：

```bash
python train.py --modal NBI --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name unet_NBI --model_name unet
```

WL 正式测试：

```bash
python test.py --modal WL --data_root ../../dataset --output_dir ../outputs/runs/unet_WL --checkpoint ../outputs/runs/unet_WL/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name unet_WL --model_name unet
```

NBI 正式测试：

```bash
python test.py --modal NBI --data_root ../../dataset --output_dir ../outputs/runs/unet_NBI --checkpoint ../outputs/runs/unet_NBI/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name unet_NBI --model_name unet
```

## 运行命令：U-Net++

以下命令需要在 `experiment_03_baseline_failure_analysis/U-Net++` 目录下运行：

```bash
cd experiment_03_baseline_failure_analysis/U-Net++
```

正式训练前同样需要确认服务器上的数据划分是否为本实验要求的 `1960/420/420`：

```bash
find ../../dataset/WL/train/images -type f | wc -l
find ../../dataset/WL/val/images -type f | wc -l
find ../../dataset/WL/test/images -type f | wc -l
find ../../dataset/NBI/train/images -type f | wc -l
find ../../dataset/NBI/val/images -type f | wc -l
find ../../dataset/NBI/test/images -type f | wc -l
```

快速 smoke test：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 1 --batch_size 2 --num_workers 0 --limit_samples 2 --run_name smoke_unetpp_WL --model_name unetpp_smoke
```

WL 正式训练：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name unetpp_WL --model_name unetpp
```

NBI 正式训练：

```bash
python train.py --modal NBI --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name unetpp_NBI --model_name unetpp
```

WL 正式测试：

```bash
python test.py --modal WL --data_root ../../dataset --output_dir ../outputs/runs/unetpp_WL --checkpoint ../outputs/runs/unetpp_WL/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name unetpp_WL --model_name unetpp
```

NBI 正式测试：

```bash
python test.py --modal NBI --data_root ../../dataset --output_dir ../outputs/runs/unetpp_NBI --checkpoint ../outputs/runs/unetpp_NBI/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name unetpp_NBI --model_name unetpp
```

## 运行命令：Attention U-Net

以下命令需要在 `experiment_03_baseline_failure_analysis/Attention U-Net` 目录下运行：

```bash
cd "experiment_03_baseline_failure_analysis/Attention U-Net"
```

正式训练前同样需要确认服务器上的数据划分是否为本实验要求的 `1960/420/420`：

```bash
find ../../dataset/WL/train/images -type f | wc -l
find ../../dataset/WL/val/images -type f | wc -l
find ../../dataset/WL/test/images -type f | wc -l
find ../../dataset/NBI/train/images -type f | wc -l
find ../../dataset/NBI/val/images -type f | wc -l
find ../../dataset/NBI/test/images -type f | wc -l
```

快速 smoke test：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 1 --batch_size 2 --num_workers 0 --limit_samples 2 --run_name smoke_attention_unet_WL --model_name attention_unet_smoke
```

WL 正式训练：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name attention_unet_WL --model_name attention_unet
```

NBI 正式训练：

```bash
python train.py --modal NBI --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name attention_unet_NBI --model_name attention_unet
```

WL 正式测试：

```bash
python test.py --modal WL --data_root ../../dataset --output_dir ../outputs/runs/attention_unet_WL --checkpoint ../outputs/runs/attention_unet_WL/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name attention_unet_WL --model_name attention_unet
```

NBI 正式测试：

```bash
python test.py --modal NBI --data_root ../../dataset --output_dir ../outputs/runs/attention_unet_NBI --checkpoint ../outputs/runs/attention_unet_NBI/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name attention_unet_NBI --model_name attention_unet
```

## 标准测试流程

对 best checkpoint 在 test 集逐图推理，保存预测 mask，并计算完整指标：

```text
Dice
IoU
Boundary_IoU
HD95
MAE
```

标准输出位置：

```text
predictions/<model>/<modal>/test/*.png
outputs/prediction_metrics/<model>_<modal>_prediction_metrics.csv
```

例如 Swin-Unet：

```text
predictions/swin_unet/WL/test/*.png
predictions/swin_unet/NBI/test/*.png
outputs/prediction_metrics/swin_unet_WL_prediction_metrics.csv
outputs/prediction_metrics/swin_unet_NBI_prediction_metrics.csv
```

## 运行命令：实验三分析

四个 baseline 的测试结果下载完成后，在项目根目录运行：

```bash
python -B experiment_03_baseline_failure_analysis/scripts/analyze_experiment_03.py
```

该脚本只读取 `dataset/`、`predictions/`、`outputs/prediction_metrics/` 和实验二频率 CSV，所有新结果只写入实验三目录内。

## 当前输出完整性检查

四个 baseline 已经在正确数据划分上完成训练和测试：

```text
WL  train/val/test = 1960/420/420
NBI train/val/test = 1960/420/420
```

当前输出完整性如下：

```text
models: swin_unet, unet, unetpp, attention_unet
modalities: WL, NBI
each prediction folder: 420 png
each prediction_metrics csv: 420 rows
frequency overlap: 420/420 for all model-modal pairs
```

8 个 prediction CSV 均可与实验二 test 频率表按 `image + modal + split` 完整合并：

```text
swin_unet      WL/NBI overlap: 420/420
unet           WL/NBI overlap: 420/420
unetpp         WL/NBI overlap: 420/420
attention_unet WL/NBI overlap: 420/420
```

## 正式结果摘要

### Test Mean Metrics

```text
WL:
  swin_unet      Dice 0.791462  IoU 0.681421  Boundary_IoU 0.201879  HD95 36.203175  MAE 0.108769
  attention_unet Dice 0.751591  IoU 0.635608  Boundary_IoU 0.174353  HD95 49.631193  MAE 0.123141
  unet           Dice 0.750929  IoU 0.631725  Boundary_IoU 0.161590  HD95 48.321444  MAE 0.128977
  unetpp         Dice 0.745571  IoU 0.629182  Boundary_IoU 0.167322  HD95 49.012759  MAE 0.125398

NBI:
  swin_unet      Dice 0.816045  IoU 0.711552  Boundary_IoU 0.215963  HD95 34.966517  MAE 0.114859
  unet           Dice 0.789900  IoU 0.685464  Boundary_IoU 0.201321  HD95 41.640963  MAE 0.132174
  attention_unet Dice 0.783209  IoU 0.677139  Boundary_IoU 0.188848  HD95 43.156168  MAE 0.130808
  unetpp         Dice 0.777736  IoU 0.670904  Boundary_IoU 0.187092  HD95 43.216310  MAE 0.129345
```

### Best Val Metrics

```text
WL:
  swin_unet      best epoch 96  best val Dice 0.796558  best val IoU 0.682884
  attention_unet best epoch 76  best val Dice 0.752802  best val IoU 0.632902
  unet           best epoch 65  best val Dice 0.748224  best val IoU 0.628678
  unetpp         best epoch 69  best val Dice 0.740360  best val IoU 0.621493

NBI:
  swin_unet      best epoch 37  best val Dice 0.822336  best val IoU 0.718049
  attention_unet best epoch 66  best val Dice 0.793860  best val IoU 0.686576
  unet           best epoch 76  best val Dice 0.790044  best val IoU 0.682888
  unetpp         best epoch 61  best val Dice 0.783090  best val IoU 0.674690
```

### Failure Counts

```text
WL:
  swin_unet      Dice<0.5 26  Boundary_IoU<0.1 95   HD95>100 14
  attention_unet Dice<0.5 44  Boundary_IoU<0.1 121  HD95>100 35
  unet           Dice<0.5 43  Boundary_IoU<0.1 128  HD95>100 32
  unetpp         Dice<0.5 43  Boundary_IoU<0.1 134  HD95>100 29

NBI:
  swin_unet      Dice<0.5 18  Boundary_IoU<0.1 103  HD95>100 10
  unet           Dice<0.5 36  Boundary_IoU<0.1 120  HD95>100 25
  attention_unet Dice<0.5 33  Boundary_IoU<0.1 125  HD95>100 29
  unetpp         Dice<0.5 34  Boundary_IoU<0.1 124  HD95>100 24
```

### 共同失败样本

```text
WL:
  common Dice<0.5: 13
  any Dice<0.5: 73
  common Boundary_IoU<0.1: 39
  common HD95>100: 6

NBI:
  common Dice<0.5: 12
  any Dice<0.5: 52
  common Boundary_IoU<0.1: 56
  common HD95>100: 3
```

### 高 Dice 低边界质量现象

```text
WL:
  Dice>=0.85 & Boundary_IoU<0.1: 24 rows / 18 images
  Dice>=0.85 & HD95>50: 49 rows / 40 images

NBI:
  Dice>=0.85 & Boundary_IoU<0.1: 24 rows / 17 images
  Dice>=0.85 & HD95>50: 53 rows / 39 images
```

### 初步解读

- NBI 上四个模型的 Dice/IoU 整体高于 WL，说明 NBI 的整体区域分割相对更容易。
- Swin-Unet 在 WL/NBI 上均取得当前最高 Dice、最高 Boundary IoU 和最低 HD95，说明全局建模确实带来了一定收益。
- 但四个模型的 Boundary IoU 都明显偏低，且低边界指标样本数量不少，说明边界质量仍然是关键问题。
- Attention U-Net 没有明显减少背景或边界失败，提示普通 attention 不一定能解决当前数据中的边界频率与伪边界干扰问题。
- NBI 虽然区域指标更高，但 `Boundary_IoU < 0.1` 的样本数并不少，说明高频纹理不等价于稳定边界。

## 实验三分析产物

当前四个 baseline 的逐图预测结果已经按以下键与实验二频率指标完整合并：

```text
image + modal + split
```

合并输入为：

```text
experiment_02_scale_vs_boundary_frequency/outputs/boundary_frequency_metrics.csv
experiment_03_baseline_failure_analysis/outputs/prediction_metrics/*.csv
```

当前已验证合并情况：

```text
all model-modal pairs overlap: 420/420
```

已生成的核心产物：

```text
outputs/merged_metrics/
  experiment_03_merged_metrics.csv
  model_modal_summary.csv
  correlation_by_model_modal.csv
  frequency_group_summary.csv
  analysis_summary.txt

outputs/failure_cases/
  all_low_dice_cases.csv
  all_low_boundary_iou_cases.csv
  all_high_hd95_cases.csv
  common_failures_by_modal.csv
  high_dice_low_boundary_cases.csv
  high_dice_high_hd95_cases.csv

figures/
visual_checks/
```

生成规模：

```text
experiment_03_merged_metrics.csv: 3360 rows
each model-modal pair: 420 rows
all_low_dice_cases.csv: 277 rows
all_low_boundary_iou_cases.csv: 950 rows
all_high_hd95_cases.csv: 198 rows
high_dice_low_boundary_cases.csv: 48 rows
high_dice_high_hd95_cases.csv: 102 rows
common_failures_by_modal.csv: 104 images
```

主要图表：

![Model Modal Metric Summary](figures/model_modal_metric_summary.png)

图说明：该图汇总四个模型在 WL/NBI test 集上的 `Dice`、`IoU`、`Boundary IoU`、`HD95` 和 `MAE` 均值。

结果解读：NBI 的 `Dice/IoU` 整体高于 WL，说明 NBI 的区域分割相对更容易；Swin-Unet 在两个模态上均取得当前最好的整体指标和边界指标。但即使是 Swin-Unet，`Boundary IoU` 也只有约 `0.20-0.22`，明显低于区域重叠指标，说明当前模型主要不是完全找不到病灶，而是边界质量仍然偏弱。

![Failure Counts](figures/failure_counts_by_model_modal.png)

图说明：该图比较三类失败样本数量：`Dice < 0.5`、`Boundary_IoU < 0.1`、`HD95 > 100`。

结果解读：低 `Boundary IoU` 样本数量明显多于低 `Dice` 样本，说明很多样本虽然区域重叠不算完全失败，但边界窄带质量已经很差。NBI 的整体 Dice 更高，但低 `Boundary IoU` 样本并不少，提示 NBI 的高频纹理并不天然等价于稳定边界。

![Correlation Heatmap](figures/correlation_heatmap_spearman.png)

图说明：该图展示频率/尺度指标与模型预测指标之间的平均 Spearman 相关性，相关性来自四个 baseline 和两个模态的逐图结果。

结果解读：`scale-Dice` 相关性较强，说明病灶面积仍然影响区域指标；但 `global_hfr-Dice` 接近无关，说明全图频率不是模型失败的直接解释变量。相比之下，`boundary_freq_gap` 与 `Boundary_IoU` 的关系更明显，提示边界局部频率差异比全图频率更贴近边界质量。不过 `boundary_freq_std-HD95` 当前相关性较弱，因此还不能直接写成“边界频率波动导致 HD95 升高”，需要结合区域错误拆分和可视化继续验证。

![Boundary Frequency vs Boundary IoU](figures/boundary_frequency_vs_boundary_iou.png)

图说明：该图从散点层面展示 `boundary_hfr`、`boundary_freq_gap` 与 `Boundary_IoU` 的关系，并区分 WL/NBI。

结果解读：散点呈现一定趋势，但离散度较大，说明单个边界频率指标不能独立解释全部边界失败。这个结果更适合支撑一个谨慎结论：边界频率信息与边界质量有关，但后续方法或论文论证不能只依赖全局相关性，还需要结合共同失败样本、错误区域拆分和可视化证据。

![High Dice Low Boundary Distribution](figures/high_dice_low_boundary_distribution.png)

图说明：该图展示 `Dice` 与 `Boundary_IoU` 的关系，并突出 `Dice >= 0.85` 但 `Boundary_IoU < 0.1` 的样本。

结果解读：图中存在一批高 Dice 但低 Boundary IoU 的点，说明区域重叠指标可能掩盖边界错误。由于本数据集中大病灶较多，模型只要覆盖到大体区域就可能得到较高 Dice，但边界窄带仍可能发生外扩、内缩或局部偏移。这是后续强调 Boundary IoU、HD95 和边界感知方法的重要依据。

代表性可视化：

![WL Common Failures](visual_checks/common_failures_WL.png)

图说明：该图展示 WL 中四个模型共同失败的代表性样本，包括原图/GT 和四个模型预测。

结果解读：共同失败样本的价值在于排除“某一个模型偶然没学好”的解释。如果 U-Net、U-Net++、Attention U-Net 和 Swin-Unet 都在同一批 WL 图像上边界质量较差，说明这些图像更可能包含数据层面的系统性困难，例如低对比、黏膜相似背景、边界过渡弱或局部模糊。

![NBI Common Failures](visual_checks/common_failures_NBI.png)

图说明：该图展示 NBI 中四个模型共同失败的代表性样本。

结果解读：NBI 共同失败样本说明，即便 NBI 整体区域指标高于 WL，强纹理和高频结构也不一定带来稳定边界。部分失败更可能来自血管纹理、反光或暗背景引入的伪边界，使模型在边界附近产生不一致预测。

![WL High Dice Low Boundary](visual_checks/high_dice_low_boundary_WL.png)

图说明：该图展示 WL 中 Dice 较高但 Boundary IoU 较低的样本。

结果解读：这类样本通常不是“病灶完全漏检”，而是模型已经覆盖了主要病灶区域，但轮廓不准。对于 WL，这种现象常对应边界外扩或内缩：正常黏膜与病灶颜色接近时，模型容易把相邻背景也纳入预测，或者在弱边界处丢失局部轮廓。

![NBI High Dice Low Boundary](visual_checks/high_dice_low_boundary_NBI.png)

图说明：该图展示 NBI 中 Dice 较高但 Boundary IoU 较低的样本。

结果解读：NBI 的高 Dice 低边界样本说明，纹理清晰并不等于边界准确。模型可能依靠强对比定位到病灶大体区域，但在血管纹理、反光和局部高频扰动附近仍出现边界断裂、外扩或局部偏移。

![WL High HD95](visual_checks/high_hd95_WL.png)

图说明：该图展示 WL 中 `HD95` 较高的代表性样本。

结果解读：`HD95` 对局部远距离边界偏移很敏感。WL 中高 HD95 样本提示，即使整体区域指标尚可，只要局部边界出现明显外扩、内缩或远端误分，HD95 就会显著升高。这类样本适合用于展示“Dice 不足以描述边界定位质量”。

![NBI High HD95](visual_checks/high_hd95_NBI.png)

图说明：该图展示 NBI 中 `HD95` 较高的代表性样本。

结果解读：NBI 高 HD95 样本往往更适合观察局部伪边界和远端误差。强纹理或反光可能诱发局部预测偏移，使边界距离指标明显变差。当前结果只能说明这类现象存在，是否由具体频率扰动导致，还需要后续区域错误拆分或扰动敏感性实验继续验证。

## 当前分析解读

已经完成的分析支持以下阶段性判断：

- 四个模型存在共同失败样本：WL 有 `45` 张图进入共同失败清单，NBI 有 `59` 张图进入共同失败清单。
- 四模型共同低 Boundary IoU 的样本为 WL `39` 张、NBI `56` 张，说明边界质量问题不是单一模型偶然失败。
- 高 Dice 低边界质量现象明确存在：WL 为 `24` 行 / `18` 张图，NBI 为 `24` 行 / `17` 张图。
- 从平均 Spearman 看，`scale-Dice` 相关性较强，说明病灶面积仍会影响区域指标；但 `global_hfr-Dice` 近乎无关，说明全图频率不是失败的直接解释变量。
- `boundary_freq_gap-Boundary_IoU` 平均相关性约为 `0.2492`，比 `global_hfr-Boundary_IoU` 更明显，提示边界局部频率差异比全图频率更贴近边界质量。
- `boundary_freq_std-HD95` 当前相关性较弱，不能直接写成“边界频率波动导致 HD95 升高”；后续需要结合区域错误拆分和可视化再判断。
- 当前更稳妥的结论是：边界质量问题真实存在，并且具有多模型共享性；边界频率指标提供了一个有价值的解释方向，但还不足以单独完成因果论证。

后续如果继续深化实验三，应重点分析：

- `scale` 与 `Dice / Boundary_IoU / HD95` 的关系。
- `global_hfr` 与 `Dice / Boundary_IoU / HD95` 的关系。
- `boundary_hfr` 与 `Boundary_IoU / HD95` 的关系。
- `boundary_freq_std` 与 `HD95` 的关系。
- `boundary_freq_gap` 与 `Boundary_IoU` 的关系。

重点筛选三类样本：

```text
低 Dice 失败样本
高 Dice 但低 Boundary IoU 样本
高 HD95 边界偏移样本
```

如果多个模型在同一批样本上失败，并且这些样本具有异常的 `boundary_hfr / boundary_freq_std / boundary_freq_gap`，就可以更有力地说明：当前胃镜病灶分割的瓶颈不只是区域识别，而是边界频率不稳定条件下的精细轮廓定位。
