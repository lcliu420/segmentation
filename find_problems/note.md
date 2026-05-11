# 胃镜病灶分割：从 MADGNet Figure 1 到边界频率的问题发现实验

## 重要警醒：dataset 只读

当前文件目录下的 `dataset` 文件夹里的东西不准做任何修改。

后续所有实验必须遵守：

- `dataset/` 只允许读取，不允许写入。
- 不在 `dataset/` 内新建、删除、重命名、移动或覆盖任何文件。
- 所有实验脚本、CSV、图片、日志、模型预测、缓存都必须写入各自的实验文件夹。
- 每个实验单独新建 `experiment_XX_*/` 目录，输出统一放到该实验目录下的 `outputs/`、`figures/`、`logs/` 或类似子目录。
- 临时缓存也必须写入实验目录内，例如 `.mplconfig/`，不能写入 `dataset/`。

## 0. 研究出发点

当前阶段不要急着提出“创新点”，而是先做问题发现实验。目标是从 WL/NBI 胃镜病灶分割数据本身出发，回答几个更具体的问题：

- 这个数据集的困难到底是不是尺度变化？
- 如果不是尺度变化，那么全图频率、局部频率、边界频率中，哪一种更能解释模型失败？
- WL 和 NBI 的失败机制是否不同？
- 主流模型是否在同一批边界频率异常样本上共同失败？

参考论文《Modality-agnostic domain generalizable medical image segmentation by multi-frequency in multi-scale attention》的 Figure 1 逻辑：作者先统计不同医学模态中的 lesion scale distribution 和 frequency distribution，再观察到 frequency variance 比 scale variance 更明显，因此认为只关注多尺度不够，还要关注多频率信息。

本数据集可以借用这个“先统计现象，再推出方法必要性”的思路，但不能简单照搬。胃镜 WL/NBI 数据的重点不只是全图频率，而可能是病灶边界区域的频率不稳定：

- WL 中，模糊、低对比和正常黏膜相似性可能削弱真实边界的高频线索。
- NBI 中，血管纹理、暗背景和反光可能引入伪高频，干扰真实边界判断。
- 如果边界频率指标比 scale 更能解释 Boundary IoU 和 HD95 的下降，那么后续方法就可以围绕“边界频率感知”展开，而不是笼统说“边界模糊”。

第一版实验只做问题发现，不提前确定最终创新点。

---

## 1. 实验一：Scale vs Global Frequency 分布

### 目的

完全参考 MADGNet Figure 1 的基础逻辑，先比较 WL/NBI 中病灶尺度分布和全图频率分布。

这一步要回答：

- WL/NBI 的 lesion scale 是否真的变化很大？
- 全图 frequency distribution 是否比 scale distribution 更能区分 WL/NBI？
- 如果只做多尺度建模，是否可能没有抓住本数据集的主要变化来源？

### 指标定义

`scale` 定义为：

```text
scale = mask foreground pixels / total pixels
```

`global_hfr` 定义为：

```text
global_hfr = high-frequency power / full-frequency power
```

其中 `hfr` 表示 high-frequency ratio。

### 计算方式

对每张图像执行：

1. 将图像 resize 到 `256x256`。
2. 转为灰度图。
3. 将灰度值归一化到 `[0, 1]`。
4. 使用 2D FFT 得到频谱。
5. 计算功率谱。
6. 将频谱中心化。
7. 以频谱半径的外侧 `1/3` 区域作为高频区域。
8. 计算高频功率与全频功率之比。

默认先用 FFT 做统计，因为它更适合直接做功率谱比例。后续如果要和模型模块结合，可以再考虑 DCT。

### 输出文件

```text
analysis_outputs/
  frequency_metrics.csv
  figures/
    scale_vs_global_frequency.png
```

`frequency_metrics.csv` 至少包含：

```text
image, modal, split, scale, global_hfr
```

### 图表设计

画一张类似 MADGNet Figure 1 的散点图：

- x 轴：`scale`
- y 轴：`global_hfr`
- 颜色：`WL` / `NBI`
- 可额外标注每个模态的均值和方差

同时统计：

```text
var(scale) for WL
var(scale) for NBI
var(global_hfr) for WL
var(global_hfr) for NBI
```

以及 WL/NBI 之间的均值差异。

### 期望观察

如果 WL/NBI 的 scale 分布相对集中，但 global frequency 分布差异更明显，说明本数据集的主要变化可能不是病灶大小，而是成像频率特征。

可能形成的问题描述：

```text
与尺度变化相比，WL/NBI 胃镜图像在频率分布上表现出更明显的模态差异，提示仅依赖多尺度建模可能不足以处理该数据集中的成像变化。
```

---

## 2. 实验二：Scale vs Boundary Frequency 分布

### 目的

把 MADGNet 的全图频率思想改造成更贴合胃镜病灶分割的问题定义：病灶边界附近的频率分布是否比全图频率更关键。

这一步要回答：

- 边界频率是否比全图频率更能体现 WL/NBI 的差异？
- 边界区域是否存在明显的频率不稳定？
- WL 的边界是不是偏低频、模糊？
- NBI 的边界是不是高频更强，但受纹理和反光干扰更明显？

### 区域划分

基于 GT mask 将图像划分为三个区域：

```text
interior         = mask 腐蚀后的病灶内部
boundary_band    = mask 膨胀区域 - mask 腐蚀区域
near_background  = mask 外侧邻近背景环带
```

默认参数：

```text
image_size = 256x256
erode_dilate_radius = 5 px
max_boundary_points = 256
patch_size = 32x32
```

边界点采样方式：

1. 将 mask resize 到 `256x256` 后二值化。
2. 用腐蚀和膨胀得到 `boundary_band`。
3. 从 `boundary_band` 中均匀采样最多 `256` 个中心点。
4. 以每个中心点裁剪 `32x32` patch。
5. 对每个 patch 计算高频功率比例。
6. 对所有 patch 的高频比例求均值和标准差。

### 统计指标

每张图至少计算：

```text
interior_hfr
boundary_hfr
near_background_hfr
boundary_freq_std
boundary_freq_gap
```

其中：

```text
boundary_freq_gap =
  abs(boundary_hfr - interior_hfr)
  + abs(boundary_hfr - near_background_hfr)
```

解释：

- `boundary_hfr`：边界区域整体高频强度。
- `boundary_freq_std`：边界 patch 之间的高频波动，表示边界频率是否稳定。
- `boundary_freq_gap`：边界与内部、邻近背景的频率差异。如果该值异常，可能说明边界频率结构混乱。

### 输出文件

继续写入：

```text
analysis_outputs/frequency_metrics.csv
```

新增字段：

```text
interior_hfr,
boundary_hfr,
near_background_hfr,
boundary_freq_std,
boundary_freq_gap
```

新增图：

```text
analysis_outputs/
  figures/
    scale_vs_boundary_frequency.png
    wl_nbi_boundary_frequency_distribution.png
```

### 图表设计

图一：`scale_vs_boundary_frequency.png`

- x 轴：`scale`
- y 轴：`boundary_hfr`
- 颜色：`WL` / `NBI`

图二：`wl_nbi_boundary_frequency_distribution.png`

- 对比 WL/NBI 的 `boundary_hfr`
- 对比 WL/NBI 的 `boundary_freq_std`
- 对比 WL/NBI 的 `boundary_freq_gap`

### 期望观察

可能出现几种情况：

1. WL 的 `boundary_hfr` 较低  
   说明 WL 边界高频线索弱，可能与低对比、模糊、正常黏膜相似有关。

2. NBI 的 `boundary_hfr` 较高，但 `boundary_freq_std` 也较高  
   说明 NBI 边界附近存在更多高频变化，但这些高频不一定都是真实边界，可能来自血管纹理和反光。

3. `boundary_hfr` 或 `boundary_freq_std` 的模态差异比 `scale` 更明显  
   可以提出更具体的问题：WL/NBI 的主要差异不仅是区域大小，而是边界频率分布差异。

可能形成的问题描述：

```text
WL/NBI 胃镜病灶图像的尺度分布并不能充分解释分割困难。相比之下，病灶边界区域的频率分布在不同模态中表现出更明显差异，提示边界频率不稳定可能是影响模型边界定位的重要因素。
```

---

## 3. 实验三：Baseline 逐图失败分析

### 当前状态

实验三第一批 baseline 已经完成训练与测试，并且已经完成训练后分析：`merged_metrics`、失败样本清单、相关性统计图和代表性失败可视化均已生成。当前不再是“待跑 baseline”的阶段，而是进入“基于结果提炼问题表述”的阶段。

已经完成的四个模型为：

```text
U-Net
U-Net++
Attention U-Net
Swin-Unet
```

当前已确认：

```text
models: swin_unet, unet, unetpp, attention_unet
modalities: WL, NBI
each prediction folder: 420 png
each prediction_metrics csv: 420 rows
frequency overlap: 420/420 for all model-modal pairs
```

这说明四个 baseline 的测试输出已经与实验二的 `boundary_frequency_metrics.csv` 按 `image + modal + split` 完整合并。当前统一 `merged_metrics`、共同失败清单和代表性失败可视化已经生成。

### 目的

前两组实验只说明数据中存在全图频率和边界频率差异，还不能说明这些差异是否真的会导致模型失败。因此实验三要训练代表性 baseline，并在最终测试阶段保存逐图预测结果和逐图完整指标。

这一步不是为了证明哪个模型平均 Dice 最高，而是为了找失败规律：

- 不同结构是否在同一批样本上失败？
- 共同失败样本是否具有异常边界频率？
- WL/NBI 的失败机制是否不同？
- 频率指标是否比 `scale` 更能解释 `Boundary_IoU` 下降或 `HD95` 升高？
- Transformer、Mamba、强语义模型或 skip 融合改进是否真的能缓解边界频率问题？

核心目标可以写成：

```text
验证实验一/二发现的频率差异，尤其是边界频率不稳定，是否会对应到模型逐图失败。
```

### Baseline 执行顺序

不要一开始就铺太大。先用基础模型把“训练、测试、保存预测、逐图指标、频率合并”的流程跑通，再扩展到更复杂模型。

当前已经完成第一批基础 CNN/attention baseline，并额外完成 Swin-Unet：

```text
U-Net
U-Net++
Attention U-Net
Swin-Unet
```

后续如果还需要补充 DeepLabV3+、PraNet、U-Mamba 或 U-Net + SDI，应优先服务于“验证问题是否仍然存在”，而不是继续做模型排行榜。

第一批 baseline：

| 模型 | 作用 |
| --- | --- |
| U-Net | 最基础 CNN baseline |
| U-Net++ | 看更密集 skip connection 是否改善边界 |
| Attention U-Net | 看注意力是否减少背景干扰 |

第二批 baseline：

| 模型 | 作用 |
| --- | --- |
| DeepLabV3+ 或 PraNet | 看强语义/医学分割模型表现 |
| Swin-Unet 或 U-Mamba | 看 Transformer/Mamba 类全局建模是否有帮助 |
| U-Net + SDI | 看 SDI 是否改善 skip 融合 |

如果第一批模型已经在同一批样本上出现一致性失败，就可以优先进入失败归因分析；第二批模型用于验证这种失败是否仍然存在。

### 训练阶段指标策略

第一批四个 baseline 已经在 GPU 服务器上完成正式训练与测试。后续新增 baseline 或改进模型仍建议放在 GPU 服务器上跑；本机主要负责整理输出、合并指标、生成失败样本清单和可视化。

训练阶段不要每个 epoch 都计算重指标。尤其是 `HD95`，计算成本较高，不适合每轮训练都算。

推荐训练记录方式：

```text
train/val/test split: 使用当前 dataset/WL 和 dataset/NBI
input_size: 统一，例如 256 或 512
loss: Dice + BCE
train 每个 epoch: loss，可选轻量 Dice
val 每个 epoch: loss, Dice, IoU
checkpoint selection: best val Dice
```

不建议每个 epoch 都计算：

```text
Boundary IoU
HD95
MAE
```

如果训练时确实想观察边界变化，可以低频率计算：

```text
每 10 个 epoch 在 val 上计算一次 Boundary IoU
HD95 只在 best checkpoint 和 final test 上计算
或只在固定 val 子集上计算 Boundary IoU / HD95
```

### 最终测试阶段

实验三真正需要的是最终测试阶段的逐图完整指标，而不是训练过程每轮的重指标。

对每个 baseline 的 best checkpoint，在 test 集上逐图计算：

```text
Dice
IoU
Boundary_IoU
HD95
MAE
```

同时保存每张 test 图的预测 mask。

逐图指标表：

```text
experiment_03_baseline_failure_analysis/
  outputs/
    prediction_metrics/
      prediction_metrics.csv
```

每一行是一张图在一个模型下的结果：

```text
image, modal, split, model, Dice, IoU, Boundary_IoU, HD95, MAE
```

预测 mask 保存到：

```text
experiment_03_baseline_failure_analysis/
  predictions/
    <model>/
      WL/
        test/
          *.png
      NBI/
        test/
          *.png
```

文件名应与原图 stem 对齐，例如：

```text
gas_xxx.jpg -> gas_xxx.png
```

### 已完成结果摘要

四个 baseline 的 test mean 指标如下：

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

失败样本数量如下：

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

当前已经出现几个关键现象：

- 多模型失败样本存在重叠，说明问题不只是单一模型能力不足。
- Dice 较高但 Boundary IoU 低、HD95 高的样本已经出现，支持“区域重叠指标会掩盖边界错误”。
- NBI 整体 Dice/IoU 高于 WL，但边界失败样本并不少，提示高频纹理并不等价于稳定边界。
- Attention U-Net 未明显减少背景或边界失败，说明普通 attention 不一定能解决当前数据的边界频率与伪边界干扰问题。

共同失败样本摘要：

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

高 Dice 低边界质量现象：

```text
WL:
  Dice>=0.85 & Boundary_IoU<0.1: 24 rows / 18 images
  Dice>=0.85 & HD95>50: 49 rows / 40 images

NBI:
  Dice>=0.85 & Boundary_IoU<0.1: 24 rows / 17 images
  Dice>=0.85 & HD95>50: 53 rows / 39 images
```

### 已完成分析产物

实验三训练后分析脚本为：

```text
experiment_03_baseline_failure_analysis/scripts/analyze_experiment_03.py
```

运行命令：

```bash
python -B experiment_03_baseline_failure_analysis/scripts/analyze_experiment_03.py
```

已经生成：

```text
experiment_03_baseline_failure_analysis/
  outputs/
    merged_metrics/
      experiment_03_merged_metrics.csv
      model_modal_summary.csv
      correlation_by_model_modal.csv
      frequency_group_summary.csv
      analysis_summary.txt
    failure_cases/
      all_low_dice_cases.csv
      all_low_boundary_iou_cases.csv
      all_high_hd95_cases.csv
      common_failures_by_modal.csv
      high_dice_low_boundary_cases.csv
      high_dice_high_hd95_cases.csv
  figures/
    model_modal_metric_summary.png
    failure_counts_by_model_modal.png
    correlation_heatmap_spearman.png
    boundary_frequency_vs_boundary_iou.png
    boundary_frequency_vs_hd95.png
    high_dice_low_boundary_distribution.png
  visual_checks/
    common_failures_WL.png
    common_failures_NBI.png
    high_dice_low_boundary_WL.png
    high_dice_low_boundary_NBI.png
    high_hd95_WL.png
    high_hd95_NBI.png
```

分析输出规模：

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

当前阶段性结论：

- 四模型共同失败样本存在：WL 有 `45` 张图进入共同失败清单，NBI 有 `59` 张图进入共同失败清单。
- 四模型共同低 Boundary IoU 的样本为 WL `39` 张、NBI `56` 张，说明边界质量问题不是单一模型偶然失败。
- 高 Dice 低边界质量现象明确存在，支持“区域重叠指标会掩盖边界错误”。
- `global_hfr` 与 Dice 的平均相关性很弱，说明全图频率不是模型失败的直接解释变量。
- `boundary_freq_gap` 与 Boundary IoU 的平均 Spearman 相关性约为 `0.2492`，比 `global_hfr` 更贴近边界质量，但相关性仍属于中等偏弱。
- `boundary_freq_std` 与 HD95 当前相关性较弱，不能直接写成“边界频率波动导致 HD95 升高”；后续应结合区域错误拆分和可视化继续验证。

### 实验三目录约定

每个实验都要有独立文件夹。实验三建议使用：

```text
experiment_03_baseline_failure_analysis/
  configs/
  scripts/
  predictions/
    <model>/
      WL/test/
      NBI/test/
  outputs/
    prediction_metrics/
    merged_metrics/
    failure_cases/
  figures/
  visual_checks/
  logs/
```

其中：

- `configs/`：保存每个 baseline 的训练参数。
- `scripts/`：保存训练后评估、逐图指标统计、结果合并脚本。
- `predictions/`：保存 test 预测 mask。
- `outputs/prediction_metrics/`：保存逐图模型指标。
- `outputs/merged_metrics/`：保存模型指标与实验二频率指标的合并表。
- `outputs/failure_cases/`：保存失败样本清单。
- `figures/`：保存相关性图、失败分布图。
- `visual_checks/`：保存代表性失败可视化。
- `logs/`：保存训练和测试日志。

### 与频率指标合并

将实验三的 `prediction_metrics.csv` 与实验二的：

```text
experiment_02_scale_vs_boundary_frequency/
  outputs/
    boundary_frequency_metrics.csv
```

按以下键合并：

```text
image + modal + split
```

当前 8 个 prediction CSV 都已验证可以与实验二频率表完整合并：

```text
swin_unet      WL/NBI overlap: 420/420
unet           WL/NBI overlap: 420/420
unetpp         WL/NBI overlap: 420/420
attention_unet WL/NBI overlap: 420/420
```

合并后的表已经保存为：

```text
experiment_03_baseline_failure_analysis/
  outputs/
    merged_metrics/
      experiment_03_merged_metrics.csv
```

合并表至少包含：

```text
image,
modal,
split,
model,
Dice,
IoU,
Boundary_IoU,
HD95,
MAE,
scale,
global_hfr,
interior_hfr,
boundary_hfr,
near_background_hfr,
boundary_freq_std,
boundary_freq_gap
```

### 失败分析重点

实验三要重点看以下问题：

1. 多个模型是否在同一批样本上失败？

如果 U-Net、U-Net++、Attention U-Net 等在同一批样本上 `Boundary_IoU` 低、`HD95` 高，说明失败可能来自数据本身的系统性困难，而不是某一个模型能力不足。

2. 共同失败样本是否具有异常边界频率？

重点分析：

```text
boundary_hfr
boundary_freq_std
boundary_freq_gap
```

与：

```text
Boundary_IoU
HD95
MAE
```

之间的关系。

3. WL/NBI 的失败机制是否不同？

可能观察：

- WL：边界高频线索弱，模型更容易边界外扩或内缩。
- NBI：边界附近高频波动强，模型更容易受到血管纹理、反光或伪边界干扰。

4. 频率指标是否比 `scale` 更能解释边界指标下降？

如果 `scale` 与失败指标关系弱，而 `boundary_freq_std` 或 `boundary_freq_gap` 与 `Boundary_IoU / HD95` 关系更强，说明实验一/二提出的“边界频率问题”更值得作为后续方法设计依据。

### 期望结论

如果多个模型在同一批边界频率异常样本上失败，可以说明：

```text
模型失败并不是某一个网络结构偶然造成的，而是数据中存在系统性的边界频率困难。
```

如果进一步发现 `boundary_freq_std` 或 `boundary_freq_gap` 能解释 `Boundary_IoU` 下降和 `HD95` 升高，可以形成更强的问题发现结论：

```text
WL/NBI 胃镜病灶分割的主要瓶颈不只是尺度变化，而是边界频率不稳定条件下的轮廓判别。
```

---

## 4. 实验四：频率指标与失败指标相关性

### 目的

量化频率指标是否真的比 scale 更能解释模型失败，尤其是边界质量指标的下降。

重点不是只看 Dice，而是看：

- Boundary IoU
- HD95
- MAE
- Dice 与 Boundary IoU 的背离样本

### 输入文件

```text
analysis_outputs/frequency_metrics.csv
analysis_outputs/prediction_metrics.csv
```

合并后得到：

```text
analysis_outputs/frequency_prediction_merged.csv
```

### 分析问题

| 分析问题 | 如果结果成立，说明什么 |
| --- | --- |
| `scale` 与 Dice / Boundary IoU / HD95 是否相关？ | 判断尺度是否真是主要困难 |
| `global_hfr` 与 Boundary IoU 是否相关？ | 判断全图频率是否影响边界质量 |
| `boundary_hfr` 与 Boundary IoU 是否相关？ | 判断边界高频强弱是否影响边界重合 |
| `boundary_freq_std` 与 HD95 是否正相关？ | 说明边界频率不稳定会导致边界偏移 |
| `boundary_freq_gap` 是否比 `scale` 更能解释 Boundary IoU 下降？ | 说明主要瓶颈不是目标大小，而是边界频率判别 |

### 统计方式

建议至少做：

```text
Spearman correlation
Pearson correlation
按指标分位数分组后的箱线图
低/中/高 boundary_freq_std 分组下的平均 Boundary IoU 和 HD95
```

可选做简单回归对比：

```text
Model A: metric ~ scale
Model B: metric ~ scale + global_hfr
Model C: metric ~ scale + boundary_hfr + boundary_freq_std + boundary_freq_gap
```

如果 Model C 对 Boundary IoU / HD95 的解释力明显更强，就能支撑边界频率问题。

### 输出文件

```text
analysis_outputs/
  frequency_prediction_merged.csv
  frequency_metric_correlation.csv
  figures/
    corr_boundary_frequency_vs_metrics.png
    boundary_freq_groups_boxplot.png
```

### 期望结论模板

如果结果成立，可以写成：

```text
在多个 baseline 中，Boundary IoU 与边界频率稳定性呈明显相关，而 HD95 在高 boundary_freq_std 样本中显著升高。相比 lesion scale，边界频率指标更能解释模型边界定位失败，说明当前模型的主要瓶颈并非病灶区域识别，而是弱边界和频率扰动条件下的轮廓判别。
```

---

## 5. 实验五：区域错误 vs 边界错误拆分

### 目的

验证模型错误主要发生在哪里。由于本数据集中大病灶较多，Dice/IoU 可能掩盖边界错误。因此需要将错误拆成病灶内部、边界窄带和外部背景。

这一步要回答：

- 模型是不是已经找到了病灶大体区域？
- 主要错误是否集中在边界窄带？
- 错误是边界内缩、边界外扩，还是远离病灶的背景误检？

### 区域划分

沿用实验二的区域定义：

```text
interior
boundary_band
exterior
```

其中 `exterior` 是远离病灶的背景区域，不包含 `boundary_band`。

### 错误类型

统计每张图：

| 错误类型 | 含义 |
| --- | --- |
| `interior_FN` | 病灶内部漏分 |
| `boundary_FN` | 边界区域漏分，通常表示预测内缩 |
| `boundary_FP` | 边界外侧误分，通常表示预测外扩 |
| `exterior_FP` | 远离病灶的背景误检 |

### 输出文件

```text
analysis_outputs/
  error_region_metrics.csv
```

字段至少包含：

```text
image,
modal,
model,
interior_FN,
boundary_FN,
boundary_FP,
exterior_FP,
boundary_error_ratio,
exterior_error_ratio
```

### 期望观察

如果多数错误集中在 `boundary_band`，可以将问题从“边界模糊”推进到更具体的表述：

```text
当前胃镜病灶分割的主要瓶颈不是目标检测式的区域定位，而是边界窄带区域的精细判别。
```

如果 WL 主要是 `boundary_FP`，可以说明：

```text
WL 低对比条件下模型容易将正常黏膜误分为病灶，表现为边界外扩。
```

如果 NBI 主要是局部 `boundary_FP` 或 `exterior_FP`，并且与高 `boundary_freq_std` 或反光区域重合，可以说明：

```text
NBI 的强纹理和反光会引入伪边界，导致局部误分割。
```

---

## 6. 实验六：WL/NBI 模态差异实验

### 目的

利用 WL/NBI 双模态数据，判断主要矛盾是边界频率问题，还是模态差异问题。

这一步要回答：

- 混合训练是否提升 WL 和 NBI？
- 直接混合是否存在负迁移？
- 跨模态泛化是否很差？
- 跨模态失败是否主要发生在边界频率异常样本上？

### 训练/测试组合

必须做：

| 训练 | 测试 | 目的 |
| --- | --- | --- |
| WL | WL | WL 单模态性能 |
| NBI | NBI | NBI 单模态性能 |
| WL + NBI | WL | 混合训练是否提升 WL |
| WL + NBI | NBI | 混合训练是否提升 NBI |

可选做：

| 训练 | 测试 | 目的 |
| --- | --- | --- |
| WL | NBI | 看 WL 到 NBI 的跨模态泛化 |
| NBI | WL | 看 NBI 到 WL 的跨模态泛化 |

### 判断逻辑

情况 A：混合训练提升两个模态  
说明 WL/NBI 之间有共享结构信息，后续可以考虑共享表征 + 边界增强。

情况 B：混合训练提升 NBI，但伤害 WL  
说明 NBI 的强纹理/高对比特征可能干扰 WL，后续可以考虑模态自适应特征选择。

情况 C：WL 到 NBI 泛化差，NBI 到 WL 也差  
说明两个模态存在明显 domain gap，后续可以考虑模态感知归一化、模态特异增强或模态条件分支。

情况 D：跨模态泛化尚可，但边界指标依然差  
说明模态差异不是主矛盾，边界频率判别才是主矛盾。

### 输出文件

```text
analysis_outputs/
  modality_transfer_metrics.csv
  modality_transfer_per_image.csv
  figures/
    modality_transfer_summary.png
```

### 期望观察

这组实验的作用是决定后续创新点偏向哪里：

- 如果边界频率异常样本在所有训练组合下都失败，方向偏边界频率建模。
- 如果混合训练出现明显负迁移，方向偏模态自适应。
- 如果跨模态失败主要由频率分布差异解释，方向可以是模态感知的边界频率校准。

---

## 7. 实验七：扰动敏感性实验

### 目的

人为控制图像扰动，观察扰动是否会改变边界频率，并进一步导致 Boundary IoU / HD95 下降。

这一步可以把“反光、模糊、低对比、纹理干扰”从主观描述变成可控实验。

### 扰动类型

对 test 图像加入：

| 扰动 | 对应问题 |
| --- | --- |
| 降低对比度 | 模拟 WL 低对比 |
| 增加模糊 | 模拟 WL 成像模糊 |
| 加高亮反光块 | 模拟 WL/NBI 反光 |
| 改变亮度 | 模拟内镜光照变化 |
| 加局部阴影 | 模拟胃镜暗区 |
| 增强纹理噪声 | 模拟 NBI 血管纹理干扰 |

### 评估指标

比较扰动前后：

```text
Dice
Boundary IoU
HD95
global_hfr
boundary_hfr
boundary_freq_std
boundary_freq_gap
```

计算下降幅度：

```text
delta_Dice = Dice_perturbed - Dice_original
delta_Boundary_IoU = Boundary_IoU_perturbed - Boundary_IoU_original
delta_HD95 = HD95_perturbed - HD95_original
```

### 输出文件

```text
analysis_outputs/
  perturbation_sensitivity.csv
  figures/
    perturbation_metric_drop.png
    perturbation_boundary_frequency_change.png
```

### 期望观察

如果反光扰动导致 `boundary_freq_std` 升高，同时 Boundary IoU 明显下降、HD95 增大，可以说明：

```text
反光并不是简单图像噪声，而是会破坏病灶边界频率连续性的结构性干扰。
```

如果模糊扰动导致 `boundary_hfr` 降低，同时 HD95 上升，可以说明：

```text
模糊会削弱病灶边界高频线索，使模型边界定位发生偏移。
```

---

## 8. 最终要验证的核心假设

### 假设 1：尺度变化不是主要困难

验证方式：

- `scale` 方差不大。
- `scale` 与 Boundary IoU / HD95 的相关性弱。
- 高失败样本不集中在特定 scale 区间。

如果成立：

```text
本数据集的主要问题不是小目标或多尺度目标分割。
```

### 假设 2：边界频率比全图频率更能解释失败

验证方式：

- `boundary_hfr`、`boundary_freq_std`、`boundary_freq_gap` 与 Boundary IoU / HD95 显著相关。
- 它们的解释力强于 `scale` 和 `global_hfr`。

如果成立：

```text
边界频率分布不稳定是 WL/NBI 胃镜病灶分割的关键瓶颈。
```

### 假设 3：WL 与 NBI 的边界频率干扰机制不同

验证方式：

- WL 中低 `boundary_hfr` 与低 Boundary IoU / 高 HD95 相关。
- NBI 中高 `boundary_freq_std` 与局部 FP / boundary error 相关。

如果成立：

```text
WL 更偏边界高频线索不足，NBI 更偏伪高频干扰。
```

### 假设 4：主流模型在边界频率异常样本上共同失败

验证方式：

- 多个 baseline 的失败样本高度重叠。
- 重叠失败样本具有异常 `boundary_freq_std` 或 `boundary_freq_gap`。

如果成立：

```text
问题不是某个模型能力不足，而是数据本身存在系统性边界频率困难。
```

---

## 9. 可能导向的方法方向

只有在上述实验成立后，再考虑方法设计。可能方向包括：

```text
boundary-frequency attention
DCT-based boundary branch
frequency-aware skip fusion
modality-aware frequency calibration
specular-aware frequency suppression
boundary frequency consistency loss
```

不要在实验前就强行确定方法。更合理的写法是：

```text
我们首先通过 scale-frequency 统计和逐图失败分析发现：模型失败与 lesion scale 的关系较弱，而与 boundary frequency instability 更相关。基于这一观察，再设计边界频率感知模块。
```

---

## 10. 阶段性执行顺序

建议按以下顺序推进：

1. 实现 `frequency_metrics.csv` 统计脚本。
2. 画 `scale_vs_global_frequency.png`，复刻 MADGNet Figure 1 的基础版本。
3. 增加 `boundary_hfr`、`boundary_freq_std`、`boundary_freq_gap`，画边界频率分布图。
4. 在 GPU 服务器训练 baseline，并保存逐图预测指标。当前第一批四个 baseline 已完成。
5. 合并频率指标和模型指标，生成统一 `merged_metrics`，做相关性分析。当前已完成。
6. 做区域错误拆分，判断错误是否集中在边界窄带。
7. 做 WL/NBI 模态训练组合，判断是否存在模态负迁移。
8. 做扰动敏感性实验，验证反光、模糊、纹理是否通过改变边界频率影响分割。
9. 根据实验结果再决定最终创新点。

---

## 11. 当前默认约定

- 第一批四个 baseline 已经在 GPU 服务器完成训练与测试；后续新增 baseline 或改进方法仍在 GPU 服务器执行。
- 当前本机优先整理输出、合并指标、生成失败分析清单和可视化脚本。
- 第一版问题发现只使用当前 `dataset/WL` 和 `dataset/NBI` 的 train/val/test。
- 当前没有患者 ID，因此暂不做患者级划分分析。
- 如果边界频率实验解释力不足，则回到模态负迁移、反光扰动敏感性或传统边界错误分析作为主线。
