# 实验二：尺度 vs 边界频率分布

本实验将实验一的全图频率分析推进到病灶局部区域，重点观察 WL/NBI 的差异是否更明显地体现在病灶边界附近，而不是全图频率或病灶尺度本身。

重要约束：`../dataset/` 只读。不要在 `dataset/` 内新建、删除、重命名、移动或覆盖任何文件。

## 目标

基于 GT mask 划分三个区域，并计算这些区域的局部高频功率比例：

```text
interior        = mask 腐蚀后的病灶内部
boundary_band   = mask 膨胀区域 - mask 腐蚀区域
near_background = 病灶外侧邻近背景环带
```

## 默认参数

```text
image_size = 256
mask_threshold = 127
erode_dilate_radius = 5
patch_size = 32
max_points_per_region = 256
high_frequency_outer_ratio = 1/3
random_seed = 20260509
```

每个区域最多采样 `256` 个 patch 中心点。每个中心点裁剪一个 `32x32` 灰度 patch，图像边缘使用 reflect padding。每个 patch 使用 FFT 计算高频功率比例。

## 运行

在仓库根目录运行：

```bash
python experiment_02_scale_vs_boundary_frequency/scripts/run_experiment_02.py
```

快速试跑命令：

```bash
python experiment_02_scale_vs_boundary_frequency/scripts/run_experiment_02.py --limit-per-split 2
```

## 输出

```text
experiment_02_scale_vs_boundary_frequency/
  outputs/
    boundary_frequency_metrics.csv
    boundary_frequency_summary_by_modal.csv
    boundary_frequency_summary_by_modal_split.csv
  figures/
    scale_vs_boundary_frequency.png
    wl_nbi_boundary_frequency_distribution.png
    global_vs_boundary_frequency.png
  visual_checks/
    region_definition_examples.png
```

## 结果图与解读

### 图 1：区域划分检查

![Region Definition Examples](visual_checks/region_definition_examples.png)

这张图用于确认区域定义是否符合直觉：绿色表示 `interior`，黄色表示 `boundary_band`，蓝色表示 `near_background`。

实验二的频率统计依赖 GT mask 划分区域，因此这张图首先确认我们统计的是病灶内部、边界窄带和邻近背景，而不是任意全图区域。

### 图 2：Scale vs Boundary Frequency

![Scale vs Boundary Frequency](figures/scale_vs_boundary_frequency.png)

这张图将实验一的纵轴从全图频率 `global_hfr` 换成边界频率 `boundary_hfr`。它用于观察：在相似病灶尺度下，WL/NBI 是否仍然存在明显的边界频率差异。

从首轮结果看，NBI 的 `boundary_hfr_mean` 约为 WL 的 `2.44x`。这说明 NBI 不只是全图高频更强，在病灶边界区域也存在更强的局部高频成分。

### 图 3：WL/NBI 边界频率分布对比

![WL/NBI Boundary Frequency Distribution](figures/wl_nbi_boundary_frequency_distribution.png)

这张图对比了 WL/NBI 的 `boundary_hfr`、`boundary_freq_std` 和 `boundary_freq_gap`。

首轮结果中，NBI 的 `boundary_freq_std_mean` 约为 WL 的 `1.82x`，`boundary_freq_gap_mean` 约为 WL 的 `1.62x`。这说明 NBI 边界附近不仅高频更强，而且局部频率波动也更明显。

这与胃镜 NBI 的视觉特点是吻合的：NBI 会突出血管和黏膜纹理，可能带来更多边界附近的高频变化；这些高频不一定全部对应真实病灶边界，也可能包含纹理或反光造成的伪边界。

### 图 4：Global Frequency vs Boundary Frequency

![Global vs Boundary Frequency](figures/global_vs_boundary_frequency.png)

这张图用于检查全图频率和边界频率是否完全一致。若二者并不完全等价，就说明只看 `global_hfr` 还不够，需要进一步关注边界局部频率。

当前结果支持这样的判断：NBI 的全图频率和边界频率都高于 WL，但边界相关指标还能进一步描述局部波动和边界-背景频率差异。因此，边界频率比全图频率更接近分割任务真正关心的区域。

### 谨慎结论

当前实验二只能说明 WL/NBI 在病灶边界区域存在频率分布差异，还不能直接证明模型失败一定由边界频率导致。

下一步需要训练或收集 baseline 模型预测结果，将逐图 Dice、Boundary IoU、HD95 与 `boundary_hfr`、`boundary_freq_std`、`boundary_freq_gap` 合并分析，才能判断边界频率是否真的解释模型失败。

## 结果解读

如果 `boundary_hfr`、`boundary_freq_std` 或 `boundary_freq_gap` 比 `scale` 更能区分 WL/NBI，说明“边界频率”可以作为下一步问题发现的核心变量，再与模型逐图失败指标合并分析。

## 首轮运行摘要

已在全量数据集上完成：

```text
total rows: 5600
WL rows: 2800
NBI rows: 2800
per modal split: train 1960, val 420, test 420
```

模态汇总：

```text
modal  count  scale_mean  global_hfr_mean  interior_hfr_mean  boundary_hfr_mean  near_background_hfr_mean  boundary_freq_std_mean  boundary_freq_gap_mean
NBI    2800   0.377250    0.000567         0.000815           0.000802           0.000877                  0.000508                0.000355
WL     2800   0.310968    0.000142         0.000241           0.000329           0.000411                  0.000279                0.000219
```

初步观察：NBI 在病灶内部、边界区域、邻近背景区域的局部频率值都高于 WL。同时，NBI 的 `boundary_freq_std` 均值也更高，说明 NBI 边界附近存在更强的局部频率波动。这个结果支持下一步将这些边界频率指标与模型逐图失败指标合并，继续做模型失败归因分析。
