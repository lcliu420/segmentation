# 实验一：尺度 vs 全图频率分布

本实验参考 MADGNet 论文 Figure 1 的问题发现逻辑：在提出方法之前，先比较病灶尺度分布和频率分布，判断数据集的主要变化是否真的来自尺度。

重要约束：`../dataset/` 只读。不要在 `dataset/` 内新建、删除、重命名、移动或覆盖任何文件。

## 目标

衡量 WL/NBI 胃镜病灶图像的差异，主要体现在病灶尺度 `scale`，还是体现在全图高频功率比例 `global_hfr`。

## 输入

```text
../dataset/
  WL/{train,val,test}/images/*.jpg
  WL/{train,val,test}/masks/*.png
  NBI/{train,val,test}/images/*.jpg
  NBI/{train,val,test}/masks/*.png
```

## 指标

```text
scale = mask 前景像素数 / 总像素数
global_hfr = 高频功率 / 全频功率
```

图像和 mask 统一 resize 到 `256x256`。图像转为灰度并归一化到 `[0, 1]`。全图频率使用 2D FFT 计算。高频区域定义为 FFT shift 后频谱半径外侧的 `1/3`。

## 运行

在仓库根目录运行：

```bash
python experiment_01_scale_vs_global_frequency/scripts/run_experiment_01.py
```

## 输出

```text
experiment_01_scale_vs_global_frequency/
  outputs/
    frequency_metrics.csv
    frequency_summary_by_modal.csv
    frequency_summary_by_modal_split.csv
  figures/
    scale_vs_global_frequency.png
    scale_vs_global_frequency_by_split.png
```

## 结果图与解读

### 图 1：Scale vs Global Frequency

![Scale vs Global Frequency](figures/scale_vs_global_frequency.png)

这张图对应 MADGNet Figure 1 的基础思路：横轴是病灶尺度 `scale`，纵轴是全图高频功率比例 `global_hfr`，颜色区分 WL 和 NBI。

从首轮结果看，NBI 的 `global_hfr_mean` 约为 WL 的 `3.98x`，而 NBI/WL 的 `scale_mean` 只约为 `1.21x`，`scale_var` 约为 `1.11x`。这说明在当前数据集中，WL/NBI 的差异在全图频率上比在病灶尺度上更明显。

因此，单纯把问题理解为“病灶大小变化”或只围绕多尺度建模，可能不足以解释 WL/NBI 的主要差异。更合理的下一步是继续分析频率信息，尤其是病灶边界区域的频率。

### 图 2：按数据划分查看 Scale vs Global Frequency

![Scale vs Global Frequency by Split](figures/scale_vs_global_frequency_by_split.png)

这张图将 train、val、test 分开显示，用来检查划分是否造成明显分布偏移。整体上，三个 split 中 WL/NBI 的相对关系保持一致：NBI 的全图高频比例整体高于 WL。

这说明实验一观察到的频率差异不是某一个 split 偶然造成的，而是贯穿 train/val/test 的模态分布现象。

## 结果解读

如果 `global_hfr` 比 `scale` 更能区分 WL/NBI，或者表现出更明显的模态差异，说明后续不应只关注病灶大小和多尺度建模，还需要继续分析频率信息，尤其是病灶边界区域的频率。

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
modal  count  scale_mean  scale_std  scale_var  global_hfr_mean  global_hfr_std  global_hfr_var
NBI    2800   0.377250    0.228985   0.052434   0.000567         0.000387        1.494583e-07
WL     2800   0.310968    0.217083   0.047125   0.000142         0.000154        2.357139e-08
```

初步观察：NBI 的全图高频比例均值明显高于 WL，而两者的病灶尺度分布差异相对更接近。这支持继续做实验二，把频率分析从全图推进到病灶边界区域。
