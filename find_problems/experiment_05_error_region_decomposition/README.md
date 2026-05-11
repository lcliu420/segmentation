# 实验五：区域错误 vs 边界错误拆分

本实验在实验三逐图预测结果的基础上，将错误拆分到病灶内部、边界窄带和远处背景区域，用于判断当前模型失败是否主要集中在边界附近。

重要约束：`../dataset/` 只读。本实验只读取实验三的合并指标和预测 mask，所有新产物均写入 `experiment_05_error_region_decomposition/`。

## 目的

实验四说明边界频率指标比全图频率更贴近边界质量变化，但仍不足以单独证明模型错误确实发生在边界区域。因此实验五进一步回答：

```text
模型错误主要是病灶内部漏分、边界内缩/外扩，还是远处背景误检？
```

## 区域定义

本实验复用实验二的区域定义：

```text
image_size = 256
mask_threshold = 127
erode_dilate_radius = 5

interior      = erosion(GT)
boundary_band = dilation(GT) - erosion(GT)
exterior      = outside dilation(GT)
```

为了区分边界内缩和外扩，额外定义：

```text
boundary_inner = boundary_band & GT
boundary_outer = boundary_band & ~GT
```

## 错误指标

```text
interior_FN = FN & interior
boundary_FN = FN & boundary_inner
boundary_FP = FP & boundary_outer
exterior_FP = FP & exterior
```

其中 `boundary_FN` 更接近边界内缩或边界漏分，`boundary_FP` 更接近边界外扩或邻近背景误分，`exterior_FP` 表示远离病灶的背景误检。

## 运行

在项目根目录运行：

```bash
python -B experiment_05_error_region_decomposition/scripts/run_experiment_05.py
```

## 输出

```text
outputs/
  error_region_metrics.csv
  error_region_summary_by_model_modal.csv
  error_region_summary_by_modal.csv
  error_region_correlation.csv
  common_failure_error_summary.csv
  high_dice_low_boundary_error_summary.csv
  high_dice_low_boundary_error_cases.csv
figures/
  error_region_summary_by_model_modal.png
  error_region_composition_by_modal.png
  boundary_fn_fp_balance.png
  error_region_vs_boundary_frequency.png
  high_dice_low_boundary_error_composition.png
visual_checks/
  error_region_examples_WL.png
  error_region_examples_NBI.png
  high_dice_low_boundary_errors_WL.png
  high_dice_low_boundary_errors_NBI.png
```

## 结果摘要

按模态汇总的平均错误组成如下：

```text
WL: boundary_error_ratio 0.3933, interior_error_ratio 0.2793, exterior_error_ratio 0.3274, boundary_FP_balance 0.4778
NBI: boundary_error_ratio 0.3956, interior_error_ratio 0.2429, exterior_error_ratio 0.3615, boundary_FP_balance 0.4982
```

共同失败样本数量：

```text
WL: 45
NBI: 59
```

高 Dice 低 Boundary IoU 样本中的错误组成：

```text
NBI: n 24, boundary_error_ratio 0.4064, exterior_error_ratio 0.3164
WL: n 24, boundary_error_ratio 0.4260, exterior_error_ratio 0.3556
```

## 图表解释

![Model Modal Error Summary](figures/error_region_summary_by_model_modal.png)

该图比较不同模型和模态下的平均错误组成。如果 `boundary_error_ratio` 占比高，说明模型错误更多集中在 GT 边界窄带，而不是完全找不到病灶主体。

![Modal Error Composition](figures/error_region_composition_by_modal.png)

该图按 WL/NBI 汇总错误区域组成，用于观察两个模态的失败形态是否一致。

![Boundary FN FP Balance](figures/boundary_fn_fp_balance.png)

该图进一步区分边界错误中的 `boundary_FN` 和 `boundary_FP`。`boundary_FN` 偏高通常对应边界内缩或漏分，`boundary_FP` 偏高通常对应边界外扩或邻近背景误分。

![Boundary Frequency vs Error](figures/error_region_vs_boundary_frequency.png)

该图观察 `boundary_freq_std` 和 `boundary_freq_gap` 与边界错误占比的关系。它只用于辅助解释，不作为因果证明。

![High Dice Low Boundary Error](figures/high_dice_low_boundary_error_composition.png)

该图专门检查高 Dice 低边界质量样本是否主要由边界错误构成。

## 谨慎结论

本实验的结论应和实验四一起使用：实验四回答“边界频率指标是否更贴近边界质量”，实验五回答“模型错误是否真的集中在边界区域”。如果 `boundary_error_ratio` 在多数模型和模态中占比较高，就可以更稳妥地说明当前瓶颈不是单纯区域识别，而是边界窄带中的精细轮廓判断。

需要注意，本实验仍然是错误归因分析，不直接证明频率扰动导致错误。后续如果要继续增强证据，可以结合扰动敏感性实验或方法消融。
