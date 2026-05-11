# 实验四：频率指标与模型失败指标相关性分析

本实验复用实验三已经生成的逐图合并指标表，进一步量化 `scale`、`global_hfr`、`boundary_hfr`、`boundary_freq_std` 和 `boundary_freq_gap` 与模型失败指标之间的关系。

重要约束：本实验不重新计算频率指标，不重新运行模型，也不修改 `../dataset/`。所有新产物只写入 `experiment_04_frequency_failure_correlation/`。

## 目的

实验一和实验二说明 WL/NBI 在全图频率和边界频率上存在差异，实验三说明多个 baseline 存在共同边界失败。本实验用于回答：

```text
边界频率指标是否比 scale 或 global_hfr 更贴近模型边界质量下降？
```

这里的分析只用于问题发现和解释力对比，不作为因果证明。

## 输入

```text
../experiment_03_baseline_failure_analysis/
  outputs/
    merged_metrics/
      experiment_03_merged_metrics.csv
```

输入表共 `3360` 行，包含 4 个模型、2 个模态，每个 model-modal pair 各 `420` 张 test 图。

## 运行

在项目根目录运行：

```bash
python -B experiment_04_frequency_failure_correlation/scripts/run_experiment_04.py
```

## 输出

```text
experiment_04_frequency_failure_correlation/
  outputs/
    frequency_prediction_merged.csv
    frequency_metric_correlation.csv
    frequency_metric_correlation_mean.csv
    frequency_group_summary.csv
    frequency_group_assignments.csv
    regression_model_comparison.csv
    regression_model_comparison_mean.csv
  figures/
    corr_boundary_frequency_vs_metrics.png
    boundary_freq_groups_boxplot.png
    regression_r2_comparison.png
  logs/
    run_summary.txt
    warnings.txt
```

## 结果图与解释

### 图 1：频率指标与预测指标相关性

![Correlation Heatmap](figures/corr_boundary_frequency_vs_metrics.png)

该图展示 8 个 model-modal 分组上 Spearman 相关性的平均值。当前结果中：

```text
scale vs Dice mean Spearman: 0.5188
scale vs Boundary_IoU mean Spearman: 0.3245
global_hfr vs Boundary_IoU mean Spearman: 0.0332
boundary_hfr vs Boundary_IoU mean Spearman: 0.1386
boundary_freq_gap vs Boundary_IoU mean Spearman: 0.2492
boundary_freq_std vs HD95 mean Spearman: 0.0205
```

从相关性角度看，`global_hfr` 与模型边界质量的关系较弱；`boundary_hfr`、`boundary_freq_gap` 与 `Boundary_IoU` 的关系更明显，说明边界局部频率比全图频率更贴近边界质量变化。与此同时，相关性整体仍属于中等或偏弱，因此不能直接写成“边界频率导致模型失败”。

### 图 2：边界频率分组下的边界指标变化

![Boundary Frequency Groups](figures/boundary_freq_groups_boxplot.png)

该图将 `boundary_freq_std` 和 `boundary_freq_gap` 按每个 model-modal 分组三分位划分为 low/mid/high，再观察 `Boundary_IoU` 和 `HD95` 的分布。它用于检查频率指标升高时，边界质量是否呈现一致变化。

当前结果更适合支持一个谨慎判断：边界频率分组确实会对应边界质量分布差异，但不同模型和不同模态之间仍存在离散性。

### 图 3：简单回归解释力对比

![Regression R2 Comparison](figures/regression_r2_comparison.png)

本实验比较三组线性模型：

```text
Model A: metric ~ scale
Model B: metric ~ scale + global_hfr
Model C: metric ~ scale + boundary_hfr + boundary_freq_std + boundary_freq_gap
```

其中 `Boundary_IoU` 的平均 adjusted R2 为：

```text
Model A: 0.1480
Model B: 0.1528
Model C: 0.1799
```

`HD95` 的平均 adjusted R2 为：

```text
Model A: 0.0348
Model B: 0.0404
Model C: 0.0436
```

如果 Model C 高于 Model A/B，说明加入边界频率指标后，对边界质量指标的解释力更强。但由于这里是简单线性回归，结论应写成解释力增强，而不是因果机制成立。

## 谨慎结论

本实验支持以下阶段性判断：

- `scale` 与 Dice/IoU 等区域重叠指标关系更明显，说明病灶面积仍会影响区域分割结果。
- `global_hfr` 与边界失败指标的关系较弱，说明全图频率不是解释模型失败的直接核心变量。
- `boundary_hfr` 和 `boundary_freq_gap` 相比 `global_hfr` 更贴近 `Boundary_IoU` 的变化，提示边界局部频率比全图频率更接近任务瓶颈。
- 当前相关性和回归解释力仍不足以单独完成因果论证，后续需要结合实验五的区域错误拆分，进一步判断错误是否主要集中在边界窄带，以及是内缩、外扩还是远处背景误检。

一句话总结：

```text
边界频率指标比全图频率更贴近边界质量变化，但相关性整体仍属于中等或偏弱，后续需要结合区域错误拆分继续验证。
```

## 运行警告

无。
