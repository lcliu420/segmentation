import argparse
import contextlib
import os
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
EXP_DIR = SCRIPT_PATH.parents[1]
WORKSPACE = EXP_DIR.parent
MPLCONFIG = EXP_DIR / ".mplconfig" / f"run_{os.getpid()}"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as cbook


@contextlib.contextmanager
def _no_lock_path(_path):
    yield


cbook._lock_path = _no_lock_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_MODELS = ["swin_unet", "unet", "unetpp", "attention_unet"]
EXPECTED_MODALITIES = ["WL", "NBI"]
EXPECTED_ROWS = 3360
EXPECTED_ROWS_PER_PAIR = 420

FREQUENCY_COLUMNS = [
    "scale",
    "global_hfr",
    "boundary_hfr",
    "boundary_freq_std",
    "boundary_freq_gap",
]
PREDICTION_COLUMNS = ["Dice", "IoU", "Boundary_IoU", "HD95", "MAE"]

REGRESSION_SPECS = {
    "A_scale": ["scale"],
    "B_scale_global": ["scale", "global_hfr"],
    "C_scale_boundary": [
        "scale",
        "boundary_hfr",
        "boundary_freq_std",
        "boundary_freq_gap",
    ],
}

GROUP_LABELS = ["low", "mid", "high"]
MODEL_LABELS = {
    "swin_unet": "Swin-Unet",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net",
}
MODAL_COLORS = {"WL": "#1f77b4", "NBI": "#d62728"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 04: frequency metrics vs failure metrics."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=WORKSPACE
        / "experiment_03_baseline_failure_analysis"
        / "outputs"
        / "merged_metrics"
        / "experiment_03_merged_metrics.csv",
    )
    parser.add_argument("--exp_dir", type=Path, default=EXP_DIR)
    return parser.parse_args()


def ensure_dirs(exp_dir):
    dirs = {
        "outputs": exp_dir / "outputs",
        "figures": exp_dir / "figures",
        "logs": exp_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def validate_input(df):
    required = ["image", "modal", "split", "model"] + FREQUENCY_COLUMNS + PREDICTION_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Input CSV is missing required columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    models = sorted(df["model"].dropna().unique().tolist())
    modalities = sorted(df["modal"].dropna().unique().tolist())
    if models != sorted(EXPECTED_MODELS):
        raise RuntimeError(f"Unexpected model set: {models}")
    if modalities != sorted(EXPECTED_MODALITIES):
        raise RuntimeError(f"Unexpected modality set: {modalities}")

    counts = df.groupby(["model", "modal"], observed=True).size()
    bad_counts = counts[counts != EXPECTED_ROWS_PER_PAIR]
    if not bad_counts.empty:
        raise RuntimeError(
            "Each model-modal pair must contain "
            f"{EXPECTED_ROWS_PER_PAIR} rows. Bad counts: {bad_counts.to_dict()}"
        )

    if df[required].isna().any().any():
        na_cols = df[required].columns[df[required].isna().any()].tolist()
        raise RuntimeError(f"Input CSV has missing values in required columns: {na_cols}")


def ordered_frame(df):
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], EXPECTED_MODELS, ordered=True)
    df["modal"] = pd.Categorical(df["modal"], EXPECTED_MODALITIES, ordered=True)
    return df.sort_values(["modal", "model", "image"]).reset_index(drop=True)


def save_correlations(df, outputs_dir):
    rows = []
    for (model, modal), group in df.groupby(["model", "modal"], observed=True):
        for x_col in FREQUENCY_COLUMNS:
            for y_col in PREDICTION_COLUMNS:
                pair = group[[x_col, y_col]].dropna()
                rows.append(
                    {
                        "model": model,
                        "modal": modal,
                        "frequency_metric": x_col,
                        "prediction_metric": y_col,
                        "n": len(pair),
                        "spearman": pair[x_col].corr(pair[y_col], method="spearman"),
                        "pearson": pair[x_col].corr(pair[y_col], method="pearson"),
                    }
                )

    corr = pd.DataFrame(rows)
    corr["model"] = pd.Categorical(corr["model"], EXPECTED_MODELS, ordered=True)
    corr["modal"] = pd.Categorical(corr["modal"], EXPECTED_MODALITIES, ordered=True)
    corr = corr.sort_values(
        ["modal", "model", "frequency_metric", "prediction_metric"]
    ).reset_index(drop=True)
    corr.to_csv(outputs_dir / "frequency_metric_correlation.csv", index=False)

    mean_corr = (
        corr.groupby(["frequency_metric", "prediction_metric"], as_index=False)
        .agg(
            n_groups=("spearman", "count"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
        )
        .reset_index(drop=True)
    )
    mean_corr["abs_spearman_mean"] = mean_corr["spearman_mean"].abs()
    mean_corr["abs_pearson_mean"] = mean_corr["pearson_mean"].abs()
    mean_corr.to_csv(outputs_dir / "frequency_metric_correlation_mean.csv", index=False)
    return corr, mean_corr


def qcut_groups(values, warnings, context):
    bins = pd.qcut(values, q=3, labels=GROUP_LABELS, duplicates="drop")
    actual_groups = [str(group) for group in bins.dropna().unique().tolist()]
    if len(actual_groups) < 3:
        warnings.append(
            f"qcut produced {len(actual_groups)} groups for {context}; groups={actual_groups}"
        )
    return bins


def save_frequency_group_summary(df, outputs_dir):
    warnings = []
    rows = []
    grouped_frames = []

    for (model, modal), group in df.groupby(["model", "modal"], observed=True):
        for metric in FREQUENCY_COLUMNS:
            context = f"{model}/{modal}/{metric}"
            bins = qcut_groups(group[metric], warnings, context)
            temp = group.assign(frequency_metric=metric, frequency_group=bins)
            grouped_frames.append(temp)
            for freq_group, subgroup in temp.groupby("frequency_group", observed=True):
                row = {
                    "model": model,
                    "modal": modal,
                    "frequency_metric": metric,
                    "frequency_group": str(freq_group),
                    "n": len(subgroup),
                    "frequency_mean": subgroup[metric].mean(),
                    "frequency_min": subgroup[metric].min(),
                    "frequency_max": subgroup[metric].max(),
                }
                for pred_col in PREDICTION_COLUMNS:
                    row[f"{pred_col}_mean"] = subgroup[pred_col].mean()
                    row[f"{pred_col}_std"] = subgroup[pred_col].std(ddof=1)
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(outputs_dir / "frequency_group_summary.csv", index=False)

    group_assignments = pd.concat(grouped_frames, ignore_index=True)
    keep_cols = [
        "image",
        "modal",
        "model",
        "frequency_metric",
        "frequency_group",
        *FREQUENCY_COLUMNS,
        *PREDICTION_COLUMNS,
    ]
    group_assignments[keep_cols].to_csv(
        outputs_dir / "frequency_group_assignments.csv", index=False
    )
    return summary, group_assignments, warnings


def fit_linear_regression(group, y_col, features):
    data = group[[y_col, *features]].dropna()
    y = data[y_col].to_numpy(dtype=float)
    x = data[features].to_numpy(dtype=float)
    n = len(data)
    p = len(features)
    if n <= p + 1:
        return n, np.nan, np.nan

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_scaled = (x - x_mean) / x_std
    design = np.column_stack([np.ones(n), x_scaled])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return n, np.nan, np.nan

    r2 = 1.0 - (ss_res / ss_tot)
    adjusted_r2 = 1.0 - (1.0 - r2) * ((n - 1) / (n - p - 1))
    return n, r2, adjusted_r2


def save_regression_comparison(df, outputs_dir):
    rows = []
    for (model, modal), group in df.groupby(["model", "modal"], observed=True):
        for y_col in PREDICTION_COLUMNS:
            for spec_name, features in REGRESSION_SPECS.items():
                n, r2, adjusted_r2 = fit_linear_regression(group, y_col, features)
                rows.append(
                    {
                        "model": model,
                        "modal": modal,
                        "prediction_metric": y_col,
                        "regression_model": spec_name,
                        "features": " + ".join(features),
                        "n": n,
                        "n_features": len(features),
                        "r2": r2,
                        "adjusted_r2": adjusted_r2,
                    }
                )

    regression = pd.DataFrame(rows)
    regression["model"] = pd.Categorical(regression["model"], EXPECTED_MODELS, ordered=True)
    regression["modal"] = pd.Categorical(
        regression["modal"], EXPECTED_MODALITIES, ordered=True
    )
    regression = regression.sort_values(
        ["modal", "model", "prediction_metric", "regression_model"]
    ).reset_index(drop=True)
    regression.to_csv(outputs_dir / "regression_model_comparison.csv", index=False)

    mean_regression = (
        regression.groupby(["prediction_metric", "regression_model"], as_index=False)
        .agg(
            n_groups=("r2", "count"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            adjusted_r2_mean=("adjusted_r2", "mean"),
            adjusted_r2_std=("adjusted_r2", "std"),
        )
        .reset_index(drop=True)
    )
    mean_regression.to_csv(
        outputs_dir / "regression_model_comparison_mean.csv", index=False
    )
    return regression, mean_regression


def save_correlation_heatmap(mean_corr, fig_path):
    heat = (
        mean_corr.pivot(
            index="frequency_metric",
            columns="prediction_metric",
            values="spearman_mean",
        )
        .reindex(index=FREQUENCY_COLUMNS, columns=PREDICTION_COLUMNS)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    im = ax.imshow(heat.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(PREDICTION_COLUMNS)))
    ax.set_yticks(np.arange(len(FREQUENCY_COLUMNS)))
    ax.set_xticklabels(PREDICTION_COLUMNS, rotation=35, ha="right")
    ax.set_yticklabels(FREQUENCY_COLUMNS)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = heat.values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Spearman r")
    ax.set_title("Frequency Metrics vs Prediction Metrics")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_group_boxplot(group_assignments, fig_path):
    metrics = ["boundary_freq_std", "boundary_freq_gap"]
    targets = ["Boundary_IoU", "HD95"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

    for row_idx, metric in enumerate(metrics):
        subset = group_assignments[group_assignments["frequency_metric"] == metric]
        for col_idx, target in enumerate(targets):
            ax = axes[row_idx, col_idx]
            data = [
                subset.loc[subset["frequency_group"].astype(str) == label, target].values
                for label in GROUP_LABELS
            ]
            ax.boxplot(
                data,
                labels=GROUP_LABELS,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                boxprops={"facecolor": "#d9e8f5", "edgecolor": "#4a6f8a"},
                whiskerprops={"color": "#4a6f8a"},
                capprops={"color": "#4a6f8a"},
            )
            means = [np.nanmean(values) if len(values) else np.nan for values in data]
            ax.plot(np.arange(1, 4), means, color="#d62728", marker="o", linewidth=1.5)
            ax.set_xlabel(f"{metric} tertile")
            ax.set_ylabel(target)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"{target} by {metric}")

    fig.suptitle("Boundary Frequency Groups vs Boundary Metrics")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_regression_figure(mean_regression, fig_path):
    fig, axes = plt.subplots(1, len(PREDICTION_COLUMNS), figsize=(17, 4.2), sharey=True)
    model_order = list(REGRESSION_SPECS.keys())
    colors = ["#8da0cb", "#66c2a5", "#fc8d62"]

    for ax, target in zip(axes, PREDICTION_COLUMNS):
        subset = mean_regression[mean_regression["prediction_metric"] == target]
        subset = subset.set_index("regression_model").reindex(model_order)
        values = subset["adjusted_r2_mean"].to_numpy(dtype=float)
        ax.bar(np.arange(len(model_order)), values, color=colors, width=0.68)
        ax.set_title(target)
        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels(["A", "B", "C"])
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(0.01, np.nanmax(mean_regression["adjusted_r2_mean"]) * 1.15))
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Mean adjusted R2")
    fig.suptitle(
        "Regression comparison: A=scale, B=scale+global_hfr, C=scale+boundary metrics"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def lookup_corr(mean_corr, frequency_metric, prediction_metric, column="spearman_mean"):
    row = mean_corr[
        (mean_corr["frequency_metric"] == frequency_metric)
        & (mean_corr["prediction_metric"] == prediction_metric)
    ]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def lookup_reg(mean_regression, prediction_metric, regression_model, column="adjusted_r2_mean"):
    row = mean_regression[
        (mean_regression["prediction_metric"] == prediction_metric)
        & (mean_regression["regression_model"] == regression_model)
    ]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def write_readme(exp_dir, mean_corr, mean_regression, warnings):
    scale_dice = lookup_corr(mean_corr, "scale", "Dice")
    scale_boundary = lookup_corr(mean_corr, "scale", "Boundary_IoU")
    global_boundary = lookup_corr(mean_corr, "global_hfr", "Boundary_IoU")
    boundary_gap_boundary = lookup_corr(
        mean_corr, "boundary_freq_gap", "Boundary_IoU"
    )
    boundary_std_hd95 = lookup_corr(mean_corr, "boundary_freq_std", "HD95")
    boundary_hfr_boundary = lookup_corr(mean_corr, "boundary_hfr", "Boundary_IoU")

    boundary_r2_a = lookup_reg(mean_regression, "Boundary_IoU", "A_scale")
    boundary_r2_b = lookup_reg(mean_regression, "Boundary_IoU", "B_scale_global")
    boundary_r2_c = lookup_reg(mean_regression, "Boundary_IoU", "C_scale_boundary")
    hd95_r2_a = lookup_reg(mean_regression, "HD95", "A_scale")
    hd95_r2_b = lookup_reg(mean_regression, "HD95", "B_scale_global")
    hd95_r2_c = lookup_reg(mean_regression, "HD95", "C_scale_boundary")

    warning_text = "无。"
    if warnings:
        warning_text = "\n".join(f"- {item}" for item in warnings)

    text = f"""# 实验四：频率指标与模型失败指标相关性分析

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
scale vs Dice mean Spearman: {scale_dice:.4f}
scale vs Boundary_IoU mean Spearman: {scale_boundary:.4f}
global_hfr vs Boundary_IoU mean Spearman: {global_boundary:.4f}
boundary_hfr vs Boundary_IoU mean Spearman: {boundary_hfr_boundary:.4f}
boundary_freq_gap vs Boundary_IoU mean Spearman: {boundary_gap_boundary:.4f}
boundary_freq_std vs HD95 mean Spearman: {boundary_std_hd95:.4f}
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
Model A: {boundary_r2_a:.4f}
Model B: {boundary_r2_b:.4f}
Model C: {boundary_r2_c:.4f}
```

`HD95` 的平均 adjusted R2 为：

```text
Model A: {hd95_r2_a:.4f}
Model B: {hd95_r2_b:.4f}
Model C: {hd95_r2_c:.4f}
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

{warning_text}
"""
    (exp_dir / "README.md").write_text(text, encoding="utf-8")


def write_logs(dirs, df, corr, group_summary, regression, warnings):
    lines = [
        "Experiment 04 run summary",
        "",
        f"merged_rows: {len(df)}",
        f"correlation_rows: {len(corr)}",
        f"group_summary_rows: {len(group_summary)}",
        f"regression_rows: {len(regression)}",
        "",
        "model_modal_rows:",
    ]
    counts = df.groupby(["model", "modal"], observed=True).size()
    for (model, modal), count in counts.items():
        lines.append(f"  {model}/{modal}: {count}")
    lines.append("")
    lines.append(f"warnings: {len(warnings)}")
    (dirs["logs"] / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    warning_text = "\n".join(warnings) if warnings else "No warnings."
    (dirs["logs"] / "warnings.txt").write_text(warning_text + "\n", encoding="utf-8")


def assert_outputs(dirs):
    expected_files = [
        dirs["outputs"] / "frequency_prediction_merged.csv",
        dirs["outputs"] / "frequency_metric_correlation.csv",
        dirs["outputs"] / "frequency_metric_correlation_mean.csv",
        dirs["outputs"] / "frequency_group_summary.csv",
        dirs["outputs"] / "regression_model_comparison.csv",
        dirs["figures"] / "corr_boundary_frequency_vs_metrics.png",
        dirs["figures"] / "boundary_freq_groups_boxplot.png",
        dirs["figures"] / "regression_r2_comparison.png",
    ]
    missing_or_empty = [
        path for path in expected_files if (not path.exists()) or path.stat().st_size == 0
    ]
    if missing_or_empty:
        raise RuntimeError(f"Missing or empty output files: {missing_or_empty}")


def main():
    args = parse_args()
    dirs = ensure_dirs(args.exp_dir)

    df = pd.read_csv(args.input_csv)
    validate_input(df)
    df = ordered_frame(df)

    merged_path = dirs["outputs"] / "frequency_prediction_merged.csv"
    df.to_csv(merged_path, index=False)

    corr, mean_corr = save_correlations(df, dirs["outputs"])
    group_summary, group_assignments, warnings = save_frequency_group_summary(
        df, dirs["outputs"]
    )
    regression, mean_regression = save_regression_comparison(df, dirs["outputs"])

    save_correlation_heatmap(
        mean_corr, dirs["figures"] / "corr_boundary_frequency_vs_metrics.png"
    )
    save_group_boxplot(
        group_assignments, dirs["figures"] / "boundary_freq_groups_boxplot.png"
    )
    save_regression_figure(
        mean_regression, dirs["figures"] / "regression_r2_comparison.png"
    )

    write_readme(args.exp_dir, mean_corr, mean_regression, warnings)
    write_logs(dirs, df, corr, group_summary, regression, warnings)
    assert_outputs(dirs)

    print("Experiment 04 completed.")
    print(f"merged_rows: {len(df)}")
    print(f"correlation_rows: {len(corr)}")
    print(f"group_summary_rows: {len(group_summary)}")
    print(f"regression_rows: {len(regression)}")
    print(f"warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
