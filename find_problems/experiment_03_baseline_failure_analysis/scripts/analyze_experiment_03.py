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
from PIL import Image


EXPECTED_MODELS = ["swin_unet", "unet", "unetpp", "attention_unet"]
EXPECTED_MODALITIES = ["WL", "NBI"]
EXPECTED_ROWS_PER_PAIR = 420

FREQUENCY_COLUMNS = [
    "scale",
    "global_hfr",
    "boundary_hfr",
    "boundary_freq_std",
    "boundary_freq_gap",
]
PREDICTION_COLUMNS = ["Dice", "IoU", "Boundary_IoU", "HD95", "MAE"]
MODEL_ORDER = ["swin_unet", "unet", "unetpp", "attention_unet"]
MODEL_LABELS = {
    "swin_unet": "Swin-Unet",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net",
}
MODAL_COLORS = {"WL": "#1f77b4", "NBI": "#d62728"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze experiment 03 baseline failures and frequency metrics."
    )
    parser.add_argument("--exp_dir", type=Path, default=EXP_DIR)
    parser.add_argument(
        "--frequency_csv",
        type=Path,
        default=WORKSPACE
        / "experiment_02_scale_vs_boundary_frequency"
        / "outputs"
        / "boundary_frequency_metrics.csv",
    )
    parser.add_argument("--max_visual_samples", type=int, default=12)
    return parser.parse_args()


def ensure_dirs(exp_dir):
    dirs = {
        "merged": exp_dir / "outputs" / "merged_metrics",
        "failure": exp_dir / "outputs" / "failure_cases",
        "figures": exp_dir / "figures",
        "visual": exp_dir / "visual_checks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_prediction_metrics(exp_dir):
    metric_dir = exp_dir / "outputs" / "prediction_metrics"
    files = sorted(metric_dir.glob("*_prediction_metrics.csv"))
    if len(files) != len(EXPECTED_MODELS) * len(EXPECTED_MODALITIES):
        raise RuntimeError(
            f"Expected 8 prediction CSV files, found {len(files)} in {metric_dir}"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        required = {
            "image",
            "modal",
            "split",
            "model",
            "Dice",
            "IoU",
            "Boundary_IoU",
            "HD95",
            "MAE",
            "pred_fg_ratio",
            "gt_fg_ratio",
        }
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
        frames.append(df)

    pred = pd.concat(frames, ignore_index=True)
    pred["image"] = pred["image"].astype(str)
    pred["modal"] = pred["modal"].astype(str)
    pred["split"] = pred["split"].astype(str)
    pred["model"] = pred["model"].astype(str)
    return pred


def validate_prediction_metrics(pred):
    observed_models = sorted(pred["model"].unique())
    observed_modalities = sorted(pred["modal"].unique())
    if observed_models != sorted(EXPECTED_MODELS):
        raise RuntimeError(f"Unexpected models: {observed_models}")
    if observed_modalities != sorted(EXPECTED_MODALITIES):
        raise RuntimeError(f"Unexpected modalities: {observed_modalities}")

    counts = pred.groupby(["model", "modal"]).size()
    for model in EXPECTED_MODELS:
        for modal in EXPECTED_MODALITIES:
            count = int(counts.get((model, modal), 0))
            if count != EXPECTED_ROWS_PER_PAIR:
                raise RuntimeError(
                    f"{model}/{modal} expected {EXPECTED_ROWS_PER_PAIR} rows, got {count}"
                )

    bounded = ["Dice", "IoU", "Boundary_IoU", "MAE", "pred_fg_ratio", "gt_fg_ratio"]
    for col in bounded:
        invalid = pred[~pred[col].between(0, 1, inclusive="both")]
        if not invalid.empty:
            raise RuntimeError(f"{col} has values outside [0, 1]")
    if (pred["HD95"] < 0).any():
        raise RuntimeError("HD95 has negative values")


def read_frequency_metrics(path):
    freq = pd.read_csv(path)
    required = {
        "image",
        "modal",
        "split",
        "image_path",
        "mask_path",
        "scale",
        "global_hfr",
        "interior_hfr",
        "boundary_hfr",
        "near_background_hfr",
        "boundary_freq_std",
        "boundary_freq_gap",
    }
    missing = required.difference(freq.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    freq["image"] = freq["image"].astype(str)
    freq["modal"] = freq["modal"].astype(str)
    freq["split"] = freq["split"].astype(str)
    return freq


def merge_metrics(pred, freq):
    test_freq = freq[freq["split"] == "test"].copy()
    merged = pred.merge(
        test_freq,
        on=["image", "modal", "split"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_freq"),
    )
    missing = merged[merged["scale"].isna()]
    if not missing.empty:
        sample = missing[["image", "modal", "model"]].head(10).to_dict("records")
        raise RuntimeError(f"Missing frequency rows after merge. Sample: {sample}")

    counts = merged.groupby(["model", "modal"]).size()
    for model in EXPECTED_MODELS:
        for modal in EXPECTED_MODALITIES:
            count = int(counts.get((model, modal), 0))
            if count != EXPECTED_ROWS_PER_PAIR:
                raise RuntimeError(
                    f"Merged {model}/{modal} expected 420 rows, got {count}"
                )
    if len(merged) != EXPECTED_ROWS_PER_PAIR * len(EXPECTED_MODELS) * len(EXPECTED_MODALITIES):
        raise RuntimeError(f"Merged row count expected 3360, got {len(merged)}")
    return merged


def add_failure_flags(df):
    out = df.copy()
    out["low_dice"] = out["Dice"] < 0.5
    out["low_boundary_iou"] = out["Boundary_IoU"] < 0.1
    out["high_hd95"] = out["HD95"] > 100
    out["high_dice_low_boundary"] = (out["Dice"] >= 0.85) & (
        out["Boundary_IoU"] < 0.1
    )
    out["high_dice_high_hd95"] = (out["Dice"] >= 0.85) & (out["HD95"] > 50)
    return out


def save_model_modal_summary(merged, out_path):
    rows = []
    for (model, modal), group in merged.groupby(["model", "modal"]):
        row = {
            "model": model,
            "modal": modal,
            "n": len(group),
        }
        for col in PREDICTION_COLUMNS:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=1)
        row["low_dice_count"] = int(group["low_dice"].sum())
        row["low_boundary_iou_count"] = int(group["low_boundary_iou"].sum())
        row["high_hd95_count"] = int(group["high_hd95"].sum())
        row["high_dice_low_boundary_count"] = int(group["high_dice_low_boundary"].sum())
        row["high_dice_high_hd95_count"] = int(group["high_dice_high_hd95"].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary["modal"] = pd.Categorical(summary["modal"], EXPECTED_MODALITIES, ordered=True)
    summary = summary.sort_values(["modal", "model"])
    summary.to_csv(out_path, index=False)
    return summary


def save_correlations(merged, out_path):
    rows = []
    for (model, modal), group in merged.groupby(["model", "modal"]):
        for x in FREQUENCY_COLUMNS:
            for y in PREDICTION_COLUMNS:
                rows.append(
                    {
                        "model": model,
                        "modal": modal,
                        "frequency_metric": x,
                        "prediction_metric": y,
                        "spearman": group[x].corr(group[y], method="spearman"),
                        "pearson": group[x].corr(group[y], method="pearson"),
                    }
                )
    corr = pd.DataFrame(rows)
    corr["model"] = pd.Categorical(corr["model"], MODEL_ORDER, ordered=True)
    corr["modal"] = pd.Categorical(corr["modal"], EXPECTED_MODALITIES, ordered=True)
    corr = corr.sort_values(["modal", "model", "frequency_metric", "prediction_metric"])
    corr.to_csv(out_path, index=False)
    return corr


def save_frequency_group_summary(merged, out_path):
    rows = []
    for (model, modal), group in merged.groupby(["model", "modal"]):
        for metric in FREQUENCY_COLUMNS:
            bins = pd.qcut(
                group[metric],
                q=3,
                labels=["low", "mid", "high"],
                duplicates="drop",
            )
            temp = group.assign(freq_group=bins)
            for freq_group, subgroup in temp.groupby("freq_group", observed=True):
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
    summary.to_csv(out_path, index=False)
    return summary


def save_failure_cases(merged, failure_dir):
    common_columns = [
        "image",
        "modal",
        "split",
        "model",
        "Dice",
        "IoU",
        "Boundary_IoU",
        "HD95",
        "MAE",
        "scale",
        "global_hfr",
        "boundary_hfr",
        "boundary_freq_std",
        "boundary_freq_gap",
        "pred_fg_ratio",
        "gt_fg_ratio",
    ]
    low_dice = merged[merged["low_dice"]].sort_values(
        ["modal", "Dice", "Boundary_IoU", "HD95"], ascending=[True, True, True, False]
    )
    low_boundary = merged[merged["low_boundary_iou"]].sort_values(
        ["modal", "Boundary_IoU", "HD95"], ascending=[True, True, False]
    )
    high_hd95 = merged[merged["high_hd95"]].sort_values(
        ["modal", "HD95", "Boundary_IoU"], ascending=[True, False, True]
    )
    high_dice_low_boundary = merged[merged["high_dice_low_boundary"]].sort_values(
        ["modal", "Boundary_IoU", "Dice"], ascending=[True, True, False]
    )
    high_dice_high_hd95 = merged[merged["high_dice_high_hd95"]].sort_values(
        ["modal", "HD95", "Dice"], ascending=[True, False, False]
    )

    low_dice[common_columns].to_csv(failure_dir / "all_low_dice_cases.csv", index=False)
    low_boundary[common_columns].to_csv(
        failure_dir / "all_low_boundary_iou_cases.csv", index=False
    )
    high_hd95[common_columns].to_csv(failure_dir / "all_high_hd95_cases.csv", index=False)
    high_dice_low_boundary[common_columns].to_csv(
        failure_dir / "high_dice_low_boundary_cases.csv", index=False
    )
    high_dice_high_hd95[common_columns].to_csv(
        failure_dir / "high_dice_high_hd95_cases.csv", index=False
    )

    grouped = (
        merged.groupby(["modal", "image", "split"], as_index=False)
        .agg(
            model_count=("model", "nunique"),
            low_dice_count=("low_dice", "sum"),
            low_boundary_iou_count=("low_boundary_iou", "sum"),
            high_hd95_count=("high_hd95", "sum"),
            mean_Dice=("Dice", "mean"),
            mean_Boundary_IoU=("Boundary_IoU", "mean"),
            mean_HD95=("HD95", "mean"),
            max_HD95=("HD95", "max"),
            scale=("scale", "first"),
            global_hfr=("global_hfr", "first"),
            boundary_hfr=("boundary_hfr", "first"),
            boundary_freq_std=("boundary_freq_std", "first"),
            boundary_freq_gap=("boundary_freq_gap", "first"),
        )
        .assign(
            common_low_dice=lambda d: d["low_dice_count"] == len(EXPECTED_MODELS),
            common_low_boundary_iou=lambda d: d["low_boundary_iou_count"]
            == len(EXPECTED_MODELS),
            common_high_hd95=lambda d: d["high_hd95_count"] == len(EXPECTED_MODELS),
        )
    )
    common = grouped[
        grouped["common_low_dice"]
        | grouped["common_low_boundary_iou"]
        | grouped["common_high_hd95"]
    ].sort_values(
        [
            "modal",
            "common_low_boundary_iou",
            "common_low_dice",
            "mean_Boundary_IoU",
            "mean_Dice",
            "mean_HD95",
        ],
        ascending=[True, False, False, True, True, False],
    )
    common.to_csv(failure_dir / "common_failures_by_modal.csv", index=False)

    return {
        "low_dice": low_dice,
        "low_boundary": low_boundary,
        "high_hd95": high_hd95,
        "high_dice_low_boundary": high_dice_low_boundary,
        "high_dice_high_hd95": high_dice_high_hd95,
        "common": common,
        "common_grouped": grouped,
    }


def bar_positions(labels):
    x = np.arange(len(labels))
    width = 0.36
    return x, width


def save_model_metric_summary_figure(summary, fig_path):
    metrics = ["Dice_mean", "IoU_mean", "Boundary_IoU_mean", "HD95_mean", "MAE_mean"]
    titles = ["Dice", "IoU", "Boundary IoU", "HD95", "MAE"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    models = MODEL_ORDER
    labels = [MODEL_LABELS[m] for m in models]
    x, width = bar_positions(labels)
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        for offset, modal in [(-width / 2, "WL"), (width / 2, "NBI")]:
            values = []
            for model in models:
                match = summary[(summary["model"] == model) & (summary["modal"] == modal)]
                values.append(float(match[metric].iloc[0]))
            ax.bar(x + offset, values, width=width, label=modal, color=MODAL_COLORS[modal])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if title != "HD95":
            ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
    axes[-1].axis("off")
    axes[0].legend()
    fig.suptitle("Model-Modal Test Mean Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_failure_counts_figure(summary, fig_path):
    count_cols = [
        ("low_dice_count", "Dice < 0.5"),
        ("low_boundary_iou_count", "Boundary IoU < 0.1"),
        ("high_hd95_count", "HD95 > 100"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    x, width = bar_positions(labels)
    for ax, (col, title) in zip(axes, count_cols):
        for offset, modal in [(-width / 2, "WL"), (width / 2, "NBI")]:
            values = []
            for model in MODEL_ORDER:
                match = summary[(summary["model"] == model) & (summary["modal"] == modal)]
                values.append(int(match[col].iloc[0]))
            ax.bar(x + offset, values, width=width, label=modal, color=MODAL_COLORS[modal])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Failure Counts by Model and Modality", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_correlation_heatmap(corr, fig_path):
    heat = (
        corr.groupby(["frequency_metric", "prediction_metric"])["spearman"]
        .mean()
        .unstack("prediction_metric")
        .reindex(index=FREQUENCY_COLUMNS, columns=PREDICTION_COLUMNS)
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(heat.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(PREDICTION_COLUMNS)))
    ax.set_yticks(np.arange(len(FREQUENCY_COLUMNS)))
    ax.set_xticklabels(PREDICTION_COLUMNS, rotation=35, ha="right")
    ax.set_yticklabels(FREQUENCY_COLUMNS)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Spearman r")
    ax.set_title("Frequency Metrics vs Prediction Metrics")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_scatter_figures(merged, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for modal in EXPECTED_MODALITIES:
        subset = merged[merged["modal"] == modal]
        axes[0].scatter(
            subset["boundary_hfr"],
            subset["Boundary_IoU"],
            s=12,
            alpha=0.35,
            label=modal,
            color=MODAL_COLORS[modal],
        )
        axes[1].scatter(
            subset["boundary_freq_gap"],
            subset["Boundary_IoU"],
            s=12,
            alpha=0.35,
            label=modal,
            color=MODAL_COLORS[modal],
        )
    axes[0].set_xlabel("boundary_hfr")
    axes[1].set_xlabel("boundary_freq_gap")
    for ax in axes:
        ax.set_ylabel("Boundary_IoU")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Boundary Frequency vs Boundary IoU")
    fig.tight_layout()
    fig.savefig(fig_dir / "boundary_frequency_vs_boundary_iou.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for modal in EXPECTED_MODALITIES:
        subset = merged[merged["modal"] == modal]
        axes[0].scatter(
            subset["boundary_freq_std"],
            subset["HD95"],
            s=12,
            alpha=0.35,
            label=modal,
            color=MODAL_COLORS[modal],
        )
        axes[1].scatter(
            subset["boundary_freq_gap"],
            subset["HD95"],
            s=12,
            alpha=0.35,
            label=modal,
            color=MODAL_COLORS[modal],
        )
    axes[0].set_xlabel("boundary_freq_std")
    axes[1].set_xlabel("boundary_freq_gap")
    for ax in axes:
        ax.set_ylabel("HD95")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Boundary Frequency vs HD95")
    fig.tight_layout()
    fig.savefig(fig_dir / "boundary_frequency_vs_hd95.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for modal in EXPECTED_MODALITIES:
        subset = merged[merged["modal"] == modal]
        highlight = subset["high_dice_low_boundary"]
        ax.scatter(
            subset.loc[~highlight, "Dice"],
            subset.loc[~highlight, "Boundary_IoU"],
            s=12,
            alpha=0.22,
            color=MODAL_COLORS[modal],
            label=f"{modal} normal",
        )
        ax.scatter(
            subset.loc[highlight, "Dice"],
            subset.loc[highlight, "Boundary_IoU"],
            s=24,
            alpha=0.8,
            marker="x",
            color=MODAL_COLORS[modal],
            label=f"{modal} high Dice low boundary",
        )
    ax.axvline(0.85, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Dice")
    ax.set_ylabel("Boundary_IoU")
    ax.set_title("High Dice but Low Boundary Quality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "high_dice_low_boundary_distribution.png", dpi=220)
    plt.close(fig)


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    return Image.open(path).convert("L")


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.38):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    arr = np.asarray(image).astype(np.float32)
    mask_arr = np.asarray(mask) > 0
    color_arr = np.array(color, dtype=np.float32)
    arr[mask_arr] = arr[mask_arr] * (1 - alpha) + color_arr * alpha
    return np.clip(arr, 0, 255).astype(np.uint8)


def get_case_metrics(merged, image, modal, model):
    row = merged[
        (merged["image"] == image) & (merged["modal"] == modal) & (merged["model"] == model)
    ]
    if row.empty:
        return ""
    r = row.iloc[0]
    return f"D {r['Dice']:.2f}  B {r['Boundary_IoU']:.2f}  H {r['HD95']:.0f}"


def save_case_visualization(merged, cases, modal, out_path, max_samples):
    selected = cases[cases["modal"] == modal].copy()
    if "mean_Boundary_IoU" in selected.columns:
        selected = selected.sort_values(
            ["mean_Boundary_IoU", "mean_Dice", "mean_HD95"],
            ascending=[True, True, False],
        )
    elif "Boundary_IoU" in selected.columns:
        selected = selected.sort_values(
            ["Boundary_IoU", "Dice", "HD95"], ascending=[True, False, False]
        )
    selected = selected.drop_duplicates("image").head(max_samples)

    if selected.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"No cases for {modal}", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    n = len(selected)
    cols = ["Image + GT", "GT", "Swin-Unet", "U-Net", "U-Net++", "Attention U-Net"]
    fig, axes = plt.subplots(n, len(cols), figsize=(18, max(2.2, 2.15 * n)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (_, case) in enumerate(selected.iterrows()):
        image_id = str(case["image"])
        image_path = WORKSPACE / "dataset" / modal / "test" / "images" / f"{image_id}.jpg"
        mask_path = WORKSPACE / "dataset" / modal / "test" / "masks" / f"{image_id}.png"
        image = load_rgb(image_path)
        gt = load_mask(mask_path)

        display_image = image.resize((180, 140), Image.Resampling.BILINEAR)
        display_gt = gt.resize((180, 140), Image.Resampling.NEAREST)

        axes[row_idx, 0].imshow(overlay_mask(display_image, display_gt, color=(255, 0, 0)))
        axes[row_idx, 0].set_title(f"{image_id}\nGT overlay", fontsize=8)
        axes[row_idx, 1].imshow(display_gt, cmap="gray")
        axes[row_idx, 1].set_title("GT mask", fontsize=8)

        for col_idx, model in enumerate(MODEL_ORDER, start=2):
            pred_path = (
                EXP_DIR
                / "predictions"
                / model
                / modal
                / "test"
                / f"{image_id}.png"
            )
            pred = load_mask(pred_path)
            pred = pred.resize((180, 140), Image.Resampling.NEAREST)
            axes[row_idx, col_idx].imshow(
                overlay_mask(display_image, pred, color=(0, 210, 70))
            )
            axes[row_idx, col_idx].set_title(
                f"{MODEL_LABELS[model]}\n{get_case_metrics(merged, image_id, modal, model)}",
                fontsize=8,
            )

        for ax in axes[row_idx, :]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_visual_checks(merged, failures, visual_dir, max_samples):
    for modal in EXPECTED_MODALITIES:
        save_case_visualization(
            merged,
            failures["common"],
            modal,
            visual_dir / f"common_failures_{modal}.png",
            max_samples,
        )
        save_case_visualization(
            merged,
            failures["high_dice_low_boundary"],
            modal,
            visual_dir / f"high_dice_low_boundary_{modal}.png",
            max_samples,
        )
        save_case_visualization(
            merged,
            failures["high_hd95"],
            modal,
            visual_dir / f"high_hd95_{modal}.png",
            max_samples,
        )


def write_analysis_summary(merged, summary, failures, out_path):
    lines = []
    lines.append("Experiment 03 analysis summary")
    lines.append("")
    lines.append(f"merged_rows: {len(merged)}")
    lines.append("model_modal_rows:")
    for (model, modal), group in merged.groupby(["model", "modal"]):
        lines.append(f"  {model}/{modal}: {len(group)}")
    lines.append("")
    lines.append("common_failures:")
    common = failures["common"]
    for modal in EXPECTED_MODALITIES:
        subset = common[common["modal"] == modal]
        lines.append(f"  {modal}: {len(subset)} images")
    lines.append("")
    lines.append("high_dice_low_boundary_rows:")
    for modal in EXPECTED_MODALITIES:
        subset = failures["high_dice_low_boundary"][
            failures["high_dice_low_boundary"]["modal"] == modal
        ]
        lines.append(f"  {modal}: {len(subset)} rows / {subset['image'].nunique()} images")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    dirs = ensure_dirs(exp_dir)

    pred = read_prediction_metrics(exp_dir)
    validate_prediction_metrics(pred)
    freq = read_frequency_metrics(args.frequency_csv.resolve())
    merged = add_failure_flags(merge_metrics(pred, freq))

    merged_path = dirs["merged"] / "experiment_03_merged_metrics.csv"
    merged.to_csv(merged_path, index=False)

    summary = save_model_modal_summary(
        merged, dirs["merged"] / "model_modal_summary.csv"
    )
    corr = save_correlations(merged, dirs["merged"] / "correlation_by_model_modal.csv")
    save_frequency_group_summary(
        merged, dirs["merged"] / "frequency_group_summary.csv"
    )
    failures = save_failure_cases(merged, dirs["failure"])

    save_model_metric_summary_figure(
        summary, dirs["figures"] / "model_modal_metric_summary.png"
    )
    save_failure_counts_figure(
        summary, dirs["figures"] / "failure_counts_by_model_modal.png"
    )
    save_correlation_heatmap(
        corr, dirs["figures"] / "correlation_heatmap_spearman.png"
    )
    save_scatter_figures(merged, dirs["figures"])
    save_visual_checks(merged, failures, dirs["visual"], args.max_visual_samples)
    write_analysis_summary(
        merged, summary, failures, dirs["merged"] / "analysis_summary.txt"
    )

    print(f"Saved merged metrics: {merged_path}")
    print(f"Merged rows: {len(merged)}")
    print("Model-modal row counts:")
    for (model, modal), group in merged.groupby(["model", "modal"]):
        print(f"  {model}/{modal}: {len(group)}")
    print("Done.")


if __name__ == "__main__":
    main()
