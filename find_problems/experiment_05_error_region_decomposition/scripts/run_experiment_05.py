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
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.cbook as cbook


@contextlib.contextmanager
def _no_lock_path(_path):
    yield


cbook._lock_path = _no_lock_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


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
FAILURE_COLUMNS = [
    "low_dice",
    "low_boundary_iou",
    "high_hd95",
    "high_dice_low_boundary",
    "high_dice_high_hd95",
]

REGION_NORM_COLUMNS = ["interior_FN", "boundary_FN", "boundary_FP", "exterior_FP"]
ERROR_RATIO_COLUMNS = [
    "interior_error_ratio",
    "boundary_error_ratio",
    "exterior_error_ratio",
    "boundary_FN_error_ratio",
    "boundary_FP_error_ratio",
]
MODEL_LABELS = {
    "swin_unet": "Swin-Unet",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net",
}
MODAL_COLORS = {"WL": "#1f77b4", "NBI": "#d62728"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 05: decompose segmentation errors by GT regions."
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
    parser.add_argument(
        "--prediction_root",
        type=Path,
        default=WORKSPACE / "experiment_03_baseline_failure_analysis" / "predictions",
    )
    parser.add_argument("--exp_dir", type=Path, default=EXP_DIR)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--erode_dilate_radius", type=int, default=5)
    parser.add_argument("--max_visual_samples", type=int, default=10)
    return parser.parse_args()


def ensure_dirs(exp_dir):
    dirs = {
        "outputs": exp_dir / "outputs",
        "figures": exp_dir / "figures",
        "visual": exp_dir / "visual_checks",
        "logs": exp_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resample_filter(name):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def read_mask_bool(path, image_size, threshold):
    with Image.open(path) as mask:
        mask = mask.convert("L").resize(
            (image_size, image_size), resample_filter("NEAREST")
        )
        arr = np.asarray(mask, dtype=np.uint8)
    return arr > threshold


def read_rgb(path, image_size):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (image_size, image_size), resample_filter("BILINEAR")
        )
        return np.asarray(image, dtype=np.uint8)


def disk_structure(radius):
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def define_regions(gt_mask, radius):
    structure = disk_structure(radius)
    eroded = ndi.binary_erosion(gt_mask, structure=structure, border_value=0)
    dilated = ndi.binary_dilation(gt_mask, structure=structure, border_value=0)

    interior = eroded
    boundary_band = dilated & ~eroded
    exterior = ~dilated
    boundary_inner = boundary_band & gt_mask
    boundary_outer = boundary_band & ~gt_mask
    return {
        "interior": interior,
        "boundary_band": boundary_band,
        "exterior": exterior,
        "boundary_inner": boundary_inner,
        "boundary_outer": boundary_outer,
        "dilated": dilated,
    }


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def validate_input(df, prediction_root):
    required = [
        "image",
        "modal",
        "split",
        "model",
        "image_path",
        "mask_path",
        *PREDICTION_COLUMNS,
        *FREQUENCY_COLUMNS,
        *FAILURE_COLUMNS,
    ]
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

    missing_predictions = []
    for row in df[["model", "modal", "split", "image"]].itertuples(index=False):
        pred_path = prediction_root / row.model / row.modal / row.split / f"{row.image}.png"
        if not pred_path.exists():
            missing_predictions.append(str(pred_path))
            if len(missing_predictions) >= 10:
                break
    if missing_predictions:
        raise RuntimeError(f"Missing prediction masks: {missing_predictions}")


def ordered_frame(df):
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], EXPECTED_MODELS, ordered=True)
    df["modal"] = pd.Categorical(df["modal"], EXPECTED_MODALITIES, ordered=True)
    return df.sort_values(["modal", "model", "image"]).reset_index(drop=True)


def compute_error_regions(df, args):
    rows = []
    region_cache = {}
    empty_region_counts = {
        "interior": 0,
        "boundary_inner": 0,
        "boundary_outer": 0,
        "exterior": 0,
    }

    for record in df.itertuples(index=False):
        image_id = str(record.image)
        modal = str(record.modal)
        split = str(record.split)
        model = str(record.model)
        cache_key = (modal, split, image_id)

        if cache_key not in region_cache:
            mask_path = WORKSPACE / str(record.mask_path)
            image_path = WORKSPACE / str(record.image_path)
            gt_mask = read_mask_bool(mask_path, args.image_size, args.mask_threshold)
            regions = define_regions(gt_mask, args.erode_dilate_radius)
            region_cache[cache_key] = {
                "gt_mask": gt_mask,
                "regions": regions,
                "mask_path": mask_path,
                "image_path": image_path,
            }
            for key in empty_region_counts:
                if int(regions[key].sum()) == 0:
                    empty_region_counts[key] += 1

        cached = region_cache[cache_key]
        gt_mask = cached["gt_mask"]
        regions = cached["regions"]
        pred_path = (
            args.prediction_root / model / modal / split / f"{image_id}.png"
        )
        pred_mask = read_mask_bool(pred_path, args.image_size, args.mask_threshold)

        fn = gt_mask & ~pred_mask
        fp = pred_mask & ~gt_mask
        total_fn_pixels = int(fn.sum())
        total_fp_pixels = int(fp.sum())
        total_error_pixels = total_fn_pixels + total_fp_pixels

        interior_fn_pixels = int((fn & regions["interior"]).sum())
        boundary_fn_pixels = int((fn & regions["boundary_inner"]).sum())
        boundary_fp_pixels = int((fp & regions["boundary_outer"]).sum())
        exterior_fp_pixels = int((fp & regions["exterior"]).sum())
        boundary_error_pixels = boundary_fn_pixels + boundary_fp_pixels

        interior_area = int(regions["interior"].sum())
        boundary_inner_area = int(regions["boundary_inner"].sum())
        boundary_outer_area = int(regions["boundary_outer"].sum())
        exterior_area = int(regions["exterior"].sum())

        row = {
            "image": image_id,
            "modal": modal,
            "split": split,
            "model": model,
            "prediction_path": str(pred_path.relative_to(WORKSPACE)),
            "image_path": str(cached["image_path"].relative_to(WORKSPACE)),
            "mask_path": str(cached["mask_path"].relative_to(WORKSPACE)),
            "image_size": args.image_size,
            "erode_dilate_radius": args.erode_dilate_radius,
            "interior_area": interior_area,
            "boundary_inner_area": boundary_inner_area,
            "boundary_outer_area": boundary_outer_area,
            "exterior_area": exterior_area,
            "total_fn_pixels": total_fn_pixels,
            "total_fp_pixels": total_fp_pixels,
            "total_error_pixels": total_error_pixels,
            "interior_FN_pixels": interior_fn_pixels,
            "boundary_FN_pixels": boundary_fn_pixels,
            "boundary_FP_pixels": boundary_fp_pixels,
            "exterior_FP_pixels": exterior_fp_pixels,
            "interior_FN": safe_ratio(interior_fn_pixels, interior_area),
            "boundary_FN": safe_ratio(boundary_fn_pixels, boundary_inner_area),
            "boundary_FP": safe_ratio(boundary_fp_pixels, boundary_outer_area),
            "exterior_FP": safe_ratio(exterior_fp_pixels, exterior_area),
            "interior_error_ratio": safe_ratio(interior_fn_pixels, total_error_pixels),
            "boundary_error_ratio": safe_ratio(boundary_error_pixels, total_error_pixels),
            "exterior_error_ratio": safe_ratio(exterior_fp_pixels, total_error_pixels),
            "boundary_FN_error_ratio": safe_ratio(boundary_fn_pixels, total_error_pixels),
            "boundary_FP_error_ratio": safe_ratio(boundary_fp_pixels, total_error_pixels),
            "boundary_fp_fn_balance": (
                safe_ratio(boundary_fp_pixels, boundary_error_pixels)
                if boundary_error_pixels > 0
                else np.nan
            ),
        }

        for col in PREDICTION_COLUMNS + FREQUENCY_COLUMNS + FAILURE_COLUMNS:
            row[col] = getattr(record, col)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], EXPECTED_MODELS, ordered=True)
    out["modal"] = pd.Categorical(out["modal"], EXPECTED_MODALITIES, ordered=True)
    out = out.sort_values(["modal", "model", "image"]).reset_index(drop=True)
    return out, region_cache, empty_region_counts


def summarize_by_group(metrics, group_cols):
    agg_cols = (
        PREDICTION_COLUMNS
        + FREQUENCY_COLUMNS
        + REGION_NORM_COLUMNS
        + ERROR_RATIO_COLUMNS
        + ["boundary_fp_fn_balance", "total_error_pixels"]
    )
    summary = (
        metrics.groupby(group_cols, observed=True)
        .agg(**{f"{col}_mean": (col, "mean") for col in agg_cols})
        .reset_index()
    )
    counts = metrics.groupby(group_cols, observed=True).size().reset_index(name="n")
    summary = counts.merge(summary, on=group_cols, how="left")

    for flag in FAILURE_COLUMNS:
        flag_counts = (
            metrics.groupby(group_cols, observed=True)[flag]
            .sum()
            .reset_index(name=f"{flag}_count")
        )
        summary = summary.merge(flag_counts, on=group_cols, how="left")
    return summary


def save_correlations(metrics, outputs_dir):
    rows = []
    x_cols = FREQUENCY_COLUMNS
    y_cols = REGION_NORM_COLUMNS + ERROR_RATIO_COLUMNS + ["boundary_fp_fn_balance"]
    for (model, modal), group in metrics.groupby(["model", "modal"], observed=True):
        for x_col in x_cols:
            for y_col in y_cols:
                pair = group[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "model": model,
                        "modal": modal,
                        "frequency_metric": x_col,
                        "error_metric": y_col,
                        "n": len(pair),
                        "spearman": pair[x_col].corr(pair[y_col], method="spearman"),
                        "pearson": pair[x_col].corr(pair[y_col], method="pearson"),
                    }
                )
    corr = pd.DataFrame(rows)
    corr.to_csv(outputs_dir / "error_region_correlation.csv", index=False)
    return corr


def save_failure_summaries(metrics, outputs_dir):
    common = (
        metrics.groupby(["modal", "image", "split"], as_index=False, observed=True)
        .agg(
            model_count=("model", "count"),
            low_dice_count=("low_dice", "sum"),
            low_boundary_iou_count=("low_boundary_iou", "sum"),
            high_hd95_count=("high_hd95", "sum"),
            high_dice_low_boundary_count=("high_dice_low_boundary", "sum"),
            Dice_mean=("Dice", "mean"),
            Boundary_IoU_mean=("Boundary_IoU", "mean"),
            HD95_mean=("HD95", "mean"),
            interior_error_ratio_mean=("interior_error_ratio", "mean"),
            boundary_error_ratio_mean=("boundary_error_ratio", "mean"),
            exterior_error_ratio_mean=("exterior_error_ratio", "mean"),
            boundary_FN_error_ratio_mean=("boundary_FN_error_ratio", "mean"),
            boundary_FP_error_ratio_mean=("boundary_FP_error_ratio", "mean"),
            boundary_fp_fn_balance_mean=("boundary_fp_fn_balance", "mean"),
            scale=("scale", "first"),
            global_hfr=("global_hfr", "first"),
            boundary_hfr=("boundary_hfr", "first"),
            boundary_freq_std=("boundary_freq_std", "first"),
            boundary_freq_gap=("boundary_freq_gap", "first"),
        )
    )
    common["common_low_dice"] = common["low_dice_count"] == common["model_count"]
    common["common_low_boundary_iou"] = (
        common["low_boundary_iou_count"] == common["model_count"]
    )
    common["common_high_hd95"] = common["high_hd95_count"] == common["model_count"]
    common_selected = common[
        common["common_low_dice"]
        | common["common_low_boundary_iou"]
        | common["common_high_hd95"]
    ].sort_values(
        ["modal", "boundary_error_ratio_mean", "Boundary_IoU_mean"],
        ascending=[True, False, True],
    )
    common_selected.to_csv(outputs_dir / "common_failure_error_summary.csv", index=False)

    high_cases = metrics[metrics["high_dice_low_boundary"]].copy()
    high_cases.sort_values(
        ["modal", "boundary_error_ratio", "Boundary_IoU", "Dice"],
        ascending=[True, False, True, False],
    ).to_csv(outputs_dir / "high_dice_low_boundary_error_cases.csv", index=False)

    if high_cases.empty:
        high_summary = pd.DataFrame()
    else:
        high_summary = summarize_by_group(high_cases, ["model", "modal"])
    high_summary.to_csv(
        outputs_dir / "high_dice_low_boundary_error_summary.csv", index=False
    )
    return common_selected, high_cases, high_summary


def plot_model_modal_summary(summary, fig_path):
    summary = summary.sort_values(["modal", "model"]).copy()
    labels = [
        f"{MODEL_LABELS[str(row.model)]}\n{row.modal}" for row in summary.itertuples()
    ]
    x = np.arange(len(summary))
    interior = summary["interior_error_ratio_mean"].to_numpy()
    boundary = summary["boundary_error_ratio_mean"].to_numpy()
    exterior = summary["exterior_error_ratio_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x, interior, label="interior_FN", color="#8da0cb")
    ax.bar(x, boundary, bottom=interior, label="boundary_FN+FP", color="#fc8d62")
    ax.bar(x, exterior, bottom=interior + boundary, label="exterior_FP", color="#66c2a5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean fraction of total error pixels")
    ax.set_title("Error Region Composition by Model and Modality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def plot_modal_composition(summary, fig_path):
    summary = summary.sort_values("modal").copy()
    x = np.arange(len(summary))
    interior = summary["interior_error_ratio_mean"].to_numpy()
    boundary = summary["boundary_error_ratio_mean"].to_numpy()
    exterior = summary["exterior_error_ratio_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.bar(x, interior, label="interior_FN", color="#8da0cb")
    ax.bar(x, boundary, bottom=interior, label="boundary_FN+FP", color="#fc8d62")
    ax.bar(x, exterior, bottom=interior + boundary, label="exterior_FP", color="#66c2a5")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["modal"].astype(str))
    ax.set_ylabel("Mean fraction of total error pixels")
    ax.set_title("Error Region Composition by Modality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def plot_boundary_balance(metrics, fig_path):
    summary = summarize_by_group(metrics, ["model", "modal"])
    summary = summary.sort_values(["modal", "model"])
    labels = [
        f"{MODEL_LABELS[str(row.model)]}\n{row.modal}" for row in summary.itertuples()
    ]
    x = np.arange(len(summary))
    fn = summary["boundary_FN_error_ratio_mean"].to_numpy()
    fp = summary["boundary_FP_error_ratio_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.0))
    width = 0.38
    ax.bar(x - width / 2, fn, width=width, label="boundary_FN / total error", color="#fdb462")
    ax.bar(x + width / 2, fp, width=width, label="boundary_FP / total error", color="#b3de69")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean fraction of total error pixels")
    ax.set_title("Boundary FN vs Boundary FP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def plot_error_vs_frequency(metrics, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for modal in EXPECTED_MODALITIES:
        subset = metrics[metrics["modal"].astype(str) == modal]
        axes[0].scatter(
            subset["boundary_freq_std"],
            subset["boundary_error_ratio"],
            s=12,
            alpha=0.28,
            color=MODAL_COLORS[modal],
            label=modal,
        )
        axes[1].scatter(
            subset["boundary_freq_gap"],
            subset["boundary_error_ratio"],
            s=12,
            alpha=0.28,
            color=MODAL_COLORS[modal],
            label=modal,
        )
    axes[0].set_xlabel("boundary_freq_std")
    axes[1].set_xlabel("boundary_freq_gap")
    for ax in axes:
        ax.set_ylabel("boundary_error_ratio")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Boundary Frequency vs Boundary Error Ratio")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def plot_high_dice_low_boundary(high_summary, fig_path):
    if high_summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No high Dice low Boundary IoU cases", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)
        return

    high_summary = high_summary.sort_values(["modal", "model"])
    labels = [
        f"{MODEL_LABELS[str(row.model)]}\n{row.modal}" for row in high_summary.itertuples()
    ]
    x = np.arange(len(high_summary))
    interior = high_summary["interior_error_ratio_mean"].to_numpy()
    boundary = high_summary["boundary_error_ratio_mean"].to_numpy()
    exterior = high_summary["exterior_error_ratio_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x, interior, label="interior_FN", color="#8da0cb")
    ax.bar(x, boundary, bottom=interior, label="boundary_FN+FP", color="#fc8d62")
    ax.bar(x, exterior, bottom=interior + boundary, label="exterior_FP", color="#66c2a5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean fraction of total error pixels")
    ax.set_title("High Dice Low Boundary Error Composition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def overlay_mask(image, mask, color, alpha=0.42):
    out = image.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_region_map(image, gt_mask, pred_mask, regions):
    fn = gt_mask & ~pred_mask
    fp = pred_mask & ~gt_mask
    out = image.astype(np.float32).copy() * 0.45
    out = out.astype(np.uint8)
    interior_fn = fn & regions["interior"]
    boundary_fn = fn & regions["boundary_inner"]
    boundary_fp = fp & regions["boundary_outer"]
    exterior_fp = fp & regions["exterior"]
    out[interior_fn] = np.array([75, 120, 255], dtype=np.uint8)
    out[boundary_fn] = np.array([255, 220, 60], dtype=np.uint8)
    out[boundary_fp] = np.array([255, 70, 210], dtype=np.uint8)
    out[exterior_fp] = np.array([255, 60, 60], dtype=np.uint8)
    return out


def draw_caption(tile, caption):
    tile = Image.fromarray(tile)
    canvas = Image.new("RGB", (tile.width, tile.height + 22), "white")
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, tile.height + 4), caption[:38], fill=(0, 0, 0))
    return np.asarray(canvas)


def make_visual_grid(selected, args, out_path, title):
    if selected.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    cols = ["Image+GT", "Prediction", "FN", "FP", "Error regions"]
    n = len(selected)
    fig, axes = plt.subplots(n, len(cols), figsize=(15, max(2.6, 2.4 * n)))
    if n == 1:
        axes = np.asarray([axes])

    for row_idx, record in enumerate(selected.itertuples(index=False)):
        image = read_rgb(WORKSPACE / record.image_path, args.image_size)
        gt_mask = read_mask_bool(WORKSPACE / record.mask_path, args.image_size, args.mask_threshold)
        pred_mask = read_mask_bool(
            WORKSPACE / record.prediction_path, args.image_size, args.mask_threshold
        )
        regions = define_regions(gt_mask, args.erode_dilate_radius)
        fn = gt_mask & ~pred_mask
        fp = pred_mask & ~gt_mask

        tiles = [
            overlay_mask(image, gt_mask, (255, 0, 0), alpha=0.36),
            overlay_mask(image, pred_mask, (0, 220, 80), alpha=0.36),
            overlay_mask(image, fn, (255, 0, 0), alpha=0.62),
            overlay_mask(image, fp, (0, 210, 255), alpha=0.62),
            make_error_region_map(image, gt_mask, pred_mask, regions),
        ]
        row_label = (
            f"{record.modal} {MODEL_LABELS.get(str(record.model), str(record.model))} "
            f"BErr {record.boundary_error_ratio:.2f}"
        )
        for col_idx, tile in enumerate(tiles):
            ax = axes[row_idx, col_idx]
            ax.imshow(tile)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_visual_checks(metrics, args, visual_dir):
    for modal in EXPECTED_MODALITIES:
        subset = metrics[metrics["modal"].astype(str) == modal]
        selected = (
            subset.sort_values(
                ["boundary_error_ratio", "Boundary_IoU", "HD95"],
                ascending=[False, True, False],
            )
            .drop_duplicates(["image"])
            .head(args.max_visual_samples)
        )
        make_visual_grid(
            selected,
            args,
            visual_dir / f"error_region_examples_{modal}.png",
            f"{modal} representative error region examples",
        )

        high = subset[subset["high_dice_low_boundary"]]
        selected_high = (
            high.sort_values(
                ["boundary_error_ratio", "Boundary_IoU", "Dice"],
                ascending=[False, True, False],
            )
            .drop_duplicates(["image"])
            .head(args.max_visual_samples)
        )
        make_visual_grid(
            selected_high,
            args,
            visual_dir / f"high_dice_low_boundary_errors_{modal}.png",
            f"{modal} high Dice low Boundary IoU errors",
        )


def validate_outputs(metrics, dirs):
    if len(metrics) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} metric rows, found {len(metrics)}")
    counts = metrics.groupby(["model", "modal"], observed=True).size()
    bad_counts = counts[counts != EXPECTED_ROWS_PER_PAIR]
    if not bad_counts.empty:
        raise RuntimeError(f"Bad model-modal row counts: {bad_counts.to_dict()}")

    ratio_cols = REGION_NORM_COLUMNS + ERROR_RATIO_COLUMNS
    for col in ratio_cols:
        values = metrics[col].dropna()
        if not ((values >= -1e-9) & (values <= 1 + 1e-9)).all():
            raise RuntimeError(f"{col} has values outside [0, 1]")

    composition_sum = (
        metrics["interior_error_ratio"]
        + metrics["boundary_error_ratio"]
        + metrics["exterior_error_ratio"]
    )
    if not (composition_sum <= 1 + 1e-9).all():
        raise RuntimeError("Error composition ratios exceed 1")

    balance = metrics["boundary_fp_fn_balance"].dropna()
    if not ((balance >= -1e-9) & (balance <= 1 + 1e-9)).all():
        raise RuntimeError("boundary_fp_fn_balance has values outside [0, 1]")

    expected_files = [
        dirs["outputs"] / "error_region_metrics.csv",
        dirs["outputs"] / "error_region_summary_by_model_modal.csv",
        dirs["outputs"] / "error_region_summary_by_modal.csv",
        dirs["outputs"] / "error_region_correlation.csv",
        dirs["outputs"] / "common_failure_error_summary.csv",
        dirs["outputs"] / "high_dice_low_boundary_error_summary.csv",
        dirs["figures"] / "error_region_summary_by_model_modal.png",
        dirs["figures"] / "error_region_composition_by_modal.png",
        dirs["figures"] / "boundary_fn_fp_balance.png",
        dirs["figures"] / "error_region_vs_boundary_frequency.png",
        dirs["figures"] / "high_dice_low_boundary_error_composition.png",
        dirs["visual"] / "error_region_examples_WL.png",
        dirs["visual"] / "error_region_examples_NBI.png",
        dirs["visual"] / "high_dice_low_boundary_errors_WL.png",
        dirs["visual"] / "high_dice_low_boundary_errors_NBI.png",
    ]
    missing_or_empty = [
        path for path in expected_files if (not path.exists()) or path.stat().st_size == 0
    ]
    if missing_or_empty:
        raise RuntimeError(f"Missing or empty output files: {missing_or_empty}")


def format_modal_table(summary_by_modal):
    lines = []
    for row in summary_by_modal.sort_values("modal").itertuples(index=False):
        lines.append(
            f"{row.modal}: boundary_error_ratio {row.boundary_error_ratio_mean:.4f}, "
            f"interior_error_ratio {row.interior_error_ratio_mean:.4f}, "
            f"exterior_error_ratio {row.exterior_error_ratio_mean:.4f}, "
            f"boundary_FP_balance {row.boundary_fp_fn_balance_mean:.4f}"
        )
    return "\n".join(lines)


def write_readme(exp_dir, summary_by_modal, common_summary, high_summary):
    modal_text = format_modal_table(summary_by_modal)
    common_counts = common_summary.groupby("modal", observed=True).size().to_dict()
    high_text = "无 high Dice low Boundary IoU 样本。"
    if not high_summary.empty:
        lines = []
        modal_high = summarize_by_group(
            pd.read_csv(exp_dir / "outputs" / "high_dice_low_boundary_error_cases.csv"),
            ["modal"],
        )
        for row in modal_high.sort_values("modal").itertuples(index=False):
            lines.append(
                f"{row.modal}: n {row.n}, boundary_error_ratio "
                f"{row.boundary_error_ratio_mean:.4f}, exterior_error_ratio "
                f"{row.exterior_error_ratio_mean:.4f}"
            )
        high_text = "\n".join(lines)

    text = f"""# 实验五：区域错误 vs 边界错误拆分

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
{modal_text}
```

共同失败样本数量：

```text
WL: {int(common_counts.get('WL', 0))}
NBI: {int(common_counts.get('NBI', 0))}
```

高 Dice 低 Boundary IoU 样本中的错误组成：

```text
{high_text}
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
"""
    (exp_dir / "README.md").write_text(text, encoding="utf-8")


def write_logs(dirs, metrics, empty_region_counts, missing_prediction_count=0):
    ratio_cols = REGION_NORM_COLUMNS + ERROR_RATIO_COLUMNS
    invalid_ratio_count = 0
    for col in ratio_cols:
        values = metrics[col].dropna()
        invalid_ratio_count += int(((values < -1e-9) | (values > 1 + 1e-9)).sum())
    composition_sum = (
        metrics["interior_error_ratio"]
        + metrics["boundary_error_ratio"]
        + metrics["exterior_error_ratio"]
    )
    bad_composition_count = int((composition_sum > 1 + 1e-9).sum())

    lines = [
        "Experiment 05 run summary",
        "",
        f"rows: {len(metrics)}",
        f"missing_prediction_count: {missing_prediction_count}",
        f"invalid_ratio_count: {invalid_ratio_count}",
        f"bad_composition_count: {bad_composition_count}",
        "",
        "empty_region_counts:",
    ]
    for key, value in empty_region_counts.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("model_modal_rows:")
    counts = metrics.groupby(["model", "modal"], observed=True).size()
    for (model, modal), count in counts.items():
        lines.append(f"  {model}/{modal}: {count}")
    (dirs["logs"] / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dirs = ensure_dirs(args.exp_dir)

    df = pd.read_csv(args.input_csv)
    validate_input(df, args.prediction_root)
    df = ordered_frame(df)

    metrics, _region_cache, empty_region_counts = compute_error_regions(df, args)
    metrics.to_csv(dirs["outputs"] / "error_region_metrics.csv", index=False)

    summary_by_model_modal = summarize_by_group(metrics, ["model", "modal"])
    summary_by_modal = summarize_by_group(metrics, ["modal"])
    summary_by_model_modal.to_csv(
        dirs["outputs"] / "error_region_summary_by_model_modal.csv", index=False
    )
    summary_by_modal.to_csv(
        dirs["outputs"] / "error_region_summary_by_modal.csv", index=False
    )

    save_correlations(metrics, dirs["outputs"])
    common_summary, _high_cases, high_summary = save_failure_summaries(
        metrics, dirs["outputs"]
    )

    plot_model_modal_summary(
        summary_by_model_modal, dirs["figures"] / "error_region_summary_by_model_modal.png"
    )
    plot_modal_composition(
        summary_by_modal, dirs["figures"] / "error_region_composition_by_modal.png"
    )
    plot_boundary_balance(metrics, dirs["figures"] / "boundary_fn_fp_balance.png")
    plot_error_vs_frequency(
        metrics, dirs["figures"] / "error_region_vs_boundary_frequency.png"
    )
    plot_high_dice_low_boundary(
        high_summary, dirs["figures"] / "high_dice_low_boundary_error_composition.png"
    )
    save_visual_checks(metrics, args, dirs["visual"])

    write_readme(args.exp_dir, summary_by_modal, common_summary, high_summary)
    write_logs(dirs, metrics, empty_region_counts)
    validate_outputs(metrics, dirs)

    print("Experiment 05 completed.")
    print(f"rows: {len(metrics)}")
    print(f"model_modal_summary_rows: {len(summary_by_model_modal)}")
    print(f"modal_summary_rows: {len(summary_by_modal)}")
    print(f"common_failure_rows: {len(common_summary)}")
    print(f"high_dice_low_boundary_rows: {int(metrics['high_dice_low_boundary'].sum())}")
    print(f"empty_region_counts: {empty_region_counts}")


if __name__ == "__main__":
    main()
