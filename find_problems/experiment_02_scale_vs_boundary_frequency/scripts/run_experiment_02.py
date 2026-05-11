from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
MPLCONFIG_DIR = EXPERIMENT_ROOT / ".mplconfig" / f"run_{os.getpid()}"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
for stale_lock in MPLCONFIG_DIR.glob("*.matplotlib-lock"):
    try:
        stale_lock.unlink()
    except OSError:
        pass
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
from matplotlib import cbook


@contextlib.contextmanager
def _no_lock_path(path):
    yield


cbook._lock_path = _no_lock_path
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


MODALS = ("WL", "NBI")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 02: Scale vs Boundary Frequency distribution."
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--erode-dilate-radius", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--max-points-per-region", type=int, default=256)
    parser.add_argument("--high-frequency-outer-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=EXPERIMENT_ROOT / "figures")
    parser.add_argument(
        "--visual-check-dir", type=Path, default=EXPERIMENT_ROOT / "visual_checks"
    )
    return parser.parse_args()


def resample_filter(name: str) -> int:
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def read_gray_image(path: Path, image_size: int) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as img:
        width, height = img.size
        img = img.convert("L").resize(
            (image_size, image_size), resample_filter("BILINEAR")
        )
        arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr, width, height


def read_rgb_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(
            (image_size, image_size), resample_filter("BILINEAR")
        )
        arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def read_mask(path: Path, image_size: int, threshold: int) -> np.ndarray:
    with Image.open(path) as mask:
        mask = mask.convert("L").resize(
            (image_size, image_size), resample_filter("NEAREST")
        )
        arr = np.asarray(mask, dtype=np.uint8)
    return arr > threshold


def disk_structure(radius: int) -> np.ndarray:
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (xx * xx + yy * yy) <= radius * radius


def define_regions(mask: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structure = disk_structure(radius)
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    dilated = ndi.binary_dilation(mask, structure=structure, border_value=0)
    dilated_outer = ndi.binary_dilation(
        mask, structure=disk_structure(radius * 2), border_value=0
    )

    interior = eroded
    boundary_band = dilated & ~eroded
    near_background = dilated_outer & ~dilated
    return interior, boundary_band, near_background


def high_frequency_mask(size: int, outer_ratio: float) -> np.ndarray:
    if not 0.0 < outer_ratio < 1.0:
        raise ValueError("--high-frequency-outer-ratio must be in (0, 1).")

    coords = np.arange(size, dtype=np.float64) - (size // 2)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    cutoff = radius.max() * (1.0 - outer_ratio)
    return radius >= cutoff


def global_high_frequency_ratio(gray: np.ndarray, high_mask: np.ndarray) -> tuple[float, float, float]:
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spectrum) ** 2
    full_power = float(power.sum())
    high_power = float(power[high_mask].sum())
    hfr = high_power / full_power if full_power > 0 else math.nan
    return hfr, full_power, high_power


def sample_points(region: np.ndarray, max_points: int) -> np.ndarray:
    coords = np.argwhere(region)
    if len(coords) == 0:
        return coords
    if len(coords) <= max_points:
        return coords
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=np.int64)
    return coords[indices]


def extract_patches(gray: np.ndarray, points: np.ndarray, patch_size: int) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, patch_size, patch_size), dtype=np.float64)

    before = patch_size // 2
    after = patch_size - before
    padded = np.pad(gray, ((before, after), (before, after)), mode="reflect")
    offsets = np.arange(patch_size)

    y = points[:, 0] + before
    x = points[:, 1] + before
    yy = y[:, None, None] + offsets[None, :, None] - before
    xx = x[:, None, None] + offsets[None, None, :] - before
    return padded[yy, xx]


def patch_hfr_values(
    gray: np.ndarray,
    region: np.ndarray,
    patch_size: int,
    max_points: int,
    patch_high_mask: np.ndarray,
) -> tuple[float, float, int]:
    points = sample_points(region, max_points)
    if len(points) == 0:
        return math.nan, math.nan, 0

    patches = extract_patches(gray, points, patch_size)
    spectra = np.fft.fftshift(np.fft.fft2(patches, axes=(1, 2)), axes=(1, 2))
    power = np.abs(spectra) ** 2
    full_power = power.sum(axis=(1, 2))
    high_power = power[:, patch_high_mask].sum(axis=1)
    valid = full_power > 0
    if not np.any(valid):
        return math.nan, math.nan, int(len(points))

    hfr = high_power[valid] / full_power[valid]
    return float(np.mean(hfr)), float(np.std(hfr, ddof=0)), int(len(points))


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    visual_examples: list[dict[str, object]] = []
    image_high_mask = high_frequency_mask(args.image_size, args.high_frequency_outer_ratio)
    patch_high_mask = high_frequency_mask(args.patch_size, args.high_frequency_outer_ratio)

    for modal in MODALS:
        for split in SPLITS:
            image_dir = args.dataset_root / modal / split / "images"
            mask_dir = args.dataset_root / modal / split / "masks"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing image directory: {image_dir}")
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

            image_paths = sorted(image_dir.glob("*.jpg"))
            if args.limit_per_split is not None:
                image_paths = image_paths[: args.limit_per_split]

            for image_path in image_paths:
                mask_path = mask_dir / f"{image_path.stem}.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")

                gray, width, height = read_gray_image(image_path, args.image_size)
                mask = read_mask(mask_path, args.image_size, args.mask_threshold)
                interior, boundary_band, near_background = define_regions(
                    mask, args.erode_dilate_radius
                )

                scale = float(mask.mean())
                global_hfr, full_power, high_power = global_high_frequency_ratio(
                    gray, image_high_mask
                )
                interior_hfr, interior_std, interior_points = patch_hfr_values(
                    gray,
                    interior,
                    args.patch_size,
                    args.max_points_per_region,
                    patch_high_mask,
                )
                boundary_hfr, boundary_std, boundary_points = patch_hfr_values(
                    gray,
                    boundary_band,
                    args.patch_size,
                    args.max_points_per_region,
                    patch_high_mask,
                )
                near_bg_hfr, near_bg_std, near_bg_points = patch_hfr_values(
                    gray,
                    near_background,
                    args.patch_size,
                    args.max_points_per_region,
                    patch_high_mask,
                )

                if np.isfinite(boundary_hfr) and np.isfinite(interior_hfr) and np.isfinite(near_bg_hfr):
                    boundary_freq_gap = abs(boundary_hfr - interior_hfr) + abs(
                        boundary_hfr - near_bg_hfr
                    )
                else:
                    boundary_freq_gap = math.nan

                rows.append(
                    {
                        "image": image_path.stem,
                        "modal": modal,
                        "split": split,
                        "image_path": str(image_path.relative_to(REPO_ROOT)),
                        "mask_path": str(mask_path.relative_to(REPO_ROOT)),
                        "width": width,
                        "height": height,
                        "scale": scale,
                        "global_hfr": global_hfr,
                        "full_power": full_power,
                        "high_power": high_power,
                        "interior_hfr": interior_hfr,
                        "boundary_hfr": boundary_hfr,
                        "near_background_hfr": near_bg_hfr,
                        "interior_freq_std": interior_std,
                        "boundary_freq_std": boundary_std,
                        "near_background_freq_std": near_bg_std,
                        "boundary_freq_gap": boundary_freq_gap,
                        "interior_points": interior_points,
                        "boundary_points": boundary_points,
                        "near_background_points": near_bg_points,
                    }
                )

                if split == "test" and len([x for x in visual_examples if x["modal"] == modal]) < 3:
                    visual_examples.append(
                        {
                            "modal": modal,
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "interior": interior,
                            "boundary_band": boundary_band,
                            "near_background": near_background,
                        }
                    )

    return rows, visual_examples


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            count=("image", "count"),
            scale_mean=("scale", "mean"),
            scale_std=("scale", "std"),
            scale_var=("scale", "var"),
            global_hfr_mean=("global_hfr", "mean"),
            global_hfr_std=("global_hfr", "std"),
            global_hfr_var=("global_hfr", "var"),
            interior_hfr_mean=("interior_hfr", "mean"),
            interior_hfr_std=("interior_hfr", "std"),
            boundary_hfr_mean=("boundary_hfr", "mean"),
            boundary_hfr_std=("boundary_hfr", "std"),
            near_background_hfr_mean=("near_background_hfr", "mean"),
            near_background_hfr_std=("near_background_hfr", "std"),
            boundary_freq_std_mean=("boundary_freq_std", "mean"),
            boundary_freq_gap_mean=("boundary_freq_gap", "mean"),
            boundary_freq_gap_std=("boundary_freq_gap", "std"),
        )
        .sort_values(group_cols)
    )
    return summary


def plot_scale_vs_boundary(df: pd.DataFrame, figure_path: Path) -> None:
    colors = {"WL": "#d62728", "NBI": "#1f77b4"}
    plt.figure(figsize=(8, 6), dpi=160)
    ax = plt.gca()

    for modal in MODALS:
        modal_df = df[df["modal"] == modal]
        ax.scatter(
            modal_df["scale"],
            modal_df["boundary_hfr"],
            s=10,
            alpha=0.35,
            c=colors[modal],
            label=modal,
            edgecolors="none",
        )
        ax.scatter(
            [modal_df["scale"].mean()],
            [modal_df["boundary_hfr"].mean()],
            s=120,
            c=colors[modal],
            marker="X",
            edgecolors="black",
            linewidths=0.8,
            label=f"{modal} mean",
        )

    ax.set_title("Scale vs Boundary Frequency Distribution")
    ax.set_xlabel("Lesion scale")
    ax.set_ylabel("Boundary high-frequency ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_boundary_distributions(df: pd.DataFrame, figure_path: Path) -> None:
    metrics = ["boundary_hfr", "boundary_freq_std", "boundary_freq_gap"]
    titles = ["Boundary HFR", "Boundary HFR Std", "Boundary Frequency Gap"]
    colors = {"WL": "#d62728", "NBI": "#1f77b4"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=160)
    for ax, metric, title in zip(axes, metrics, titles):
        data = [df[df["modal"] == modal][metric].dropna().to_numpy() for modal in MODALS]
        box = ax.boxplot(data, labels=MODALS, patch_artist=True, showfliers=False)
        for patch, modal in zip(box["boxes"], MODALS):
            patch.set_facecolor(colors[modal])
            patch.set_alpha(0.45)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("WL/NBI Boundary Frequency Distribution")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_global_vs_boundary(df: pd.DataFrame, figure_path: Path) -> None:
    colors = {"WL": "#d62728", "NBI": "#1f77b4"}
    plt.figure(figsize=(8, 6), dpi=160)
    ax = plt.gca()

    for modal in MODALS:
        modal_df = df[df["modal"] == modal]
        ax.scatter(
            modal_df["global_hfr"],
            modal_df["boundary_hfr"],
            s=10,
            alpha=0.35,
            c=colors[modal],
            label=modal,
            edgecolors="none",
        )

    ax.set_title("Global vs Boundary Frequency")
    ax.set_xlabel("Global high-frequency ratio")
    ax.set_ylabel("Boundary high-frequency ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def make_region_overlay(
    image_path: Path,
    image_size: int,
    interior: np.ndarray,
    boundary_band: np.ndarray,
    near_background: np.ndarray,
) -> np.ndarray:
    rgb = read_rgb_image(image_path, image_size)
    overlay = rgb.copy()
    overlay[near_background] = 0.55 * overlay[near_background] + 0.45 * np.array([0.1, 0.35, 1.0])
    overlay[interior] = 0.55 * overlay[interior] + 0.45 * np.array([0.0, 0.8, 0.25])
    overlay[boundary_band] = 0.45 * overlay[boundary_band] + 0.55 * np.array([1.0, 0.9, 0.0])
    return np.clip(overlay, 0.0, 1.0)


def plot_visual_checks(
    examples: list[dict[str, object]],
    image_size: int,
    figure_path: Path,
) -> None:
    if not examples:
        return

    cols = 3
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=160)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, example in zip(axes_arr, examples):
        overlay = make_region_overlay(
            example["image_path"],
            image_size,
            example["interior"],
            example["boundary_band"],
            example["near_background"],
        )
        ax.imshow(overlay)
        ax.set_title(f"{example['modal']} {example['image_path'].stem}")
        ax.axis("off")

    for ax in axes_arr[len(examples) :]:
        ax.axis("off")

    fig.suptitle("Region definitions: green=interior, yellow=boundary, blue=near background")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def validate_outputs(df: pd.DataFrame, limit_per_split: int | None) -> None:
    if limit_per_split is None:
        expected_total = 5600
        if len(df) != expected_total:
            raise AssertionError(f"Expected {expected_total} rows, got {len(df)}.")

        expected_by_modal = {"WL": 2800, "NBI": 2800}
        actual_by_modal = df.groupby("modal").size().to_dict()
        if actual_by_modal != expected_by_modal:
            raise AssertionError(f"Unexpected modal counts: {actual_by_modal}")

        expected_by_split = {"train": 1960, "val": 420, "test": 420}
        for modal in MODALS:
            actual = df[df["modal"] == modal].groupby("split").size().to_dict()
            if actual != expected_by_split:
                raise AssertionError(f"Unexpected split counts for {modal}: {actual}")

    bounded_cols = [
        "scale",
        "global_hfr",
        "interior_hfr",
        "boundary_hfr",
        "near_background_hfr",
    ]
    for col in bounded_cols:
        finite = df[col].dropna()
        if not finite.between(0.0, 1.0).all():
            raise AssertionError(f"{col} contains values outside [0, 1].")

    if not (df["boundary_freq_std"].dropna() >= 0).all():
        raise AssertionError("boundary_freq_std must be non-negative.")
    if not (df["boundary_freq_gap"].dropna() >= 0).all():
        raise AssertionError("boundary_freq_gap must be non-negative.")
    if not (df["boundary_points"] > 0).all():
        raise AssertionError("boundary_points must be positive for all samples.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.visual_check_dir.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    rows, visual_examples = collect_rows(args)
    df = pd.DataFrame(rows)
    validate_outputs(df, args.limit_per_split)

    metrics_path = args.output_dir / "boundary_frequency_metrics.csv"
    summary_modal_path = args.output_dir / "boundary_frequency_summary_by_modal.csv"
    summary_split_path = args.output_dir / "boundary_frequency_summary_by_modal_split.csv"
    figure_scale_boundary_path = args.figure_dir / "scale_vs_boundary_frequency.png"
    figure_dist_path = args.figure_dir / "wl_nbi_boundary_frequency_distribution.png"
    figure_global_boundary_path = args.figure_dir / "global_vs_boundary_frequency.png"
    visual_check_path = args.visual_check_dir / "region_definition_examples.png"

    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summarize(df, ["modal"]).to_csv(summary_modal_path, index=False, encoding="utf-8-sig")
    summarize(df, ["modal", "split"]).to_csv(
        summary_split_path, index=False, encoding="utf-8-sig"
    )
    plot_scale_vs_boundary(df, figure_scale_boundary_path)
    plot_boundary_distributions(df, figure_dist_path)
    plot_global_vs_boundary(df, figure_global_boundary_path)
    plot_visual_checks(visual_examples, args.image_size, visual_check_path)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_modal_path}")
    print(f"Saved: {summary_split_path}")
    print(f"Saved: {figure_scale_boundary_path}")
    print(f"Saved: {figure_dist_path}")
    print(f"Saved: {figure_global_boundary_path}")
    print(f"Saved: {visual_check_path}")
    print("Experiment 02 completed successfully.")


if __name__ == "__main__":
    main()
