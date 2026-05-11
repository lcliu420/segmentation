from __future__ import annotations

import argparse
import os
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
MPLCONFIG_DIR = EXPERIMENT_ROOT / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
for stale_lock in MPLCONFIG_DIR.glob("*.matplotlib-lock"):
    try:
        stale_lock.unlink()
    except OSError:
        pass
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MODALS = ("WL", "NBI")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 01: Scale vs Global Frequency distribution."
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--high-frequency-outer-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=EXPERIMENT_ROOT / "figures")
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


def read_mask(path: Path, image_size: int, threshold: int) -> np.ndarray:
    with Image.open(path) as mask:
        mask = mask.convert("L").resize(
            (image_size, image_size), resample_filter("NEAREST")
        )
        arr = np.asarray(mask, dtype=np.uint8)
    return arr > threshold


def high_frequency_mask(image_size: int, outer_ratio: float) -> np.ndarray:
    if not 0.0 < outer_ratio < 1.0:
        raise ValueError("--high-frequency-outer-ratio must be in (0, 1).")

    coords = np.arange(image_size, dtype=np.float64) - (image_size // 2)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    max_radius = radius.max()
    cutoff = max_radius * (1.0 - outer_ratio)
    return radius >= cutoff


def global_high_frequency_ratio(gray: np.ndarray, high_mask: np.ndarray) -> tuple[float, float, float]:
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spectrum) ** 2
    full_power = float(power.sum())
    high_power = float(power[high_mask].sum())
    global_hfr = high_power / full_power if full_power > 0 else 0.0
    return global_hfr, full_power, high_power


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    high_mask = high_frequency_mask(args.image_size, args.high_frequency_outer_ratio)

    for modal in MODALS:
        for split in SPLITS:
            image_dir = args.dataset_root / modal / split / "images"
            mask_dir = args.dataset_root / modal / split / "masks"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing image directory: {image_dir}")
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

            image_paths = sorted(image_dir.glob("*.jpg"))
            for image_path in image_paths:
                mask_path = mask_dir / f"{image_path.stem}.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")

                gray, width, height = read_gray_image(image_path, args.image_size)
                mask = read_mask(mask_path, args.image_size, args.mask_threshold)
                scale = float(mask.mean())
                global_hfr, full_power, high_power = global_high_frequency_ratio(
                    gray, high_mask
                )

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
                    }
                )

    return rows


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
        )
        .sort_values(group_cols)
    )
    return summary


def plot_scale_vs_frequency(df: pd.DataFrame, figure_path: Path) -> None:
    colors = {"WL": "#d62728", "NBI": "#1f77b4"}
    plt.figure(figsize=(8, 6), dpi=160)
    ax = plt.gca()

    for modal in MODALS:
        modal_df = df[df["modal"] == modal]
        ax.scatter(
            modal_df["scale"],
            modal_df["global_hfr"],
            s=10,
            alpha=0.35,
            c=colors[modal],
            label=modal,
            edgecolors="none",
        )
        ax.scatter(
            [modal_df["scale"].mean()],
            [modal_df["global_hfr"].mean()],
            s=120,
            c=colors[modal],
            marker="X",
            edgecolors="black",
            linewidths=0.8,
            label=f"{modal} mean",
        )

    ax.set_title("Scale vs Global Frequency Distribution")
    ax.set_xlabel("Lesion scale (foreground pixels / total pixels)")
    ax.set_ylabel("Global high-frequency ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_scale_vs_frequency_by_split(df: pd.DataFrame, figure_path: Path) -> None:
    colors = {"WL": "#d62728", "NBI": "#1f77b4"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160, sharex=True, sharey=True)

    for ax, split in zip(axes, SPLITS):
        split_df = df[df["split"] == split]
        for modal in MODALS:
            modal_df = split_df[split_df["modal"] == modal]
            ax.scatter(
                modal_df["scale"],
                modal_df["global_hfr"],
                s=10,
                alpha=0.4,
                c=colors[modal],
                label=modal,
                edgecolors="none",
            )
        ax.set_title(split)
        ax.set_xlabel("Lesion scale")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Global high-frequency ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Scale vs Global Frequency by Split", y=1.02)
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()


def validate_outputs(df: pd.DataFrame) -> None:
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

    if not df["scale"].between(0.0, 1.0).all():
        raise AssertionError("scale contains values outside [0, 1].")
    if not df["global_hfr"].between(0.0, 1.0).all():
        raise AssertionError("global_hfr contains values outside [0, 1].")
    if not (df["full_power"] > 0).all():
        raise AssertionError("full_power must be positive for all images.")
    if not (df["high_power"] >= 0).all():
        raise AssertionError("high_power must be non-negative for all images.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args)
    df = pd.DataFrame(rows)
    validate_outputs(df)

    metrics_path = args.output_dir / "frequency_metrics.csv"
    summary_modal_path = args.output_dir / "frequency_summary_by_modal.csv"
    summary_split_path = args.output_dir / "frequency_summary_by_modal_split.csv"
    figure_path = args.figure_dir / "scale_vs_global_frequency.png"
    figure_split_path = args.figure_dir / "scale_vs_global_frequency_by_split.png"

    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summarize(df, ["modal"]).to_csv(summary_modal_path, index=False, encoding="utf-8-sig")
    summarize(df, ["modal", "split"]).to_csv(
        summary_split_path, index=False, encoding="utf-8-sig"
    )
    plot_scale_vs_frequency(df, figure_path)
    plot_scale_vs_frequency_by_split(df, figure_split_path)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_modal_path}")
    print(f"Saved: {summary_split_path}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {figure_split_path}")
    print("Experiment 01 completed successfully.")


if __name__ == "__main__":
    main()
