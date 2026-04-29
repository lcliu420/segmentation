from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("WL", "NBI")
OUT = ROOT / "analysis_outputs"
VIS = OUT / "visual_check"
SEED = 20260429
SAMPLE_N = 30


def binary_dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(np.var(lap))


def resize_for_metrics(rgb: Image.Image, mask: Image.Image, max_side: int = 640) -> tuple[Image.Image, Image.Image]:
    w, h = rgb.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return rgb, mask
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return rgb.resize(size, Image.Resampling.BILINEAR), mask.resize(size, Image.Resampling.NEAREST)


def size_bucket(foreground_ratio: float) -> str:
    if foreground_ratio < 0.01:
        return "small_<1%"
    if foreground_ratio < 0.10:
        return "medium_1-10%"
    return "large_>=10%"


def risk_types(row: dict[str, float | str]) -> list[str]:
    risks = []
    if float(row["foreground_ratio"]) < 0.01:
        risks.append("小目标漏检型")
    if float(row["lesion_bg_rgb_distance"]) < 55 or float(row["boundary_gray_diff"]) < 3:
        risks.append("低对比度/边界模糊型")
    if float(row["specular_ratio"]) > 0.02:
        risks.append("反光干扰型")
    if float(row["laplacian_var"]) < 35:
        risks.append("模糊成像型")
    if float(row["boundary_complexity"]) > 15.0 or float(row["elongation"]) > 4.0:
        risks.append("形状不规则型")
    return risks


def compute_metrics(modality: str, image_path: Path) -> dict[str, float | str]:
    mask_path = ROOT / modality / "masks" / f"{image_path.stem}.png"
    with Image.open(image_path) as image_raw, Image.open(mask_path) as mask_raw:
        image = image_raw.convert("RGB")
        mask_img = mask_raw.convert("L")
        width, height = image.size
        image_m, mask_m = resize_for_metrics(image, mask_img)

    rgb = np.asarray(image_m).astype(np.float32)
    gray = np.asarray(image_m.convert("L")).astype(np.float32)
    mask = np.asarray(mask_m) > 0

    total = mask.size
    area = int(mask.sum())
    fg_ratio = area / total if total else 0.0
    brightness = float(gray.mean())
    contrast = float(gray.std())
    specular = np.logical_and(rgb.max(axis=2) > 245, rgb.mean(axis=2) > 220)
    specular_ratio = float(specular.mean())
    blur = laplacian_variance(gray)

    if area > 0 and area < total:
        lesion_mean = rgb[mask].mean(axis=0)
        bg_mean = rgb[~mask].mean(axis=0)
        color_dist = float(np.linalg.norm(lesion_mean - bg_mean))
    else:
        color_dist = 0.0

    dilated = binary_dilate(mask)
    eroded = binary_erode(mask)
    inner_edge = mask & ~eroded
    outer_edge = dilated & ~mask
    if inner_edge.any() and outer_edge.any():
        boundary_diff = float(abs(gray[inner_edge].mean() - gray[outer_edge].mean()))
    else:
        boundary_diff = 0.0

    boundary = dilated ^ eroded
    perimeter = int(boundary.sum())
    boundary_complexity = float(perimeter / math.sqrt(area)) if area else 0.0

    if area:
        ys, xs = np.where(mask)
        box_w = int(xs.max() - xs.min() + 1)
        box_h = int(ys.max() - ys.min() + 1)
        elongation = float(max(box_w, box_h) / max(1, min(box_w, box_h)))
    else:
        box_w = 0
        box_h = 0
        elongation = 0.0

    return {
        "modality": modality,
        "stem": image_path.stem,
        "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
        "mask_path": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
        "width": width,
        "height": height,
        "foreground_ratio": fg_ratio,
        "size_bucket": size_bucket(fg_ratio),
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "lesion_bg_rgb_distance": color_dist,
        "boundary_gray_diff": boundary_diff,
        "specular_ratio": specular_ratio,
        "laplacian_var": blur,
        "boundary_complexity": boundary_complexity,
        "elongation": elongation,
        "bbox_width": box_w,
        "bbox_height": box_h,
    }


def percentile_ranks(values: list[float], reverse: bool = False) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    ranks = [0.0] * len(values)
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / denom
    return ranks


def add_scores(rows: list[dict[str, float | str]]) -> None:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["modality"])].append(row)

    for modality_rows in grouped.values():
        low_boundary = percentile_ranks([float(r["boundary_gray_diff"]) for r in modality_rows])
        low_color = percentile_ranks([float(r["lesion_bg_rgb_distance"]) for r in modality_rows])
        high_specular = percentile_ranks([float(r["specular_ratio"]) for r in modality_rows], reverse=True)
        low_blur = percentile_ranks([float(r["laplacian_var"]) for r in modality_rows])
        high_complexity = percentile_ranks([float(r["boundary_complexity"]) for r in modality_rows], reverse=True)
        small_area = percentile_ranks([float(r["foreground_ratio"]) for r in modality_rows])

        for i, row in enumerate(modality_rows):
            score = (
                0.30 * low_boundary[i]
                + 0.22 * low_color[i]
                + 0.15 * high_specular[i]
                + 0.13 * low_blur[i]
                + 0.12 * high_complexity[i]
                + 0.08 * small_area[i]
            )
            row["risk_score"] = score
            row["risk_types"] = ";".join(risk_types(row)) or "未标记"


def save_csv(rows: list[dict[str, float | str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "modality",
        "stem",
        "image_path",
        "mask_path",
        "width",
        "height",
        "foreground_ratio",
        "size_bucket",
        "brightness_mean",
        "contrast_std",
        "lesion_bg_rgb_distance",
        "boundary_gray_diff",
        "specular_ratio",
        "laplacian_var",
        "boundary_complexity",
        "elongation",
        "bbox_width",
        "bbox_height",
        "risk_score",
        "risk_types",
    ]
    with (OUT / "dataset_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> str:
    lines = ["# 数据集诊断汇总", ""]
    for modality in MODALITIES:
        rs = [r for r in rows if r["modality"] == modality]
        lines.extend([f"## {modality}", ""])
        lines.append(f"- 样本数：{len(rs)}")
        for key, label in [
            ("foreground_ratio", "mask 前景占比"),
            ("brightness_mean", "亮度"),
            ("contrast_std", "对比度"),
            ("lesion_bg_rgb_distance", "病灶-背景 RGB 距离"),
            ("boundary_gray_diff", "局部边界灰度差"),
            ("specular_ratio", "反光占比"),
            ("laplacian_var", "Laplacian 清晰度"),
            ("boundary_complexity", "边界复杂度"),
        ]:
            vals = np.array([float(r[key]) for r in rs], dtype=np.float64)
            lines.append(
                f"- {label}：mean={vals.mean():.6f}, median={np.median(vals):.6f}, "
                f"min={vals.min():.6f}, max={vals.max():.6f}"
            )
        bucket_counts = Counter(str(r["size_bucket"]) for r in rs)
        risk_counts = Counter()
        for r in rs:
            for item in str(r["risk_types"]).split(";"):
                risk_counts[item] += 1
        lines.append(f"- 病灶面积分布：{dict(bucket_counts)}")
        lines.append(f"- 风险类型计数：{dict(risk_counts)}")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def overlay_tile(row: dict[str, float | str], label: str, tile_size: tuple[int, int] = (220, 190)) -> Image.Image:
    image_path = ROOT / str(row["image_path"])
    mask_path = ROOT / str(row["mask_path"])
    with Image.open(image_path) as image_raw, Image.open(mask_path) as mask_raw:
        image = image_raw.convert("RGB")
        mask = mask_raw.convert("L")

    canvas_w, canvas_h = tile_size
    header_h = 34
    max_img_h = canvas_h - header_h
    scale = min(canvas_w / image.width, max_img_h / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(new_size, Image.Resampling.BILINEAR)
    mask = mask.resize(new_size, Image.Resampling.NEAREST)

    arr = np.asarray(image).astype(np.float32)
    m = np.asarray(mask) > 0
    overlay = arr.copy()
    overlay[m] = overlay[m] * 0.55 + np.array([255, 0, 0], dtype=np.float32) * 0.45
    edge = binary_dilate(m) ^ binary_erode(m)
    overlay[edge] = np.array([255, 230, 0], dtype=np.float32)
    tile = Image.new("RGB", tile_size, (25, 25, 25))
    tile.paste(Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)), ((canvas_w - new_size[0]) // 2, header_h))

    draw = ImageDraw.Draw(tile)
    text = (
        f"{label} fg={float(row['foreground_ratio']) * 100:.1f}% "
        f"bd={float(row['boundary_gray_diff']):.1f}"
    )
    draw.rectangle((0, 0, canvas_w, header_h), fill=(0, 0, 0))
    draw.text((6, 6), text[:34], fill=(255, 255, 255), font=ImageFont.load_default())
    return tile


def make_montage(rows: list[dict[str, float | str]], output_path: Path, title: str) -> None:
    cols = 5
    tile_size = (220, 190)
    title_h = 36
    rows_count = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * tile_size[0], rows_count * tile_size[1] + title_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill=(255, 255, 255), font=ImageFont.load_default())
    for idx, row in enumerate(rows):
        tile = overlay_tile(row, str(idx + 1))
        x = (idx % cols) * tile_size[0]
        y = title_h + (idx // cols) * tile_size[1]
        canvas.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def save_visual_samples(rows: list[dict[str, float | str]]) -> None:
    rng = random.Random(SEED)
    sample_rows = []
    for modality in MODALITIES:
        rs = [r for r in rows if r["modality"] == modality]
        rs_by_risk = sorted(rs, key=lambda r: float(r["risk_score"]))
        easy_pool = rs_by_risk[: max(120, SAMPLE_N)]
        difficult_pool = rs_by_risk[len(rs_by_risk) // 2 - 80 : len(rs_by_risk) // 2 + 80]
        failure_pool = rs_by_risk[-max(120, SAMPLE_N) :]

        groups = {
            "easy": rng.sample(easy_pool, SAMPLE_N),
            "difficult": rng.sample(difficult_pool, SAMPLE_N),
            "likely_failure": rng.sample(failure_pool, SAMPLE_N),
        }
        for group, selected in groups.items():
            selected = sorted(selected, key=lambda r: float(r["risk_score"]))
            make_montage(
                selected,
                VIS / f"{modality}_{group}_30.jpg",
                f"{modality} {group} samples: red=mask, yellow=boundary",
            )
            for r in selected:
                sample_rows.append(
                    {
                        "modality": modality,
                        "group": group,
                        "stem": r["stem"],
                        "image_path": r["image_path"],
                        "mask_path": r["mask_path"],
                        "risk_score": r["risk_score"],
                        "risk_types": r["risk_types"],
                    }
                )

    with (OUT / "visual_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["modality", "group", "stem", "image_path", "mask_path", "risk_score", "risk_types"],
        )
        writer.writeheader()
        writer.writerows(sample_rows)


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for modality in MODALITIES:
        image_dir = ROOT / modality / "images"
        for image_path in sorted(image_dir.glob("*.jpg")):
            rows.append(compute_metrics(modality, image_path))
    add_scores(rows)
    save_csv(rows)
    summarize(rows)
    save_visual_samples(rows)
    print(f"Wrote {OUT / 'dataset_metrics.csv'}")
    print(f"Wrote {OUT / 'summary.md'}")
    print(f"Wrote {OUT / 'visual_samples.csv'}")
    print(f"Wrote visual checks to {VIS}")


if __name__ == "__main__":
    main()
