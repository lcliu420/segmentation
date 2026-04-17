from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ("NBI", "WL")


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise RuntimeError(f"Refusing to reset directory outside project root: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)


def copy_split(files: list[Path], destination: Path) -> None:
    ensure_dir(destination)
    for src in files:
        shutil.copy2(src, destination / src.name)


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare train/val split and pending output folders."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Reset prepared_data before generating a new split.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    origin_data = root / "origin_data"
    prepared_data = root / "prepared_data"
    sorted_output = root / "sorted_output"

    if not origin_data.exists():
        raise FileNotFoundError(f"origin_data folder not found: {origin_data}")

    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, str]] = []

    for output_class in ("NBI", "WL", "uncertain"):
        ensure_dir(sorted_output / output_class)

    if args.clean:
        reset_dir(prepared_data, root)

    for class_name in CLASSES:
        class_dir = origin_data / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        images = collect_images(class_dir)
        if not images:
            raise RuntimeError(f"No images found in: {class_dir}")

        rng.shuffle(images)
        val_count = max(1, round(len(images) * args.val_ratio))
        train_files = images[val_count:]
        val_files = images[:val_count]

        copy_split(train_files, prepared_data / "train" / class_name)
        copy_split(val_files, prepared_data / "val" / class_name)

        manifest_rows.extend(
            {"filename": path.name, "class": class_name, "split": "train"}
            for path in train_files
        )
        manifest_rows.extend(
            {"filename": path.name, "class": class_name, "split": "val"}
            for path in val_files
        )

    write_manifest(manifest_rows, prepared_data / "split_manifest.csv")

    print(f"Prepared data under: {prepared_data}")
    print(f"Sorted output folders under: {sorted_output}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()
