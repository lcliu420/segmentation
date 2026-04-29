from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs"
METRICS = OUT / "dataset_metrics.csv"
MANIFEST = OUT / "split_manifest.csv"
MODALITIES = ("NBI", "WL")
SEED = 20260429
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def allocate_count(n: int) -> dict[str, int]:
    train = int(round(n * RATIOS["train"]))
    val = int(round(n * RATIOS["val"]))
    test = n - train - val
    return {"train": train, "val": val, "test": test}


def load_rows() -> list[dict[str, str]]:
    if not METRICS.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS}")
    with METRICS.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"modality", "stem", "image_path", "mask_path", "foreground_ratio", "size_bucket"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in {METRICS}: {sorted(missing)}")
    return rows


def choose_splits(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(SEED)
    split_rows: list[dict[str, str]] = []

    for modality in MODALITIES:
        modality_rows = [r for r in rows if r["modality"] == modality]
        if len(modality_rows) != 2800:
            raise ValueError(f"{modality} expected 2800 rows, got {len(modality_rows)}")

        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in modality_rows:
            groups[row["size_bucket"]].append(row)

        assigned: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
        for bucket_rows in groups.values():
            bucket_rows = bucket_rows[:]
            rng.shuffle(bucket_rows)
            counts = allocate_count(len(bucket_rows))
            train_end = counts["train"]
            val_end = train_end + counts["val"]
            assigned["train"].extend(bucket_rows[:train_end])
            assigned["val"].extend(bucket_rows[train_end:val_end])
            assigned["test"].extend(bucket_rows[val_end:])

        target = allocate_count(len(modality_rows))
        for split in ("train", "val", "test"):
            delta = len(assigned[split]) - target[split]
            if delta > 0:
                for other in ("train", "val", "test"):
                    if other == split:
                        continue
                    need = target[other] - len(assigned[other])
                    if need <= 0:
                        continue
                    move_n = min(delta, need)
                    assigned[other].extend(assigned[split][-move_n:])
                    del assigned[split][-move_n:]
                    delta -= move_n
                    if delta == 0:
                        break

        for split, selected in assigned.items():
            for row in selected:
                item = dict(row)
                item["split"] = split
                item["new_image_path"] = f"{modality}/{split}/images/{Path(row['image_path']).name}"
                item["new_mask_path"] = f"{modality}/{split}/masks/{Path(row['mask_path']).name}"
                split_rows.append(item)

    return split_rows


def ensure_clean_target() -> None:
    for modality in MODALITIES:
        for split in ("train", "val", "test"):
            for kind in ("images", "masks"):
                target = ROOT / modality / split / kind
                if target.exists():
                    for path in target.iterdir():
                        if path.is_file():
                            break


def copy_files(split_rows: list[dict[str, str]]) -> None:
    ensure_clean_target()
    for modality in MODALITIES:
        for split in ("train", "val", "test"):
            for kind in ("images", "masks"):
                (ROOT / modality / split / kind).mkdir(parents=True, exist_ok=True)

    for row in split_rows:
        copy_one(ROOT / row["image_path"], ROOT / row["new_image_path"])
        copy_one(ROOT / row["mask_path"], ROOT / row["new_mask_path"])


def copy_one(src: Path, dst: Path) -> None:
    if dst.exists():
        if src.exists() and src.stat().st_size == dst.stat().st_size:
            return
        if not src.exists():
            return
        raise RuntimeError(f"Both source and target exist but differ: {src} -> {dst}")

    if not src.exists():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def save_manifest(split_rows: list[dict[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "modality",
        "split",
        "stem",
        "image_path",
        "mask_path",
        "new_image_path",
        "new_mask_path",
        "foreground_ratio",
        "size_bucket",
    ]
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(split_rows, key=lambda r: (r["modality"], r["split"], r["stem"])))


def main() -> None:
    rows = load_rows()
    split_rows = choose_splits(rows)
    save_manifest(split_rows)
    copy_files(split_rows)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in split_rows:
        counts[(row["modality"], row["split"])] += 1
    for modality in MODALITIES:
        print(modality)
        for split in ("train", "val", "test"):
            print(f"  {split}: {counts[(modality, split)]}")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
