from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch


def ensure_dir(path: Union[str, Path]) -> Path:
    """创建目录；如果目录已存在则直接复用。"""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """将字典保存为格式化 JSON 文件。"""

    output_path = Path(path)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_history_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """将每个 epoch 的训练日志保存为 CSV。"""

    if not rows:
        return

    output_path = Path(path)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    """固定随机种子，尽量提高实验可复现性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_experiment_name(
    modality: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    image_size: Sequence[int],
    scheduler: str,
    use_amp: bool = False,
) -> str:
    """
    根据关键训练参数自动生成实验名称。

    命名风格示例：
    - `wl_unet_e100_bs8_512x512_cosine_amp_20260420_213000`
    - `nbi_unet_e50_bs4_640x640_none_20260420_221500`
    """

    if len(image_size) != 2:
        raise ValueError("image_size 必须包含两个整数。")

    height, width = int(image_size[0]), int(image_size[1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name_parts = [
        modality.lower(),
        model_name.lower(),
        f"e{epochs}",
        f"bs{batch_size}",
        f"{height}x{width}",
        scheduler.lower(),
    ]
    if use_amp:
        name_parts.append("amp")
    name_parts.append(timestamp)
    return "_".join(name_parts)


def infer_experiment_name_from_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[str]:
    """
    尝试从 checkpoint 路径中推断实验目录名。

    如果路径形如：
    - `outputs/<experiment_name>/checkpoints/best.pt`

    则返回 `<experiment_name>`。
    """

    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name != "checkpoints":
        return None
    if checkpoint.parent.parent == checkpoint.parent:
        return None
    return checkpoint.parent.parent.name
