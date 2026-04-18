from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Union

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
