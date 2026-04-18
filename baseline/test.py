"""
胃部医学图像分割 baseline 的测试入口。

当前状态：
- 工程骨架已经搭建完成
- 下一步将继续补充测试、推理和评估流程
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测试 WL 或 NBI 模态的分割 baseline。"
    )
    parser.add_argument(
        "--modality",
        choices=["WL", "NBI"],
        required=True,
        help="选择需要测试的模态。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("sorted_075"),
        help="数据集根目录。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="训练完成后的模型权重路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="用于保存评估指标和预测结果的目录。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("测试脚手架已经准备好。")
    print(f"当前模态：{args.modality}")
    print(f"数据集根目录：{args.data_root}")
    print(f"模型权重：{args.checkpoint}")
    print(f"输出目录：{args.output_dir}")
    print("下一步将实现模型加载、推理和指标计算。")


if __name__ == "__main__":
    main()
