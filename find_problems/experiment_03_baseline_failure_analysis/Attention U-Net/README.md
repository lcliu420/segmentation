# Attention U-Net：实验三 Baseline 逐图失败分析

本目录包含官方 Attention Gated Networks 源码参考与实验三专用的轻量 PyTorch Attention U-Net baseline。实际训练和测试请使用本目录根部的 `train.py` 与 `test.py`，它们已经对齐实验三的统一输出规范。

重要约束：`../../dataset/` 只读。不要在 `dataset/` 内新建、删除、重命名、移动或覆盖任何文件。所有训练日志、checkpoint、预测图和指标 CSV 都应写入 `experiment_03_baseline_failure_analysis/` 内。

## 源码参考说明

本目录保留了从官方仓库下载的源码：

```text
source_code/
```

其中 `source_code/models/networks/unet_2D.py` 提供 2D U-Net 编码解码结构，`source_code/models/layers/grid_attention_layer.py` 提供 additive attention gate 思路。本实验的 `models/attention_unet.py` 参考这些实现，改造成适合当前 `jpg/png` 二分类逐图失败分析流程的轻量 2D Attention U-Net。

## 数据路径

建议在 `experiment_03_baseline_failure_analysis/Attention U-Net` 目录内运行命令。此时数据集路径是：

```text
../../dataset
```

也可以不传 `--data_root`，代码默认会自动指向 `find_problems/dataset`。

## 训练前检查

训练前建议先确认服务器数据划分为正式实验三使用的 `1960/420/420`：

```bash
find ../../dataset/WL/train/images -type f | wc -l
find ../../dataset/WL/val/images -type f | wc -l
find ../../dataset/WL/test/images -type f | wc -l
find ../../dataset/NBI/train/images -type f | wc -l
find ../../dataset/NBI/val/images -type f | wc -l
find ../../dataset/NBI/test/images -type f | wc -l
```

也可以先检查 mask 是否被正常读到前景：

```bash
python - <<'PY'
from datasets.gastric_dataset import GastricSegmentationDataset

ds = GastricSegmentationDataset("../../dataset", "WL", "val", img_size=224, limit_samples=5)
for sample in ds:
    label = sample["label"]
    print(sample["case_name"], int(label.sum()), float(label.float().mean()))
PY
```

输出里的 `label.sum()` 不应该全是 `0`。训练脚本也会在开始时自动打印 train/val 前若干张 mask 的 `fg_ratio` 摘要；如果检查到全背景，会直接报错停止。

## 训练阶段

训练阶段不建议每个 epoch 计算 `Boundary IoU / HD95 / MAE`，尤其是 `HD95`。默认只记录：

```text
train: loss
val: loss, Dice, IoU
checkpoint: best val Dice
```

WL 示例：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name attention_unet_WL --model_name attention_unet
```

NBI 示例：

```bash
python train.py --modal NBI --data_root ../../dataset --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224 --num_workers 8 --run_name attention_unet_NBI --model_name attention_unet
```

训练输出：

```text
../outputs/runs/attention_unet_WL/
  best_model.pth
  last_model.pth
  history.csv
  log.txt
```

## 最终测试阶段

对 best checkpoint 在 test 集逐图计算：

```text
Dice
IoU
Boundary_IoU
HD95
MAE
```

WL 示例：

```bash
python test.py --modal WL --data_root ../../dataset --output_dir ../outputs/runs/attention_unet_WL --checkpoint ../outputs/runs/attention_unet_WL/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name attention_unet_WL --model_name attention_unet
```

NBI 示例：

```bash
python test.py --modal NBI --data_root ../../dataset --output_dir ../outputs/runs/attention_unet_NBI --checkpoint ../outputs/runs/attention_unet_NBI/best_model.pth --img_size 224 --batch_size 1 --num_workers 0 --run_name attention_unet_NBI --model_name attention_unet
```

测试输出：

```text
../predictions/attention_unet/WL/test/*.png
../predictions/attention_unet/NBI/test/*.png
../outputs/prediction_metrics/attention_unet_WL_prediction_metrics.csv
../outputs/prediction_metrics/attention_unet_NBI_prediction_metrics.csv
```

CSV 字段：

```text
image, modal, split, model, Dice, IoU, Boundary_IoU, HD95, MAE, pred_fg_ratio, gt_fg_ratio
```

## 快速流程检查

完整训练请放到 GPU 服务器。若只想检查代码流程，可以限制样本数：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 1 --batch_size 2 --num_workers 0 --limit_samples 2 --run_name smoke_attention_unet_WL --model_name attention_unet_smoke
python test.py --modal WL --data_root ../../dataset --output_dir ../outputs/runs/smoke_attention_unet_WL --checkpoint ../outputs/runs/smoke_attention_unet_WL/best_model.pth --num_workers 0 --limit_samples 2 --run_name smoke_attention_unet_WL --model_name attention_unet_smoke
```

如果第 1 个 epoch 开始 `val_dice` 就恒等于 `1.0`，优先检查 mask 是否被读成全背景，以及 `--data_root` 是否写错。
