# Swin-Unet：实验三 Baseline 逐图失败分析

本目录中的 Swin-Unet 已改造成 WL/NBI 胃镜病灶二分类分割 baseline。它服务于实验三：训练阶段只记录轻量指标，最终测试阶段保存逐图预测 mask，并输出逐图完整指标，后续用于和实验二的边界频率指标合并分析。

重要约束：`../../dataset/` 只读。不要在 `dataset/` 内新建、删除、重命名、移动或覆盖任何文件。所有训练日志、checkpoint、预测图和指标 CSV 都应写入 `experiment_03_baseline_failure_analysis/` 内。

## 数据路径

建议在 `experiment_03_baseline_failure_analysis/Swin-Unet` 目录内运行命令。此时数据集路径是：

```text
../../dataset
```

也可以不传 `--data_root`，代码默认会自动指向 `find_problems/dataset`。

不要使用 `../../../dataset`。这个路径会跳出 `find_problems`，在服务器上容易解析到错误的数据目录。

## 训练前 mask 检查

如果服务器上的 mask 是 `0/1`，使用 `arr > 127` 会把所有前景读成背景，从而导致 `val_dice=1.0 / val_iou=1.0` 这类假满分。当前代码已改为把任意非零像素作为前景，即同时兼容 `0/1` 和 `0/255`。

训练前建议先跑：

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
python train.py --modal WL --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224
```

NBI 示例：

```bash
python train.py --modal NBI --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150 --batch_size 8 --base_lr 1e-4 --img_size 224
```

训练输出：

```text
../outputs/runs/swin_unet_WL/
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
python test.py --modal WL --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir ../outputs/runs/swin_unet_WL --checkpoint ../outputs/runs/swin_unet_WL/best_model.pth --img_size 224
```

NBI 示例：

```bash
python test.py --modal NBI --data_root ../../dataset --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir ../outputs/runs/swin_unet_NBI --checkpoint ../outputs/runs/swin_unet_NBI/best_model.pth --img_size 224
```

测试输出：

```text
../predictions/swin_unet/WL/test/*.png
../predictions/swin_unet/NBI/test/*.png
../outputs/prediction_metrics/swin_unet_WL_prediction_metrics.csv
../outputs/prediction_metrics/swin_unet_NBI_prediction_metrics.csv
```

CSV 字段：

```text
image, modal, split, model, Dice, IoU, Boundary_IoU, HD95, MAE, pred_fg_ratio, gt_fg_ratio
```

## 快速流程检查

完整训练请放到 GPU 服务器。若只想检查代码流程，可以限制样本数：

```bash
python train.py --modal WL --data_root ../../dataset --max_epochs 1 --batch_size 2 --num_workers 0 --limit_samples 2 --run_name smoke_swin_unet_WL --model_name swin_unet_smoke
python test.py --modal WL --data_root ../../dataset --output_dir ../outputs/runs/smoke_swin_unet_WL --checkpoint ../outputs/runs/smoke_swin_unet_WL/best_model.pth --num_workers 0 --limit_samples 2 --run_name smoke_swin_unet_WL --model_name swin_unet_smoke
```

如果第 1 个 epoch 开始 `val_dice` 就恒等于 `1.0`，优先检查 mask 是否被读成全背景，以及 `--data_root` 是否仍然写错。
