# UNetSDI 实验记录

训练阶段只看轻指标：`loss / Dice / IoU`。最终结果以 `Test.py` 对 `best.pth` 计算的完整指标为准：`Dice / IoU / Boundary IoU / HD95 / MAE`。

## 当前模型边界

当前 `UNetSDI` 实验可以概括为：普通 U-Net encoder + 普通 U-Net decoder + SDI skip connection。

未额外加入：

- PVT / Transformer backbone
- attention 模块
- boundary 分支
- boundary loss
- deep supervision
- 多尺度训练，默认 single-scale
- wandb
- 训练阶段重指标；训练阶段只看 `loss / Dice / IoU`

## 训练总览

| Run | Modal | 参数摘要 | Best epoch | Best val Dice | Best val IoU | Last val Dice | Best checkpoint | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 001 | WL | `e100, lr1e-4, AdamW, bs8, 352, aug, single` | 79 | 0.7473 | 0.6256 | 0.7409 | `outputs/UNetSDI/wl_unetsdi_e100_bs8_352_adamw_aug_single/checkpoints/best.pth` | 正常收敛，后期轻微过拟合 |
| 002 | NBI | `e100, lr1e-4, AdamW, bs8, 352, aug, single` | 91 | 0.7951 | 0.6857 | 0.7915 | `outputs/UNetSDI/nbi_unetsdi_e100_bs8_352_adamw_aug_single/checkpoints/best.pth` | 正常收敛，明显优于 WL |

## 训练命令

WL:

```bash
python Train.py --data_root dataset --modal WL --model unet_sdi --epoch 100 --lr 1e-4 --optimizer AdamW --batchsize 8 --trainsize 352 --augmentation True --num_workers 8 --train_save outputs --device cuda
```

NBI:

```bash
python Train.py --data_root dataset --modal NBI --model unet_sdi --epoch 100 --lr 1e-4 --optimizer AdamW --batchsize 8 --trainsize 352 --augmentation True --num_workers 8 --train_save outputs --device cuda
```

## 简要分析

- WL: best 在 epoch 79，最后一轮 val Dice 降到 0.7409，说明 80 epoch 后继续训练收益不大。
- NBI: best 在 epoch 91，val Dice 0.7951；整体比 WL 高约 4.8 Dice 点，符合 NBI 边界和纹理更清晰的特点。
- 后续新增实验时，在总览表追加一行即可，重点记录：参数摘要、best epoch、best val Dice/IoU、checkpoint 和一句备注。

## 待测试

```bash
python Test.py --data_root dataset --modal WL --model unet_sdi --pth_path outputs/UNetSDI/wl_unetsdi_e100_bs8_352_adamw_aug_single/checkpoints/best.pth --save_pred True --device cuda
```

```bash
python Test.py --data_root dataset --modal NBI --model unet_sdi --pth_path outputs/UNetSDI/nbi_unetsdi_e100_bs8_352_adamw_aug_single/checkpoints/best.pth --save_pred True --device cuda
```
