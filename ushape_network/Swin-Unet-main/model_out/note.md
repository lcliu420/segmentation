# Swin-Unet 训练记录

说明：本次 `Swin-Unet` 实验的 `history.csv` 只记录了 `train_loss / val_loss / DSC / mIoU`，没有记录 `Precision` 和 `Recall`，所以这里重点分析验证集 `DSC` 与 `mIoU` 的变化。

## 1. WL only - Swin-Unet - 150 epoch

```bash
python train.py --dataset WL --root_path datasets/MyDataset/WL --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir model_out/wl_swinunet_e150_bs8_224 --img_size 224 --batch_size 8 --max_epochs 150 --base_lr 0.01
```

### 最优验证结果
- Best Epoch：`98`
- Best Val Dice：`0.8290`
- Best Val IoU：`0.7080`
- Best Val Loss：`0.1642`（第 `40` 轮）

### 训练过程简要观察
- 前期收敛比较快，验证集 `Dice` 在 `1-10` 轮均值约为 `0.7329`，到了 `11-30` 轮已经提升到约 `0.7971`，说明模型前期就能较快学到有效分割模式。
- 中期继续稳定提升，`31-60` 轮验证集 `Dice` 均值约 `0.8145`，`61-100` 轮进一步提升到约 `0.8203`，但增幅已经明显放缓。
- 最优结果出现在第 `98` 轮，`Best Val Dice=0.8290`，说明对 `WL only` 来说，训练到 `100 epoch` 左右基本已经接近最佳状态。
- 后期 `101-150` 轮验证集 `Dice` 均值约 `0.8234`，只比 `61-100` 轮略高，最后一轮 `val_dice=0.8248`，与最佳值仅差约 `0.0043`，整体已经进入平台区，没有明显严重过拟合。
- `val_loss` 的最低点出现在第 `40` 轮，而 `Dice/mIoU` 的最好结果出现在更后面，说明这个任务上单看 `val_loss` 不能完全代表最终分割质量，还是应优先关注 `Dice` 和 `mIoU`。

## 2. NBI only - Swin-Unet - 150 epoch

```bash
python train.py --dataset NBI --root_path datasets/MyDataset/NBI --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir model_out/nbi_swinunet_e150_bs8_224 --img_size 224 --batch_size 8 --max_epochs 150 --base_lr 0.01
```

### 最优验证结果
- Best Epoch：`91`
- Best Val Dice：`0.8623`
- Best Val IoU：`0.7579`
- Best Val Loss：`0.1813`（第 `22` 轮）

### 训练过程简要观察
- 前期起步就很快，验证集 `Dice` 在 `1-10` 轮均值约 `0.7913`，明显高于 `WL`，说明 `NBI` 模态对病灶分割更友好，模型更容易学到稳定特征。
- 中期持续稳步提升，`11-30` 轮验证集 `Dice` 均值约 `0.8392`，`31-60` 轮约 `0.8477`，到 `61-100` 轮进一步来到约 `0.8534`，整体训练过程比较平稳。
- 最优结果出现在第 `91` 轮，`Best Val Dice=0.8623`、`Best Val IoU=0.7579`，说明 `NBI only` 在 `100 epoch` 左右已经基本收敛，后续收益有限。
- 后期 `101-150` 轮验证集 `Dice` 均值约 `0.8564`，最后一轮 `val_dice=0.8582`，与最佳值仅差约 `0.0041`，说明后期虽然有轻微波动，但总体仍稳定处于高位平台区，没有明显严重过拟合。
- 和 `WL` 相比，`NBI` 在整个训练阶段都表现更强：不仅前期收敛更快，而且最终最优 `Dice` 更高，说明在当前数据和设置下，`NBI` 模态的可分性和鲁棒性都更好。

## 3. 两组实验对比结论

- `NBI` 的最优验证结果明显优于 `WL`：`Dice 0.8623 vs 0.8290`，`mIoU 0.7579 vs 0.7080`，说明 `NBI` 模态对当前分割任务更有优势。
- 两组实验都在 `90-100` 轮附近达到最佳或接近最佳结果，说明 `150 epoch` 能保证充分训练，但真正有效提升主要集中在前 `100 epoch` 左右。
- 两组实验最后一轮与最佳轮的 `Dice` 差距都只有约 `0.004`，整体训练过程比较稳定，没有明显严重过拟合。
- 如果后续继续做 baseline，对 `Swin-Unet` 来说可以优先考虑保留 `150 epoch` 设置用于完整训练记录；如果想节省时间，也可以尝试把训练轮数收缩到 `100 epoch` 左右做快速实验。
