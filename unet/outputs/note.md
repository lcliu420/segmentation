# Unet 训练记录

## 1. WL only - U-Net - 100 epoch

python train.py --modality WL --device cuda --amp --epochs 100 --batch-size 8 --num-workers 8 --image-size 512 512 --scheduler cosine

### 最优验证结果
- Best Epoch：`94`
- Best Val Dice：`0.7095`
- Best Val IoU：`0.5841`
- Best Val Precision：`0.7388`
- Best Val Recall：`0.7728`

### 训练过程简要观察
- 前期模型收敛较快，验证集 `Dice` 从较低水平逐步提升。
- 中后期继续稳定优化，没有出现明显严重过拟合。
- 最优结果出现在第 `94` 轮，说明相较于 `50 epoch`，拉长到 `100 epoch` 是有效的。
- 在 `90-100` 轮附近验证指标进入平台区，后续若继续加 `epoch`，预计提升空间有限。

## 2. NBI only - U-Net - 100 epoch

python train.py --modality NBI --device cuda --amp --epochs 100 --batch-size 8 --num-workers 8 --image-size 512 512 --scheduler cosine

### 最优验证结果
- Best Epoch：`91`
- Best Val Dice：`0.7931`
- Best Val IoU：`0.6785`
- Best Val Precision：`0.7878`
- Best Val Recall：`0.8551`

### 训练过程简要观察
- 前期模型起步较快，验证集 `Dice` 在较早阶段就达到较高水平，说明 `NBI` 模态对病灶分割更友好。
- 中后期继续稳定提升，验证集表现持续增强，没有出现明显严重过拟合。
- 最优结果出现在第 `91` 轮，说明 `100 epoch` 对这组 `NBI only` baseline 是充分且有效的。
- 在 `80-100` 轮附近验证指标进入高位平台区，后续即使继续增加 `epoch`，预计提升空间也不会太大。
## 3. WL only - U-Net - 150 epoch

python train.py --modality WL --device cuda --amp --epochs 150 --batch-size 8 --num-workers 8 --image-size 512 512 --scheduler cosine

### 最优验证结果
- Best Epoch：`131`
- Best Val Dice：`0.7155`
- Best Val IoU：`0.5911`
- Best Val Precision：`0.7412`
- Best Val Recall：`0.7827`

### 训练过程简要观察
- 前期提升仍然比较明显，验证集 `Dice` 从 `1-10` 轮均值约 `0.3780`，逐步提升到 `31-60` 轮均值约 `0.6425`，训练过程稳定。
- `61-100` 轮后验证集继续缓慢上升，`101-150` 轮的验证集 `Dice` 均值约 `0.7080`，说明延长训练对 `WL` 还有一定帮助。
- 最优结果出现在第 `131` 轮，较 `100 epoch` 版本有小幅提升，说明 `WL only` 在 `100 epoch` 后还没有完全到顶。
- 后期 `131-150` 轮已经进入平台区，最后一轮 `val_dice=0.7138`，与最佳结果差距很小，没有出现明显严重过拟合。

## 4. NBI only - U-Net - 150 epoch

python train.py --modality NBI --device cuda --amp --epochs 150 --batch-size 8 --num-workers 8 --image-size 512 512 --scheduler cosine

### 最优验证结果
- Best Epoch：`119`
- Best Val Dice：`0.7939`
- Best Val IoU：`0.6799`
- Best Val Precision：`0.7776`
- Best Val Recall：`0.8696`

### 训练过程简要观察
- 前期起步依旧很快，验证集 `Dice` 在 `1-10` 轮均值就达到约 `0.6031`，明显高于 `WL`，说明 `NBI` 模态的可分性更强。
- 中期到后期持续稳步提升，`61-100` 轮验证集 `Dice` 均值约 `0.7756`，`101-150` 轮进一步提升到约 `0.7883`，训练过程整体很平稳。
- 最优结果出现在第 `119` 轮，较 `100 epoch` 版本只带来极小幅提升，说明 `NBI only` 在 `100 epoch` 左右已经基本收敛。
- 后期 `119-150` 轮基本处于高位平台区，最后一轮 `val_dice=0.7877`，与最佳值接近，没有明显严重过拟合，但继续加 `epoch` 的收益已经很有限。
