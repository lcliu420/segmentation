# CSWin-UNet + PFESA 模型架构与 new_dataset 结果说明

## 1. 当前模型整体思路

当前模型以 CSWin-UNet 作为分割基底，并没有替换整个网络结构。CSWin-UNet 论文提供的是主干 encoder-decoder 结构，包括 CSWin Transformer Block、skip connection 和 CARAFE 上采样；PFESA 论文提供的是无参数频率注意力模块。当前改动点是把 PFESA 模块插入到 encoder 和 decoder 之间的 skip feature 上，用于增强传给 decoder 的跳跃连接特征。

PFESA 的插入位置是可配置的，支持：

```text
none / x1 / x2 / x3 / x12 / x13 / x23 / x123
```

其中：

- `x1`：Stage 1 输出的浅层 skip feature，空间分辨率最高，主要包含边界和局部细节。
- `x2`：Stage 2 输出的中层 skip feature，包含局部纹理和一定结构信息。
- `x3`：Stage 3 输出的深层 skip feature，语义更强，空间分辨率更低。
- `x123`：同时在 `x1 + x2 + x3` 三条 skip 上加入 PFESA。

需要注意的是，PFESA 只增强送入 decoder concat 的 skip feature，不改变 encoder 主干继续下采样的路径。也就是说，`merge1 / merge2 / merge3` 的输入仍然是原始 stage 输出，PFESA 不会影响主干特征向后传播。

因此，这个模型不能表述为“PFESA 替换了 CSWin-UNet”，更准确的表述是：

```text
CSWin-UNet with PFESA on configurable skip features
```

## 2. 与两篇参考论文的对应关系

CSWin-UNet 论文中的核心结构是当前模型的主体：输入图像先经过 convolutional token embedding，然后进入多尺度 CSWin Transformer encoder；decoder 端也使用 CSWin Transformer Block，并用 CARAFE 进行内容感知上采样；encoder 与 decoder 之间通过 skip connection 融合浅层空间细节和深层语义信息。

PFESA 论文中的核心思想是当前模型的新增模块：利用 FFT 进行频域解耦，将特征拆成高频边界/细节成分和低频结构成分，再分别构造 Edge Attention 与 Structure Attention。论文强调该模块是 parameter-free 的，并且可以用于 skip connection 的自适应特征细化。

当前代码属于对两篇论文的组合迁移：保留 CSWin-UNet 的主网络，只借用 PFESA 的频域注意力思想增强 `x1 / x2 / x3` skip feature。也就是说，图中 PFESA 是后加模块，不是原始 CSWin-UNet 论文自带模块。

## 3. 图中数据流解释

模型输入为 RGB 图像，经过 CSWin-UNet encoder：

```text
Input
-> Conv Token Embedding
-> Stage 1
-> Patch Merging
-> Stage 2
-> Patch Merging
-> Stage 3
-> Patch Merging
-> Stage 4 Bottleneck
```

encoder 中保存三条 skip feature：

```text
x1: Stage 1 输出
x2: Stage 2 输出
x3: Stage 3 输出
```

如果当前 `pfesa_skip_mode` 包含对应层号，则该 skip 会先经过 PFESA，再送入 decoder；如果不包含，则保持原始 skip。

如果图中三条 skip 上都画了 PFESA，应理解为 `pfesa_skip_mode=x123` 的结构示意；如果实际运行的是 `x1 / x2 / x3 / x12 / x13 / x23`，则只有对应的 skip 经过 PFESA，其余 skip 仍直接送入 decoder。

decoder 的真实代码流程为：

```text
Stage_up4
-> CARAFE 2x
-> concat x3
-> Linear Fusion
-> Stage_up3

-> CARAFE 2x
-> concat x2
-> Linear Fusion
-> Stage_up2

-> CARAFE 2x
-> concat x1
-> Linear Fusion
-> Stage_up1

-> CARAFE 4x
-> 1x1 Conv
-> Segmentation Mask
```

其中通道变化是逐级变小的：

```text
Stage_up4: H/32 x W/32 x C4
CARAFE 2x + x3 fusion -> Stage_up3: H/16 x W/16 x C3
CARAFE 2x + x2 fusion -> Stage_up2: H/8  x W/8  x C2
CARAFE 2x + x1 fusion -> Stage_up1: H/4  x W/4  x C1
CARAFE 4x + 1x1 Conv -> Output: H x W x Ncls
```

因此，图中最关键的结构关系是：

- encoder 负责提取多尺度特征；
- PFESA 只作用在 skip feature 上；
- decoder 通过 CARAFE 上采样后，与 PFESA-refined skip feature 进行 concat；
- concat 后通过线性层压缩通道，再进入对应 decoder stage；
- 最后通过 CARAFE 4x 恢复到原图分辨率，并用 1x1 Conv 输出分割 mask。

需要特别注意，decoder 不是三条互相独立的并行分支，而是从 Stage 4 bottleneck 开始，依次经过 `x3 -> x2 -> x1` 三次 skip fusion 的逐级上采样路径。

### 当前架构图检查结论

当前 `模型架构图.png` 的整体思路是对的：左侧是 CSWin-UNet encoder，中间是可选 PFESA skip refinement，右侧是 CSWin-UNet decoder + CARAFE。它可以用于表达“在 skip feature 上插入 PFESA”的核心想法。

但图中有两个地方需要按真实代码修正或在汇报时特别说明：

- decoder 主流程应当是一条连续路径：`Stage_up4 -> x3 fusion -> Stage_up3 -> x2 fusion -> Stage_up2 -> x1 fusion -> Stage_up1 -> final CARAFE 4x`，不是三条互相独立的并行 decoder 支路。
- 每次 CARAFE 2x 上采样后，通道数应当同步降一级：`C4 -> C3 -> C2 -> C1`。也就是说，`x3` 融合处是 `C3`，`x2` 融合处是 `C2`，`x1` 融合处是 `C1`。如果图中对应位置仍标成上一层通道数，应视为标注问题。

## 4. PFESA 模块内部逻辑

PFESA 全称为 Parameter-Free Edge and Structure Attention，是一个无参数频率注意力模块。当前代码中，PFESA 先把 token 形式的 skip feature 转回二维 feature map：

```text
B, L, C -> B, C, H, W
```

然后进行 FFT 频域分解：

```text
Feature
-> FFT
-> Gaussian low-frequency mask
-> high-frequency mask = 1 - low-frequency mask
```

低频分支用于结构注意力：

```text
low-frequency feature -> Structure Attention
```

高频分支用于边界/细节注意力：

```text
high-frequency feature -> Edge Attention
```

最后将两个注意力相加并经过 sigmoid：

```text
attention = sigmoid(edge_attention + structure_attention)
output = attention * original_skip_feature
```

再把特征转回 token 形式：

```text
B, C, H, W -> B, L, C
```

从课题角度看，PFESA 对应的是“频率增强”的尝试：高频分支更偏向边界和局部细节增强，低频分支更偏向结构信息建模，并有助于降低噪声/伪边界对 skip feature 的干扰。这里的“伪边界抑制”是结合课题目标对 PFESA 低频结构分支作用的解释，并不等同于加入了额外的显式伪边界监督分支。

## 5. outputs_new 验证集结果

以下结果来自清洗后的 `new_dataset`，模态为 `ALL`，指标为验证集 best epoch 结果。目前 `outputs_new` 中没有 test 指标，因此这里不能视为最终测试集结论。

| 模型 | best epoch | Val Dice | Delta Dice | Val IoU | Val Boundary IoU | Val HD95 | Val MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| x123 | 86 | 0.885479 | +0.001014 | 0.802455 | 0.337396 | 19.175577 | 0.059714 |
| x23 | 72 | 0.885237 | +0.000772 | 0.801105 | 0.336616 | 19.870700 | 0.060038 |
| x13 | 62 | 0.884897 | +0.000432 | 0.801481 | 0.340406 | 19.502356 | 0.059972 |
| x2 | 78 | 0.884716 | +0.000250 | 0.799833 | 0.331567 | 19.712700 | 0.059763 |
| baseline | 61 | 0.884466 | 0 | 0.800809 | 0.338314 | 19.663332 | 0.059606 |
| x12 | 98 | 0.884278 | -0.000188 | 0.800294 | 0.337150 | 20.416333 | 0.060493 |
| x3 | 63 | 0.882825 | -0.001641 | 0.797871 | 0.333621 | 20.143431 | 0.061558 |
| x1 | 72 | 0.882699 | -0.001766 | 0.797907 | 0.334613 | 20.108473 | 0.062094 |

## 6. 结果分析

从验证集 best Dice 看，`x123` 是当前综合表现最好的配置，相比 baseline 有小幅提升：

```text
Dice: +0.001014
IoU:  +0.001646
HD95: -0.487755
```

这说明在清洗后的 `new_dataset` 上，同时对 `x1 + x2 + x3` 三条 skip feature 引入 PFESA，有一定的综合收益。

如果单独关注边界指标，`x13` 的 Boundary IoU 最好：

```text
x13 Boundary IoU: 0.340406
baseline Boundary IoU: 0.338314
```

这说明浅层 `x1` 和深层 `x3` 的组合可能对边界区域有一定帮助，但综合 Dice 不如 `x123`。

`x1` 和 `x3` 在这批结果中低于 baseline，说明单独在某一条 skip 上加入 PFESA 不一定稳定有效。尤其是 `x1`，虽然浅层包含最多边界细节，但也更容易包含噪声和伪边界，因此单独增强浅层高频信息可能会带来干扰。

## 7. 当前结论

- 当前模型可以描述为：CSWin-UNet with PFESA on configurable skip features。
- PFESA 是在 CSWin-UNet skip feature 上后加的频率增强模块，不是完整替换网络。
- 在 `new_dataset` 的验证集结果中，综合最优配置是 `x123`。
- 边界指标最优配置是 `x13`。
- `x1/x3` 单独使用时效果不如 baseline，不建议作为主推配置。
- 当前结果来自清洗后的 `new_dataset`，不能和原始数据集结果直接混为同一组主实验结论。
- 由于目前没有 test 指标，后续需要用各模型的 `best.pth` 在 `new_dataset` 的 test split 上跑 `test.py`，再判断最终泛化表现。
