# my_unet 项目上下文总结

本文档用于在后续新对话中快速恢复 `my_unet` 项目的整体背景，重点说明 `baseline`、`baseline+PFESA` 两个核心代码目录，以及当前围绕 CSWin-UNet、PFESA 所做的阶段性改造。

## 1. 项目整体作用

当前项目路径为：

```text
F:\code\git\segmentation\my_unet
```

`my_unet` 是一个医学图像二分类分割实验项目，当前课题方向围绕：

```text
边界频率增强 + 伪边界抑制
```

目前的整体研究路线是：

1. 以 **CSWin-UNet** 作为稳定基底网络。
2. 从参考论文中筛选适合当前课题的模块。
3. 将模块以可开关、可消融的方式接入当前网络结构。
4. 当前第一类改造是：把 **PFESA** 频域注意力模块加入 CSWin-UNet 的 skip feature。
5. 后续如需继续处理边界问题，优先考虑更直接影响 segmentation 分支的边界建模或伪边界抑制机制。

需要注意：`dataset` 和 `new_dataset` 与 `my_unet` 是同级目录，不在 `my_unet` 内部。

```text
F:\code\git\segmentation
  dataset
  new_dataset
  my_unet
```

## 2. 顶层目录说明

`baseline/`

原始 CSWin-UNet 基线实现。这个目录作为干净基底保留，不用于 PFESA 网络结构改造。目前该目录的训练/测试指标已同步为只输出 `hd95_medpy`。

`baseline+PFESA/`

基于 `baseline` 复制出的 PFESA 实验目录。当前所有 PFESA variant、分尺度 `base_ratio` 等实验性改造都集中在这个目录下。

`参考文章/`

存放参考论文 PDF，包括 CSWin-UNet、PFESA，以及后续可能用于边界分支、频率增强、wavelet 高频分支、skip fusion 等方向的论文。

`对应代码/`

存放参考论文对应的开源代码，例如 PFESA 的官方代码，用于对照论文实现和迁移模块。

`outputs/`

原始数据集上的训练、测试输出。

`outputs_new/`

使用清洗后的 `new_dataset` 重新训练得到的输出结果。已有历史 CSV 不回改，因此部分旧结果仍包含旧 `hd95` 列；新训练/测试默认只关注 `hd95_medpy`。

`delete/`

用于全量样本筛查的输出目录。之前将 train、val、test 合并到 `DELETE/test` 中，用现有 `test.py` 统计每张图的指标，辅助找出共同分割差的样本。

`汇报/`

存放模型架构图、汇报 PPT、讲稿、说明文档等材料。

`run_new_all.sh`

服务器端训练脚本，用于在 `new_dataset` 上批量运行 baseline 和 PFESA 不同 skip 组合实验。该脚本后续原则上保持稳定，不再为每个新模块反复修改；新增实验优先使用单条训练命令显式传参。

## 3. baseline：原始 CSWin-UNet 基线

`baseline` 是当前项目的原始基线模型目录，对应 **CSWin-UNet** 的代码实现。

它的作用是：

- 训练原始 CSWin-UNet。
- 作为 PFESA 和后续新模块的对照组。
- 保留干净基线，原则上不在这里做 PFESA 或其它网络结构改造。

核心结构来自 CSWin-UNet 论文：

- Encoder 使用 CSWin Transformer Block。
- 保留 U-Net 式 encoder-decoder 结构。
- 保留多尺度 skip connection。
- Decoder 中使用上采样与 skip feature 融合。
- 使用 CARAFE 做内容感知上采样。
- 当前任务是二分类分割，通常设置 `num_classes=2`。

选择 CSWin-UNet 作为基底的原因：

- 它是一个较强的 Transformer-UNet 医学图像分割基底。
- U-Net 式多尺度 skip connection 明确，`x1`、`x2`、`x3` 插入点清楚，便于做模块消融。
- CSWin Transformer Block 通过横向和纵向条带注意力建模长程上下文，有利于给边界判断提供语义基础。
- CARAFE 上采样关注内容感知的特征重组，有利于恢复空间细节。
- 原始 CSWin-UNet 本身没有显式频率增强、边界分支或伪边界抑制机制，因此适合作为当前课题的模块验证平台。

## 4. baseline+PFESA：PFESA skip feature 消融实验

`baseline+PFESA` 是在 `baseline` 基础上复制出的实验目录，用于实现：

```text
CSWin-UNet + configurable PFESA variants on skip features
```

关键原则：

- PFESA 不是替换整个 CSWin-UNet。
- PFESA 不是 CSWin-UNet 原论文自带模块。
- PFESA 只作用在送入 decoder concat 的 skip feature 上。
- PFESA 不改变 encoder 主干下采样路径。
- PFESA 不改变 CSWin Transformer Block 本身。
- PFESA 不改变 CARAFE 上采样结构。
- 后续小改动尽量拆成独立模块，通过参数开关启用，不随意复制 `cswin_unet.py`。

当前支持的 `pfesa_skip_mode` 包括：

```text
none
x1
x2
x3
x12
x13
x23
x123
```

含义：

- `none`：关闭 PFESA，用于 sanity check 或对照。
- `x1`：只增强浅层 skip feature。
- `x2`：只增强中层 skip feature。
- `x3`：只增强深层 skip feature。
- `x12`：增强 `x1 + x2`。
- `x13`：增强 `x1 + x3`。
- `x23`：增强 `x2 + x3`。
- `x123`：增强 `x1 + x2 + x3`。

当前模型数据流可以简化为：

```text
Input
 -> CSWin-UNet Encoder
 -> 得到 x1, x2, x3, bottleneck
 -> 选中的 skip feature 经过 PFESA variant
 -> Decoder 中与上采样特征 concat
 -> 输出 segmentation mask
```

PFESA 插入位置的关键点是：

```text
Stage output -> optional PFESA -> decoder concat
Stage output -> Patch Merging -> next encoder stage
```

也就是说，PFESA 只增强送入 decoder 的 skip，不影响后续 encoder 主干路径。

## 5. PFESA variant 管理

当前 PFESA 已从单一文件演进为 variant 管理方式，避免后续反复直接改乱 `cswin_unet.py`。

相关文件：

```text
baseline+PFESA/networks/PFESA.py
baseline+PFESA/networks/PFESA_residual.py
baseline+PFESA/networks/pfesa_factory.py
```

当前可选 variant：

| variant | 文件 | 说明 |
|---|---|---|
| `original` | `PFESA.py` | 原始 PFESA，输出形式为 `out_att * x` |
| `residual` | `PFESA_residual.py` | 残差式 PFESA，输出形式为 `x + gamma * out_att * x` |

`residual` 版本中：

```python
self.gamma = nn.Parameter(torch.zeros(1))
return x + self.gamma * out_att * x
```

因此初始化时近似等价于原始 skip，不会一开始强行破坏 skip feature；训练过程中每个 PFESA 插入点会各自学习一个 `gamma`。

训练时通过参数选择：

```bash
--pfesa_variant original
--pfesa_variant residual
```

## 6. 分尺度 base_ratio

原始 PFESA 中 `x1/x2/x3` 共用同一个 `base_ratio=0.1`。但不同 skip feature 的空间分辨率不同，同一个频率阈值在不同尺度上的含义并不一致。

因此当前 `baseline+PFESA` 支持两种方式：

单一 ratio，兼容旧命令：

```bash
--pfesa_base_ratio 0.1
```

等价于：

```text
x1 = 0.1
x2 = 0.1
x3 = 0.1
```

分尺度 ratio：

```bash
--pfesa_base_ratios 0.05 0.1 0.15
```

顺序固定为：

```text
x1 x2 x3
```

即：

```text
x1 = 0.05
x2 = 0.10
x3 = 0.15
```

分尺度 ratio 对 `original` 和 `residual` 两个 PFESA variant 都生效。

## 7. 边界监督当前状态

之前曾尝试设计 decoder 端可选 boundary head / boundary loss，但该部分当前已经从代码中移除。

当前 `baseline+PFESA` 仍然是单输出 segmentation-only 结构：

```text
decoder final feature
   -> segmentation head
   -> seg_logits
```

也就是说：

- 训练脚本不再提供 `use_boundary_head`、`boundary_loss_weight`、`boundary_width` 参数。
- 模型 forward 只返回 segmentation logits，不返回 boundary logits。
- 测试脚本只加载 segmentation-only checkpoint。
- 后续如果重新设计边界监督，应优先考虑让边界信息更直接影响 segmentation 分支，而不是只额外挂一个辅助 head。

## 8. 指标口径说明

当前训练和测试主要输出：

```text
Dice
IoU
Boundary IoU
HD95 MedPy
MAE
```

之前项目中同时存在旧 `hd95` 和新 `hd95_medpy`。现在已统一为：

- 新训练/测试只计算和输出 `hd95_medpy`。
- 旧 `hd95_score` 函数仍保留在 `metrics.py` 中，用于历史对照。
- 旧 `hd95_score` 不再被 `compute_metrics` 调用。
- 旧 `outputs_new` CSV 不回改，因此历史结果表中仍可能出现旧 `Val HD95` 列。

后续分析边界相关改动时，重点看：

```text
Boundary IoU 越高越好
HD95 MedPy 越低越好
Dice 不能明显下降
MAE 不应明显变差
```

如果后续重新引入边界相关监督，理想表现应该是：

```text
Dice 基本不降
Boundary IoU 上升
HD95 MedPy 下降
MAE 不变或下降
```

## 9. 核心参考论文一：CSWin-UNet

参考论文：

```text
Liu 等 - 2025 - CSWin-UNet Transformer UNet with cross-shaped windows for medical image segmentation.pdf
```

这篇论文提供当前项目的基底网络。

主要结构：

- Convolutional Token Embedding。
- 多阶段 CSWin Transformer Encoder。
- 横向和纵向 stripe self-attention。
- U-Net 式 skip connection。
- CARAFE 内容感知上采样。
- Decoder 逐级恢复空间分辨率并融合 skip feature。

与当前课题的关系：

- CSWin-UNet 本身不是频率增强模型。
- CSWin-UNet 本身没有显式边界分支。
- CSWin-UNet 本身没有专门的伪边界抑制模块。
- 但它提供了强基底、清晰 skip 层级和较强上下文建模能力，适合承载当前课题中的新增模块。

因此，选择 CSWin-UNet 的逻辑不是“它已经解决了频率增强和伪边界抑制”，而是“它适合作为稳定且结构清晰的改造平台”。

## 10. 核心参考论文二：PFESA

参考论文：

```text
Li 等 - PFESA FFT-based Parameter-Free Edge and Structure Attention for Medical Image Segmentation.pdf
```

这篇论文提供当前已经缝合到 `baseline+PFESA` 中的频域注意力模块。

PFESA 的核心思想：

- 对特征做 FFT。
- 通过频域解耦得到高频和低频信息。
- 高频部分强调边缘、纹理和细节。
- 低频部分保留主体结构信息。
- 高频分支生成 Edge Attention。
- 低频分支生成 Structure Attention。
- 两个注意力融合后乘回原始特征。

简化流程：

```text
skip token [B, L, C]
 -> reshape 为 feature map [B, C, H, W]
 -> FFT
 -> 高频/低频解耦
 -> 高频分支做 Edge Attention
 -> 低频分支做 Structure Attention
 -> EA + SA 后 sigmoid
 -> 调制原始 skip feature
 -> reshape 回 token
```

与当前课题的关系：

- 对“频率增强”直接有帮助。
- 高频分支与“边界细节增强”相关。
- 低频结构分支有助于保持主体结构，并可能缓解部分噪声或伪边界干扰。
- 但 PFESA 不是显式伪边界抑制模块，也没有直接判断“哪些高频是真边界”的能力。

因此，PFESA 适合作为频域增强起点，但后续仍需要 residual、分尺度 ratio、边界监督、前景 gate 或伪边界抑制机制继续补强。

## 11. 当前模型结构总结

当前已经实现的模型能力可以概括为：

```text
CSWin-UNet
 + configurable PFESA variants on skip features
```

其中：

- 基底：CSWin-UNet。
- PFESA 插入位置：`x1`、`x2`、`x3` 三个 skip feature 的不同组合。
- PFESA variant：`original`、`residual`。
- PFESA ratio：支持单一 `base_ratio` 或分尺度 `base_ratios`。
- 主干路径：不变。
- decoder：仍沿用 CSWin-UNet 的 decoder 与 CARAFE 上采样。

从课题角度看：

- 频率增强：PFESA 是直接相关模块。
- 边界增强：PFESA 高频分支是当前已经接入的相关尝试；decoder 边界监督暂时移除，后续如需要可重新设计。
- 伪边界抑制：目前仍不够显式，后续可能需要 foreground gate、语义 gate 或高频噪声抑制分支。

## 12. 当前 outputs_new 实验结论

`outputs_new` 中保存的是使用清洗后的 `new_dataset` 重新训练后的结果。

重要说明：

```text
当前 outputs_new 是清洗后 new_dataset 上的验证集 best epoch 结果，不是最终 test 结果。
```

旧结果表使用当时的历史指标口径，仍显示 `Val HD95`。新训练/测试默认应看 `HD95 MedPy`。

验证集 best 结果摘要：

| 模型 | best epoch | Val Dice | ΔDice | Val IoU | Val Boundary IoU | Val HD95 | Val MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| x123 | 86 | 0.885479 | +0.001014 | 0.802455 | 0.337396 | 19.175577 | 0.059714 |
| x23 | 72 | 0.885237 | +0.000772 | 0.801105 | 0.336616 | 19.870700 | 0.060038 |
| x13 | 62 | 0.884897 | +0.000432 | 0.801481 | 0.340406 | 19.502356 | 0.059972 |
| x2 | 78 | 0.884716 | +0.000250 | 0.799833 | 0.331567 | 19.712700 | 0.059763 |
| baseline | 61 | 0.884466 | 0 | 0.800809 | 0.338314 | 19.663332 | 0.059606 |
| x12 | 98 | 0.884278 | -0.000188 | 0.800294 | 0.337150 | 20.416333 | 0.060493 |
| x3 | 63 | 0.882825 | -0.001641 | 0.797871 | 0.333621 | 20.143431 | 0.061558 |
| x1 | 72 | 0.882699 | -0.001766 | 0.797907 | 0.334613 | 20.108473 | 0.062094 |

当前结论：

- 综合指标最优的是 `x123`，说明多尺度 skip 同时做 PFESA 增强有小幅收益。
- 边界指标最优的是 `x13`，说明浅层细节和深层语义组合可能更有利于边界。
- `x1` 和 `x3` 单独使用时低于 baseline，说明单独增强某一尺度可能会放大噪声，或者破坏原本的 skip 分布。
- 当前收益幅度整体较小，不能夸大为决定性提升。
- 这些结果来自清洗后的 `new_dataset`，不能直接和原始数据集结果混为同一组主实验结论。
- residual、分尺度 ratio 属于后续新增实验能力，是否带来提升仍需要同配置对照实验验证。

## 13. 数据清洗相关说明

之前为了找出共同分割较差的样本，使用了 `DELETE` modal：

- 将 `ALL` 的 train、val、test 全部放入 `DELETE/test`。
- 使用 baseline 和 PFESA-x1 的 `test.py` 对全量样本输出逐图指标。
- 根据逐图指标筛选出可能有问题的样本。
- 再根据筛选结果复制保留样本，构建新的 `new_dataset`。

筛选规则是：任意一个模型失败就剔除。

剔除条件包括：

- baseline 或 PFESA-x1 任一模型 `dice < 0.70`。
- baseline 或 PFESA-x1 任一模型 `boundary_iou < 0.15`。
- baseline 或 PFESA-x1 任一模型 `hd95 > 50`。

注意：这个清洗策略比较激进，适合做探索实验。如果要写论文主实验，仍建议保留原始测试集结果作为公平对照。

## 14. 后续研究方向

当前已经完成的是：

```text
CSWin-UNet + PFESA skip frequency enhancement
```

当前已经具备的实验能力：

1. PFESA skip 插入位置消融：`x1/x2/x3/x12/x13/x23/x123`。
2. PFESA variant 管理：`original`、`residual`。
3. 分尺度 `base_ratio`：`--pfesa_base_ratios 0.05 0.1 0.15`。
4. 指标统一：新实验默认只输出 `hd95_medpy`。
仍待继续验证：

1. `residual` 是否比 `original` 更稳定。
2. 分尺度 `base_ratio` 是否改善边界指标。
3. 后续如果重新设计边界监督，边界信息是否能真正反馈到 segmentation head。

后续可以继续考虑：

1. 在 `x13` 或 `x123` 的基础上补充最终 test split 结果。
2. 对比原始数据集和清洗后 `new_dataset` 的结果，避免结论混淆。
3. 让边界信息更直接参与 segmentation 分支，而不是只作为辅助 head。
4. 对 EA/SA 做可学习加权融合。
5. 增加 foreground gate 或 decoder 语义 gate，抑制反光、血管纹理、褶皱等伪边界。
6. 增加高频噪声抑制机制。
7. 继续参考其它论文中的 boundary branch、wavelet 高频分支、skip fusion 或 MoE skip connection。

总体来说，当前 `my_unet` 项目的阶段是：

```text
已经完成 CSWin-UNet 基底理解、PFESA skip 消融实验、
PFESA variant 管理、residual PFESA、以及分尺度 ratio。
当前已移除 decoder 边界监督插槽。
下一步应通过严格同配置对照实验判断 PFESA 相关改动是否真正改善
Boundary IoU 和 HD95 MedPy，并继续围绕伪边界抑制机制设计。
```
