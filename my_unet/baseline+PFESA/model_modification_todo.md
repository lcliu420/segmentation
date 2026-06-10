# PFESA 后续修改清单

本文档专门记录后续如果要修改 PFESA，可以从哪些位置入手，以及这些改动为什么可能改善 Dice、IoU、Boundary IoU、HD95 和 MAE。当前训练和测试输出中，HD95 默认指 `hd95_medpy` 口径；旧 `hd95` 只保留函数实现用于历史对照，不再参与新实验输出。文档只作为研究清单，不代表已经修改对应算法。

## 当前 PFESA 的问题

当前 PFESA 实现位置：

```text
baseline+PFESA/networks/PFESA.py
```

当前 PFESA 的核心输出形式是：

```python
return out_att * x
```

其中 `out_att = sigmoid(Structure Attention + Edge Attention)`。该模块目前只在 `x1/x2/x3` skip feature 送入 decoder concat 之前使用。

- PFESA 相比 baseline 的收益很小，旧 `x123` 只比 baseline 高约 `+0.001` Dice。
- `x1` 和 `x3` 单独使用时低于 baseline，说明单尺度频率增强可能放大噪声或破坏原始 skip 分布。
- `x13` 的 Boundary IoU 相对最好，说明浅层细节加深层语义可能更有利于边界，但综合 Dice 仍不够高。
- `x123` 综合略好但不稳定，新旧重复实验存在波动。
- 胃镜图像中的反光、血管纹理、褶皱、黑边都是高频响应，高频不一定等于真实病灶边界。
- 当前 PFESA 没有边界监督、没有前景/语义 gate，也没有显式伪边界抑制机制。
- `x1/x2/x3` 当前共用同一个 `base_ratio=0.1`，但不同分辨率下频率含义并不一致。

## 文件管理约定

当前已经加入 PFESA 多版本管理：

```text
baseline+PFESA/networks/PFESA.py
baseline+PFESA/networks/pfesa_factory.py
```

约定如下：

- `PFESA.py` 永远作为原始版本，variant 名称为 `original`。
- 后续不要反复覆盖 `PFESA.py`，每个新想法都单独建一个 `PFESA_xxx.py`。
- 每个变体文件内部都提供统一接口 `TokenPFESA`，这样 `cswin_unet.py` 不需要跟着每个变体反复修改。
- 新增变体后，只在 `pfesa_factory.py` 里增加一条注册关系。
- 训练和测试通过 `--pfesa_variant xxx` 选择版本。
- 默认 `--pfesa_variant original`，旧命令不传这个参数时仍使用原始 PFESA。

推荐命名：

| 文件 | variant 名 | 用途 |
|---|---|---|
| `PFESA.py` | `original` | 原始 PFESA，对照组 |
| `PFESA_residual.py` | `residual` | 残差式 PFESA，已注册 |
| `PFESA_scale_ratio.py` | `scale_ratio` | 分尺度 `base_ratio` |
| `PFESA_weighted_ea_sa.py` | `weighted_ea_sa` | EA/SA 加权融合 |
| `PFESA_guided.py` | `guided` | 前景/边界引导 PFESA |

新增一个 PFESA 变体的固定流程：

1. 新建 `baseline+PFESA/networks/PFESA_xxx.py`。
2. 文件中提供 `TokenPFESA`，输入输出保持 `[B, L, C] -> [B, L, C]`。
3. 在 `pfesa_factory.py` 中注册：`"xxx": XxxTokenPFESA`。
4. 训练时传入：`--pfesa_variant xxx`。
5. 测试同一个 checkpoint 时也传入同样的 `--pfesa_variant xxx`。
6. 在实验记录里写清楚 `pfesa_variant`、`pfesa_skip_mode` 和 `pfesa_base_ratio`。

已实现的 residual 版本使用：

```bash
--pfesa_variant residual
```

训练 residual checkpoint 后，测试同一个 checkpoint 时也必须传入 `--pfesa_variant residual`。

已支持固定分尺度 `base_ratio`，顺序固定为 `x1 x2 x3`：

```bash
--pfesa_base_ratios 0.05 0.1 0.15
```

推荐和 residual 组合使用：

```bash
--pfesa_variant residual --pfesa_base_ratios 0.05 0.1 0.15
```

## PFESA 改造优先级清单

| 优先级 | 改法 | 为什么可能提升 | 预期改善指标 | 风险 |
|---|---|---|---|---|
| 高 | 残差式 PFESA：不要 `out = att * x`，改成 `out = x + gamma * att * x`，`gamma` 可学习且初始为 0 或很小 | 避免 PFESA 一开始破坏原始 skip，训练自己决定用多少频域增强 | Dice、IoU 更稳定；MAE 可能下降；HD95 不易被错误增强拉高 | 引入可学习参数，需要和原 PFESA 做消融 |
| 高 | 分尺度 `base_ratio`：给 `x1/x2/x3` 不同频率阈值，例如 `x1=0.05, x2=0.1, x3=0.15`，或改成可学习 | 不同分辨率的频率含义不同，不应共用一个 `0.1` | Dice、IoU 可能更稳；Boundary IoU 可能提升；HD95 可能下降 | 需要调参，错误比例会削弱有效边界 |
| 高 | 边界/前景引导 PFESA：用 decoder 高层语义生成 foreground gate，只在可能病灶区域增强高频 | 抑制背景反光、血管纹理、褶皱等伪边界，避免高频盲目增强 | Boundary IoU、HD95 最可能受益；Dice/IoU 也可能提升 | 需要可靠语义 gate，结构复杂度增加 |
| 高 | EA/SA 加权融合：不要简单 `EA + SA`，改成 `w_e * EA + w_s * SA`，权重可学习或分层设置 | 让模型自动决定高频边界信息和低频结构信息哪个更重要 | Dice、IoU 和 Boundary IoU 可能更平衡；MAE 可能下降 | 权重学习不稳定时可能偏向某一分支 |
| 中 | 多频带 PFESA：不只分低频/高频，增加中频分支 | 医学边界不一定全在最高频，中频可能包含更稳定的形状和轮廓信息 | Boundary IoU、HD95 可能提升；Dice 可能小幅提升 | 频带划分复杂，计算量增加 |
| 中 | PFESA + boundary loss | 让频域增强有边界监督目标，避免盲目增强所有高频 | Boundary IoU、HD95 最可能提升；Dice 可能随边界更准而提升 | boundary loss 权重过大会牺牲区域 Dice |
| 中 | PFESA 只放 `x13` 或 `x23`，不强推 `x123` | 当前结果显示 `x13` 边界最好，`x123` 综合略好但不稳定；减少不必要的尺度干扰 | Boundary IoU 可能提升；HD95 可能更稳定 | Dice 综合最优不一定超过 `x123` |
| 中 | 高频噪声抑制分支：对高频响应过强但低频结构不支持的区域降权 | 针对胃镜伪边界更直接，减少反光、纹理、褶皱被当成边界 | HD95、Boundary IoU 可能明显改善；MAE 可能下降 | 如果抑制过强，真实细小边界也会被压掉 |
| 低 | 把 PFESA 放 decoder feature，而不是 encoder skip | decoder feature 已有更多语义信息，可能比浅层 skip 更能区分真假边界 | Boundary IoU、HD95 可能改善 | 会改变 decoder 特征分布，消融解释更复杂 |
| 低 | 把 FFT 改成 wavelet/DWT | wavelet 有空间局部性，比全局 FFT 更适合局部边界和反光噪声 | Boundary IoU、HD95 可能提升；Dice 可能受益 | 需要新增变换实现，工程复杂度较高 |

## 推荐实验顺序

建议按风险从低到高推进，不要一次叠加多个改动。

1. **Residual PFESA**
   - 最小改动，只改变输出形式。
   - 推荐先试 `gamma` 可学习且初始化为 0。
   - 目标是确认 PFESA 不再破坏原始 skip。

2. **分尺度 `base_ratio`**
   - 在 residual 版本稳定后，再给 `x1/x2/x3` 设置不同频率阈值。
   - 固定分尺度 ratio 已支持，推荐先试固定值，再考虑可学习。
   - 初始候选：`x1=0.05, x2=0.1, x3=0.15`。

3. **EA/SA 加权融合**
   - 在保留 residual 的基础上，让模型学习高频和低频的权重。
   - 可以先用每层两个标量权重，避免过多参数。

4. **PFESA 插入位置重测**
   - 重点比较 `x13`、`x23`、`x123`。
   - 不建议只看 Dice，也要看 Boundary IoU、HD95、MAE。

5. **PFESA + boundary loss**
   - 如果前面改动仍然无法改善 HD95，再加入边界监督。
   - 目标是让频域增强真正服务于真实边界，而不是所有高频。

6. **边界/前景引导 PFESA 或高频噪声抑制分支**
   - 如果可视化显示反光、血管、褶皱导致假边界，再做这类改动。
   - 这是更贴近伪边界抑制的方向。

7. **Wavelet/DWT 替换 FFT**
   - 放在最后作为较大改造。
   - 适合当前面证明局部频率建模确实是瓶颈时再做。

## 每次实验记录模板

```text
实验名称：
pfesa_variant：
PFESA 改动点：
是否 residual：
base_ratio 设置：
EA/SA 融合方式：
PFESA 插入位置：
boundary loss 设想/备注：
训练命令：
测试命令：
随机种子：
num_workers：
best epoch：

Val Dice：
Val IoU：
Val Boundary IoU：
Val HD95 MedPy：
Val MAE：

Test Dice：
Test IoU：
Test Boundary IoU：
Test HD95 MedPy：
Test MAE：

相比 baseline 的变化：
相比原 PFESA x123 的变化：
失败样本变化：
结论：
下一步：
```
