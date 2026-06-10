# 参考论文阅读笔记：面向“频率增强 + 边界分支 + 伪边界抑制”的模块筛选

本文档用于记录当前 `参考文章` 目录下 8 篇论文的逐篇阅读结果。重点不是完整翻译论文，而是判断每篇论文的网络结构、核心改进、能否迁移到当前 CSWin-UNet 基底，以及是否服务于胃镜病灶分割中的弱边界、频率信息和伪边界干扰问题。

当前课题背景：本项目数据集包含 WL/NBI 胃镜病灶分割图像。已有数据诊断显示，主要困难不是极小病灶，而是局部边界灰度差很低、病灶和正常黏膜过渡平滑、NBI 中血管纹理和反光容易形成伪边界。因此后续模块选择应优先围绕“增强真实病灶边界相关高频信息，同时抑制纹理/反光/褶皱等非病灶高频干扰”展开。

## 1. LB-UNet: A Lightweight Boundary-assisted UNet for Skin Lesion Segmentation

### 1. 论文要解决的问题

LB-UNet 面向皮肤病灶分割，主要解决两个问题：一是轻量化部署需求，二是皮肤病灶边界模糊导致分割不准。论文认为许多轻量模型虽然参数少，但对模糊边界的建模能力不足，因此提出轻量 U-Net 结构并引入边界辅助模块。

### 2. 网络整体结构

整体是 6-stage U-Net。encoder 和 decoder 的通道数较小，约为 `{8, 16, 24, 32, 48, 64}`。前几层使用普通卷积，深层用 Group Shuffle Attention (GSA) 替换部分卷积以降低参数量和计算量。下采样用 `2x2` 卷积替代 max pooling。与普通 U-Net 最大的不同是引入 Prediction Map Auxiliary (PMA)，让中间层不仅预测区域，还预测边界，并把预测信息反向用于 skip connection。

### 3. 核心模块/改进点

- GSA (Group Shuffle Attention)：按通道分组，每组内用轻量注意力，再 shuffle 融合，用于降低参数和计算量。
- PMA (Prediction Map Auxiliary)：由 RBP、GBG、PIF 三部分组成。
- RBP (Segmentation Region and Boundary Prediction)：从 decoder 中间特征分别预测区域图 `R` 和边界图 `B`。对于第 2-4 stage，同时使用区域和边界预测增强特征；对于更深层只使用区域预测。
- GBG (GA-Based Boundary Generator)：用边界线和遗传算法选择的关键点生成更强的边界监督图。边界线来自传统边缘检测，关键点用于强化边界形状信息。
- PIF (Prediction Information Fusion)：把预测到的区域图和边界图作为 attention 信息加入 skip connection，形式类似 `decoder feature + encoder feature + encoder feature * region + encoder feature * boundary`。

### 4. 与当前 CSWin-UNet 的关系

当前 CSWin-UNet 已经有 3 个 skip connection：`x3 -> stage_up3`、`x2 -> stage_up2`、`x1 -> stage_up1`。LB-UNet 的 PIF 思想可以迁移到这些 skip 融合位置，不需要替换 CSWin 主干。当前 CSWin-UNet 的 skip 是直接 concat 后 linear 降维，缺少“区域/边界选择性引导”；PIF 可作为更有针对性的 skip fusion 方案。

### 5. 对本课题的可借鉴点

- 频率增强：直接帮助有限。LB-UNet 不做频域分解，但边界预测图可以作为后续频率增强的空间约束，避免全图增强高频。
- 边界分支：非常有帮助。RBP 是一个轻量边界分支参考，可以从 CSWin-UNet 高分辨率 decoder feature 中预测 boundary map，并用 mask 生成边界标签监督。
- 伪边界抑制：有间接帮助。PIF 用真实边界预测调制 skip feature，可减少无关区域特征进入 decoder；但论文没有专门区分真实边界和纹理/反光伪边界，需要结合频域或背景抑制策略扩展。

### 6. 是否建议后续缝合

建议借鉴，但不要照搬完整 LB-UNet。最值得迁移的是 RBP + PIF 思想：在 CSWin-UNet decoder 高分辨率阶段增加轻量区域/边界预测头，并用边界图调制 skip fusion。GBG 的遗传算法边界关键点较复杂，且当前胃镜 mask 已可直接形态学生成边界，第一阶段不建议引入 GA。

## 2. PFESA: FFT-based Parameter-Free Edge and Structure Attention for Medical Image Segmentation

### 1. 论文要解决的问题

PFESA 关注 U-Net skip connection 中的两个问题：浅层 encoder 特征噪声较多，直接传给 decoder 会干扰重建；下采样会导致高频边缘信息衰减。普通注意力模块有参数，容易在医学小数据上过拟合，且可解释性弱。因此论文提出基于 FFT 的无参数频域注意力，在 skip connection 中同时增强边缘和结构。

### 2. 网络整体结构

PFESA 不是完整新主干，而是一个 plug-and-play attention 模块，可插入任意 U-Net/Transformer U-Net 的 skip connection。输入输出通道一致，不引入可训练参数。论文在 2D 和 3D 医学分割任务上把 PFESA 加到 skip connection 中，与 SE、CBAM、ECA、SimAM 等 attention 做对比。

### 3. 核心模块/改进点

- Frequency-Domain Feature Decoupling：对输入特征做 FFT，用 Gaussian filter 分离低频结构成分和高频边缘成分，再 inverse FFT 回到空间域。
- Edge Attention (EA)：对高频特征做基于局部能量/方差的 SNR 注意力，强化高频边缘响应，降低噪声主导区域权重。
- Structure Attention (SA)：对低频结构特征做均值-方差形式的 SNR 注意力，增强器官/病灶整体形态结构。
- Fusion：将 EA 和 SA 相加后经过 sigmoid 得到注意力图，再与原始输入特征逐元素相乘。整个模块无训练参数。

### 4. 与当前 CSWin-UNet 的关系

PFESA 非常适合当前 CSWin-UNet 的 skip connection。当前结构中 encoder skip feature 直接与 decoder upsample feature concat，可能把浅层纹理、反光、血管等噪声一并传入。PFESA 可作为 concat 前的 skip refinement：对 `self.x1/self.x2/self.x3` 或上采样后的 decoder feature 做频域筛选，再融合。

### 5. 对本课题的可借鉴点

- 频率增强：高度相关。PFESA 明确把高频当作边缘细节，把低频当作结构信息，适合支撑“频率增强”这个创新点。
- 边界分支：间接相关。PFESA 本身不输出 boundary map，但 EA 可作为边界敏感特征的生成方式，与显式边界分支结合。
- 伪边界抑制：很有参考价值。PFESA 不只是增强高频，还用 SNR 思路降低高频噪声区域响应；这和 NBI 中血管纹理、反光点等伪边界问题契合。

### 6. 是否建议后续缝合

强烈建议优先考虑。它和当前课题标题中的“频率增强”最直接匹配，同时实现成本相对低、参数无关、适合医学小数据。建议第一版不要全层都加，可先在 `x1` 或 decoder 最后高分辨率层做轻量实验，再比较 `x1/x2/x3` 多尺度加入效果。

## 3. Rethinking Boundary Detection in Deep Learning-Based Medical Image Segmentation

### 1. 论文要解决的问题

该论文重新思考医学图像分割中的边界检测。作者认为 CNN 擅长局部，Transformer 擅长全局，但许多方法没有充分利用传统边缘算子的显式边界先验。医学图像中对象边界往往弱、模糊、不完整，因此论文提出 CTO，把 CNN、Transformer 和边缘检测 Operator 结合起来。

### 2. 网络整体结构

CTO 是 encoder-decoder 结构。encoder 是双流：主流 CNN stream 使用 Res2Net 捕获局部多尺度特征，辅助 StitchViT stream 捕获长程依赖。两个 stream 的特征融合后进入 decoder。decoder 是 boundary-guided decoder，通过 Sobel 算子从特征中提取边界 mask，再通过 Boundary-Injected Module (BIM) 把边界增强特征注入到多级 decoder。

### 3. 核心模块/改进点

- StitchViT：通过 stitch operation 让不同 attention head 采样不同稀疏率的位置，用较低代价捕获长程依赖。
- BEM (Boundary-Extracted Module)：对低层和高层 CNN 特征使用固定 Sobel 卷积核，得到水平/垂直梯度，经过 sigmoid 后增强输入特征；再融合高低层边界特征，输出边界增强特征，并用 ground truth boundary map 监督。
- BIM (Boundary-Injected Module)：把 BEM 生成的 boundary-enhanced feature 注入 decoder。核心是 BIO (Boundary Injection Operation)，包含 foreground path 和 background path。
- Background path：使用 `(1 - sigmoid(previous decoder feature))` 形成背景注意力，强调背景区域建模，帮助减少背景误分。
- Loss：分割损失使用 CE + mIoU，多级 deep supervision；边界损失使用 Dice，最终总损失是 segmentation loss + 加权 boundary loss。

### 4. 与当前 CSWin-UNet 的关系

当前 CSWin-UNet 已经有 Transformer 型主干，不一定需要再引入 Res2Net + StitchViT 双流。更值得借鉴的是 BEM/BIM 思想：用固定 Sobel 或形态学边界从特征/标签中产生显式边界监督，并将边界信息注入 decoder。CSWin-UNet 的 decoder stage 和 skip fusion 位置都适合放轻量 BIM 变体。

### 5. 对本课题的可借鉴点

- 频率增强：间接相关。Sobel 本质是高频/梯度算子，可作为空间域高频边界提取方式；但不是完整频域模块。
- 边界分支：高度相关。BEM 是明确边界分支参考，并且不需要额外标注，只需从 mask 生成边界监督。
- 伪边界抑制：很有参考价值。BIM 的 background path 明确建模背景，`1 - foreground attention` 的思路可迁移为伪边界抑制分支，让模型学习哪些边缘属于背景或非病灶结构。

### 6. 是否建议后续缝合

建议借鉴 BEM + background-aware injection，不建议迁移完整 CTO 双流主干。当前 CSWin-UNet 已有全局建模能力，重复加 ViT stream 意义不大。更合理的做法是在 CSWin-UNet decoder 中增加轻量边界头和背景/伪边界抑制门控。

## 4. CSWin-UNet: Transformer UNet with Cross-Shaped Windows for Medical Image Segmentation

### 1. 论文要解决的问题

CSWin-UNet 解决的是医学图像分割中 CNN 感受野有限、普通 Transformer 计算开销大、Swin Transformer 局部窗口交互不足的问题。论文希望在保持较低计算量的同时增强长程依赖和边界细节恢复。

### 2. 网络整体结构

整体是 U-shaped encoder-decoder。输入 `224x224` 图像先经过 `7x7 stride=4` convolutional token embedding，得到 `56x56` token。encoder 有 4 个 stage，分辨率依次为 `56x56 -> 28x28 -> 14x14 -> 7x7`，通道逐层翻倍。decoder 对称上采样，使用 CARAFE 恢复分辨率，并通过 3 个 skip connection 融合 encoder 特征。最后用 `4x CARAFE` 恢复到原图尺寸并通过 `1x1 conv` 输出 segmentation mask。

### 3. 核心模块/改进点

- CSWin Self-Attention：将多头注意力分成两组，一组做横向条带 attention，一组做纵向条带 attention，最后 concat。这样比全局 attention 省计算，比局部窗口 attention 有更好的长程交互。
- LePE (Locally-enhanced Positional Encoding)：通过 depthwise convolution 对 value 分支加入局部位置信息。
- Merge Block：encoder 中用 `3x3 stride=2` conv 降采样，同时通道翻倍。
- CARAFE：decoder 中用内容感知重组上采样替代 bilinear 或 transposed convolution，有利于细节和边界恢复。
- 消融结论：`[1, 2, 9, 1]` block 配置较优；3 个 skip connection 较优；CARAFE 比 bilinear 和 transposed convolution 效果更好。

### 4. 与当前 CSWin-UNet 的关系

这是当前项目的基底论文，当前代码基本对应论文结构。`networks/cswin_unet.py` 中 `CSWinTransformer` 实现完整网络；`LePEAttention` 和 `CSWinBlock` 实现 cross-shaped window attention；`CARAFE/CARAFE4` 实现上采样；`forward_features` 存储 `x1/x2/x3` 作为 skip feature；`forward_up_features` 中 concat skip 后用 linear 降维。

### 5. 对本课题的可借鉴点

- 频率增强：直接帮助较弱。CSWin-UNet 没有显式频域模块，但 CARAFE 保细节，可以作为频率增强后的 decoder 基底。
- 边界分支：直接帮助较弱。论文声称 CARAFE 有助于边界恢复，但没有显式边界监督或边界预测头。
- 伪边界抑制：基本没有专门设计。CSWin attention 能利用上下文减少部分误分，但没有针对反光、血管纹理、褶皱伪边界的抑制机制。

### 6. 是否建议后续缝合

建议继续作为主干基底和 baseline，而不是作为创新模块来源。后续改造应尽量保持 CSWin-UNet 主体不动，优先在 skip fusion、decoder 高分辨率层、辅助边界监督和频域注意力处增加模块，这样消融清晰、风险较低。

## 5. Rethinking U-Net: Task-Adaptive Mixture of Skip Connections for Enhanced Medical Image Segmentation

### 1. 论文要解决的问题

该论文认为 U-Net 的传统 skip connection 是固定的一对一连接，但不同 decoder stage 对 encoder 信息的需求并不相同。固定 skip 可能导致语义差距和冗余特征传递，限制跨任务泛化能力。因此论文提出 Task-Adaptive Mixture of Skip Connections (TA-MoSC)，把 skip connection 视为任务分配问题。

### 2. 网络整体结构

论文提出 UTANet，在 U-Net 基础上用 TA-MoSC 替换原始 skip connection。TA-MoSC 包含 Feature Aggregation Stripe、Router Bank、Skip-Connection Expert Bank 和 Docker。先把多层 encoder feature resize 到同一尺度并 concat，形成聚合特征；不同 decoder stage 的 router 根据聚合特征选择专家组合；专家输出经 Docker 调整后送入对应 decoder。

### 3. 核心模块/改进点

- Feature Aggregation Stripe：将 `E1-E4` resize 到相同大小后 concat，再用 `1x1 conv` 降维，形成包含多层语义的统一特征。
- Router Bank：每个 decoder stage 一个 gate，根据全局 pooled 聚合特征生成专家选择概率。
- SC Bank / Experts：多个轻量卷积专家共享给所有 skip stage，每个专家是小型卷积子网络。
- Top-K sparse routing：每个 stage 只激活概率最高的 K 个专家，论文中 K=2 效果较好。
- Balanced Expert Utilization (BEU)：包括 Expert Variance Loss 和 Unused Experts Handling，避免某些专家长期不用。
- 训练策略：先训练原始 encoder-decoder，再冻结主干训练 TA-MoSC，降低训练难度。

### 4. 与当前 CSWin-UNet 的关系

当前 CSWin-UNet 的 skip fusion 是固定 concat，确实存在和 TA-MoSC 所说类似的问题：`x1/x2/x3` 直接送到对应 decoder 层，没有根据当前样本、模态或边界难度动态选择信息。TA-MoSC 可以启发“自适应 skip 融合”，但完整 MoE 机制会明显增加工程复杂度。

### 5. 对本课题的可借鉴点

- 频率增强：间接相关。可以把“频域专家/边界专家/语义专家”做成 expert bank，但这属于后续复杂版本。
- 边界分支：间接相关。router 可以根据边界难度选择更偏边界的专家，但论文本身没有显式边界监督。
- 伪边界抑制：有潜在价值。可设计一个“抑制专家”专门处理反光/纹理背景，但原论文没有现成伪边界模块，需要较大改造。

### 6. 是否建议后续缝合

第一阶段不建议完整缝合 TA-MoSC。它适合作为后续增强 skip fusion 的高级方案，但当前课题主线是频率和边界，MoE skip 会引入额外变量，消融解释变复杂。可先借鉴“skip 不应直接 concat，应先筛选/加权”的思想，用更轻量的 frequency/boundary gate 实现。

## 6. CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion

### 1. 论文要解决的问题

CMUNeXt 关注轻量医学分割中的全局上下文不足问题。纯 CNN 有局部感受野限制，Transformer 虽能建模全局但计算开销和数据需求高。论文希望用全卷积、大核和倒置瓶颈结构，在保留 CNN 归纳偏置的同时扩大感受野。

### 2. 网络整体结构

整体仍是 U-shaped encoder-decoder。encoder 和 decoder 使用 CMUNeXt Block。下采样采用 max pooling，作者认为医学图像中低分辨率和边缘细微变化较多，pooling 可过滤部分噪声且计算开销低。decoder 使用 bilinear upsampling + convolution。skip 融合使用专门的 Skip-Fusion Block，而不是简单 concat。

### 3. 核心模块/改进点

- CMUNeXt Block：使用 large-kernel depthwise convolution 提取更大范围空间信息，再用两个 pointwise convolution 做通道混合。
- Inverted bottleneck：pointwise convolution 中间隐藏维度扩大为输入的 4 倍，增强通道表达和空间-通道混合。
- Skip-Fusion Block：encoder feature 和 decoder feature 分别经过普通卷积和 BN 后 concat，再通过 pointwise convolution 融合，目的是让 skip connection 更平滑。
- 轻量优势：相比大型 Transformer 或重型 U-Net，参数和 GFLOPs 更低，速度更快。

### 4. 与当前 CSWin-UNet 的关系

当前 CSWin-UNet 已经有 Transformer 全局/条带上下文，不缺全局建模主干。CMUNeXt 对当前项目更有价值的是两个局部模块：large-kernel depthwise convolution 可作为 decoder 局部上下文补充；Skip-Fusion Block 可替代简单 concat + linear，减少 encoder/decoder 特征融合突兀。

### 5. 对本课题的可借鉴点

- 频率增强：间接相关。大核 depthwise convolution 可扩大局部上下文，帮助判别某个高频响应是否处在合理病灶上下文中，但不是显式频域方法。
- 边界分支：帮助有限。没有显式 boundary head 或 boundary loss。
- 伪边界抑制：有一定帮助。大核上下文和 smoother skip fusion 可能减少局部纹理误判，但无法单独承担伪边界抑制。

### 6. 是否建议后续缝合

可作为辅助模块候选，不建议作为主创新。若后续发现 CSWin-UNet decoder 对局部纹理判断不足，可以在高分辨率 decoder feature 后加入轻量 large-kernel depthwise block；但优先级低于 PFESA、BEM/PMA、FEM 这类更贴合课题的模块。

## 7. Gradient-Guided Network With Fourier Enhancement for Glioma Segmentation in Multimodal 3D MRI

### 1. 论文要解决的问题

GFNet 面向 3D 多模态 MRI 胶质瘤分割，主要解决弱边界和局部感受野不足问题。作者认为普通 U-Net skip connection 难以保留边缘信息，CNN 特征局部性强，缺少全局信息；同时训练过程中难分像素往往集中在边界附近。因此提出 Dual-path Gradient-guided Training (DGT) 和 Fourier Edge-enhancement Module (FEM)。

### 2. 网络整体结构

整体以 U-shape 网络为 backbone。训练阶段采用 Siamese 式双路径，两条路径共享网络参数。第一条路径正常前向并计算 loss，从 encoder hidden feature 的梯度中得到重要性权重；第二条路径把这些梯度权重与特征结合，指导网络关注难分区域。推理阶段只用一条路径，不增加推理开销。FEM 被放在 skip connection 中，对多尺度 encoder feature 做 Fourier 边缘增强后再送入 decoder。

### 3. 核心模块/改进点

- DGT (Dual-path Gradient-guided Training)：根据第一分支 loss 对 encoder feature 的梯度，得到通道归一化的重要性权重。梯度大的区域通常是难分区域或弱边界区域。第二分支将梯度权重与 feature 相加或融合，引导训练关注困难区域。
- FEM (Fourier Edge-enhancement Module)：对 encoder multi-scale feature 做 FFT，分解 amplitude 和 phase。amplitude 中心区域视为低频，外围视为高频边缘。用 binary mask 或其他 high-pass filter 分离高频 amplitude，再用原 phase inverse FFT 重建边缘特征，最后与原特征相加。
- High-pass filter 消融：论文测试了 binary、Butterworth、Gaussian、Exponential、Bessel 等高通滤波器，说明高通滤波选择会影响效果。
- Loss：两条路径都使用 Dice + CE，总 loss 是两条路径 loss 之和。

### 4. 与当前 CSWin-UNet 的关系

FEM 与当前 CSWin-UNet 非常匹配，因为 CSWin-UNet 的 skip feature `x1/x2/x3` 正是多尺度 encoder feature，可以在 concat 之前做 Fourier edge enhancement。DGT 则是训练策略级改造，涉及二次前向/梯度提取，工程复杂度和显存成本较高，当前阶段需要谨慎。

### 5. 对本课题的可借鉴点

- 频率增强：高度相关。FEM 是最直接的 Fourier 高频边缘增强模块，可作为课题中的“频率增强”核心候选。
- 边界分支：间接相关。DGT 利用梯度关注弱边界/难分像素，但不产生显式 boundary prediction；可与独立边界头结合。
- 伪边界抑制：需要改造。原 FEM 主要增强高频，可能同时放大 NBI 血管纹理和反光伪边界。必须加入 boundary mask、结构注意力或背景抑制门控，不能单独使用原始 FEM。

### 6. 是否建议后续缝合

建议优先借鉴 FEM，但谨慎使用 DGT。第一阶段可将 FEM 改成 2D 版本，插入 CSWin-UNet 的高分辨率 skip 或 decoder feature；同时配合边界监督/伪边界抑制，避免全图高频放大。DGT 可作为后续训练策略增强，不建议和第一版结构改造同时上。

## 8. GobletNet: Wavelet-Based High-Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images

### 1. 论文要解决的问题

GobletNet 面向电子显微镜图像分割。作者认为 EM 图像高频纹理和轮廓非常丰富，但高频也包含大量噪声。现有模型没有充分利用 EM 图像本身的频率特性，因此提出基于 wavelet 的高频融合网络。

### 2. 网络整体结构

GobletNet 是双分支 encoder-decoder。第一支 semantic encoder 输入原图，提取语义信息；第二支 HF detail encoder 输入 wavelet transform 生成的高频图，提取细节和轮廓信息；fusion decoder 同时接收语义和高频细节特征生成分割结果。每个 encoder/decoder 层由残差块构成。每个对应层之间通过 Fusion-Attention Module (FAM) 融合两支特征。

### 3. 核心模块/改进点

- Wavelet HF image：对原图做 wavelet transform，得到低频 `LL`、水平高频 `HL`、垂直高频 `LH`、对角高频 `HH`。
- 高频输入构造：`H = HL + LH + HH + lambda * LL`。加入少量低频成分可降低纯高频噪声干扰。
- HF detail encoder：单独处理高频图，提取纹理和轮廓。
- FAM (Fusion-Attention Module)：concat semantic feature 和 detail feature，用 `1x1 conv` 得到融合特征，再生成两个 attention feature，分别调制语义分支和高频分支，实现语义-细节融合。
- 消融结论：FAM 加在浅层、中层、深层都有帮助，全部层加入效果最好；适当加入低频分量可降低噪声。

### 4. 与当前 CSWin-UNet 的关系

GobletNet 的完整双 encoder 结构迁移到 CSWin-UNet 会比较重，因为当前 CSWin-UNet 已经是 23M 参数的 Transformer U-Net。更实际的迁移方式是借鉴“wavelet 高频图作为辅助输入/辅助分支”和“FAM 融合语义与高频细节”的思想，而不是完整复制双分支网络。

### 5. 对本课题的可借鉴点

- 频率增强：高度相关。Wavelet 比 FFT 更保留空间位置，适合边界/纹理局部化；对胃镜图像可尝试生成高频边界先验。
- 边界分支：有帮助。HF detail branch 可视为一种高频边界分支，但需要加监督或门控才能让它关注病灶边界。
- 伪边界抑制：很有启发。论文明确指出纯高频会带噪声，加入低频 `LL` 可降低噪声干扰。这与 NBI 中高频血管/反光伪边界问题很契合。

### 6. 是否建议后续缝合

建议借鉴 wavelet 高频构造和 FAM 思想，但不建议完整双 encoder。更轻量的方案是在输入或浅层 feature 上生成 wavelet 高频图，把它作为边界/频率辅助分支，再用 FAM-like gate 与 CSWin-UNet 的 `x1` 或 decoder 高分辨率特征融合。`lambda * LL` 的低频补偿思想很适合用于伪边界抑制。

## 总表：模块筛选与后续缝合建议

### 1. 最推荐借鉴的模块

| 优先级 | 论文 | 模块 | 推荐原因 | 建议接入位置 |
| --- | --- | --- | --- | --- |
| 高 | PFESA | FFT 高频/低频解耦 + EA/SA 无参数注意力 | 同时服务频率增强和噪声抑制，参数少，适合医学小数据 | CSWin-UNet skip feature `x1/x2/x3` concat 前 |
| 高 | GFNet | Fourier Edge-enhancement Module (FEM) | 直接对应频率增强和弱边界增强 | 高分辨率 skip 或 decoder feature |
| 高 | Rethinking Boundary Detection | BEM + boundary loss + background path | 显式边界分支和背景抑制思路清晰 | decoder 高分辨率层、边界辅助头 |
| 高 | LB-UNet | RBP + PIF | 区域/边界预测反向引导 skip fusion | skip fusion 替代直接 concat |
| 中 | GobletNet | Wavelet HF image + FAM | 高频细节分支和低频降噪思想有用 | 输入辅助分支或浅层 feature 融合 |
| 中 | CMUNeXt | Large-kernel depthwise block + Skip-Fusion | 轻量补充局部-全局上下文 | decoder 高分辨率层或 skip fusion |
| 低 | TA-MoSC | MoE skip routing | 思想有价值但复杂度较高 | 后续高级版 skip 自适应融合 |
| 基底 | CSWin-UNet | CSWin block + CARAFE | 当前主干和 baseline | 保持主体稳定 |

### 2. 可作为消融实验的模块组合

建议按从简单到复杂的顺序设计消融，避免一次加入太多模块导致贡献不清。

| 实验编号 | 组合 | 目的 |
| --- | --- | --- |
| A0 | CSWin-UNet baseline | 当前基线 |
| A1 | baseline + 边界辅助头 / boundary loss | 验证显式边界监督是否提升 Boundary IoU 和 HD95 |
| A2 | baseline + PFESA skip refinement | 验证频域高低频注意力是否改善边界和噪声 |
| A3 | baseline + FEM 高频增强 | 验证 Fourier 高频边缘增强是否有效 |
| A4 | baseline + 边界辅助头 + PFESA | 验证“边界监督 + 频域注意力”的互补性 |
| A5 | baseline + 边界辅助头 + PFESA/FEM + 背景/伪边界抑制 gate | 对应完整课题“频率增强 + 边界分支 + 伪边界抑制” |
| A6 | baseline + wavelet 高频辅助分支 + FAM-like fusion | 作为频率增强的替代方案，与 FFT 路线对比 |

### 3. 不建议直接使用的复杂/不匹配模块

- 完整 TA-MoSC / MoE skip routing：会引入 router、expert bank、Top-K、专家均衡 loss 和两阶段训练，变量过多，不适合作为第一版课题主线。
- 完整 CTO 双流 encoder：当前 CSWin-UNet 已经是 Transformer U-shaped 主干，再加入 CNN + StitchViT 双流会偏离“在 CSWin-UNet 上缝合模块”的路线。
- LB-UNet 的 GA boundary keypoint generator：对当前二值 mask 可用形态学边界生成监督，GA 关键点复杂且收益不确定。
- 完整 GobletNet 双 encoder：电子显微镜图像和胃镜图像频率特性不同，完整双分支较重；建议只借鉴 wavelet 高频输入和 FAM。
- 单独使用 GFNet FEM 不加抑制：可能放大 NBI 血管纹理、反光点和褶皱边缘，必须配合边界监督或伪边界抑制。

### 4. 与当前数据集痛点的对应关系

| 数据集痛点 | 证据/现象 | 对应模块方向 |
| --- | --- | --- |
| 边界弱、局部灰度差低 | WL/NBI 的 boundary gray diff 中位数都约为 1.2 | 边界辅助头、BEM、RBP、boundary loss |
| WL 低对比和平滑过渡 | WL 清晰度和颜色差低于 NBI，高风险样本边界软 | 低频结构注意力、边界监督、上下文 gate |
| NBI 高频纹理强 | NBI Laplacian 清晰度明显高，血管/纹理突出 | PFESA、FEM、wavelet HF branch |
| NBI 反光伪边界多 | NBI 反光干扰比例高于 WL | SNR 注意力、背景 path、低频补偿、伪边界抑制 gate |
| 直接 skip 可能传递噪声 | CSWin-UNet 当前 skip 是 concat + linear | PFESA skip refinement、PIF、Skip-Fusion |

### 5. 当前最合理的研究主线

从这 8 篇论文看，最稳妥的方向不是替换 CSWin-UNet 主干，而是在其 decoder/skip 部分做一个轻量、可解释、可消融的边界频率模块：

1. 用 PFESA 或 FEM 在 skip/high-resolution decoder feature 上提取真实边界相关高频。
2. 用 RBP/BEM 思路增加显式边界分支和 boundary loss。
3. 用 SA、background path 或 `lambda * LL` 低频补偿思想抑制纹理、反光、褶皱等伪边界。
4. 保持 CSWin block 和 CARAFE 不变，这样论文叙述可写成“以 CSWin-UNet 为上下文基底，提出面向胃镜弱边界和伪边界干扰的边界频率增强模块”。
