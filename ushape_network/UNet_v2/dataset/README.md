# 胃镜病灶分割数据集诊断

本目录包含胃镜病灶分割数据集的两个成像模态：`WL` 与 `NBI`。每个模态包含 2800 张图像及其对应分割 mask，共 5600 对样本。当前已按 `70% / 15% / 15%` 划分为 `train`、`val`、`test`。

本次诊断时间：2026-04-29。诊断目标不是比较模型指标，而是先回答：这个数据集真正困难在哪里，后续论文方法应该围绕什么问题展开。

## 当前目录结构

```text
dataset/
  WL/
    train/
      images/  1960 张 jpg
      masks/   1960 张 png
    val/
      images/   420 张 jpg
      masks/    420 张 png
    test/
      images/   420 张 jpg
      masks/    420 张 png
  NBI/
    train/
      images/  1960 张 jpg
      masks/   1960 张 png
    val/
      images/   420 张 jpg
      masks/    420 张 png
    test/
      images/   420 张 jpg
      masks/    420 张 png
  scripts/
    dataset_diagnosis.py
    split_dataset.py
  analysis_outputs/
    dataset_metrics.csv
    summary.md
    split_manifest.csv
    visual_samples.csv
    visual_check/
```

说明：划分后的 `train/val/test` 目录已经完整可用，训练和评估时应使用这些目录。

数据完整性复核结果：

| 模态 | images | masks | 图像格式 | mask 格式 | 文件名配对缺失 | 图像/mask 尺寸不一致 |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| WL | 2800 | 2800 | `.jpg` | `.png` | 0 | 0 |
| NBI | 2800 | 2800 | `.jpg` | `.png` | 0 | 0 |

图像和 mask 通过相同 stem 配对，例如 `gas_xxx.jpg` 对应 `gas_xxx.png`。mask 为二值标注，像素值为 `0` 和 `255`，未发现空 mask。

## 可复现诊断

运行：

```bash
python scripts/dataset_diagnosis.py
```

输出：

- `analysis_outputs/dataset_metrics.csv`：逐图统计指标。
- `analysis_outputs/summary.md`：WL/NBI 汇总统计。
- `analysis_outputs/visual_samples.csv`：容易、困难、疑似失败样本清单。
- `analysis_outputs/visual_check/WL_easy_30.jpg`
- `analysis_outputs/visual_check/WL_difficult_30.jpg`
- `analysis_outputs/visual_check/WL_likely_failure_30.jpg`
- `analysis_outputs/visual_check/NBI_easy_30.jpg`
- `analysis_outputs/visual_check/NBI_difficult_30.jpg`
- `analysis_outputs/visual_check/NBI_likely_failure_30.jpg`

说明：当前目录没有 U-Net 或 Swin-Unet 的预测 mask，因此这里的 `likely_failure` 是按数据难度指标筛出的“疑似失败/高风险样本”，不是模型真实失败结果。

## 统计体检

| 指标 | WL | NBI | 初步解释 |
| --- | ---: | ---: | --- |
| 样本数 | 2800 | 2800 | 两个模态数量均衡 |
| mask 前景占比均值 | 31.10% | 37.73% | NBI 病灶区域整体更大 |
| mask 前景占比中位数 | 26.58% | 33.64% | NBI 中位病灶面积也更大 |
| 图像亮度均值 | 96.46 | 92.38 | NBI 整体略暗 |
| 图像对比度均值 | 45.33 | 50.38 | NBI 全局对比度更强 |
| 病灶-背景 RGB 距离均值 | 68.28 | 76.31 | NBI 病灶与背景颜色差异更明显 |
| 局部边界灰度差中位数 | 1.19 | 1.23 | 两个模态的局部边界都很弱 |
| 反光占比均值 | 0.82% | 1.56% | NBI 反光干扰更明显 |
| Laplacian 清晰度均值 | 219.73 | 778.46 | NBI 纹理/血管结构更锐利，WL 更容易模糊或平滑 |
| 边界复杂度均值 | 10.01 | 9.78 | 两个模态边界复杂度接近，WL 略高 |

病灶大小分布：

| 模态 | 小病灶 <1% | 中等病灶 1%-10% | 大病灶 >=10% |
| --- | ---: | ---: | ---: |
| WL | 3 | 473 | 2324 |
| NBI | 0 | 262 | 2538 |

关键判断：

- 数据集不是以极小病灶为主，大多数 mask 前景占比超过 10%。
- WL 的中等病灶更多，且整体对比度、病灶-背景颜色差、清晰度都低于 NBI；WL 更需要关注正常黏膜与病灶区域混淆。
- NBI 的纹理和血管结构更清晰，病灶与背景差异更明显，但反光占比更高，容易引入高亮干扰。
- 两个模态的局部边界灰度差中位数都只有约 1.2，说明主困难点不是全局颜色差，而是病灶边缘附近过渡弱、边界模糊。
- 当前文件名没有患者 ID 或病例号，无法统计“每个患者的图片数分布”。如果后续能拿到原始映射表，应优先补做患者级统计和患者级划分。

## 可视化体检

红色区域为 mask 覆盖区域，黄色线为 mask 边界。每个模态各筛出 30 张容易样本、30 张困难样本、30 张疑似失败样本。

| 模态 | 容易样本 | 困难样本 | 疑似失败/高风险样本 |
| --- | --- | --- | --- |
| WL | `analysis_outputs/visual_check/WL_easy_30.jpg` | `analysis_outputs/visual_check/WL_difficult_30.jpg` | `analysis_outputs/visual_check/WL_likely_failure_30.jpg` |
| NBI | `analysis_outputs/visual_check/NBI_easy_30.jpg` | `analysis_outputs/visual_check/NBI_difficult_30.jpg` | `analysis_outputs/visual_check/NBI_likely_failure_30.jpg` |

WL 疑似失败样本：

![WL likely failure](analysis_outputs/visual_check/WL_likely_failure_30.jpg)

NBI 疑似失败样本：

![NBI likely failure](analysis_outputs/visual_check/NBI_likely_failure_30.jpg)

可视化观察：

- WL 高风险样本中，正常黏膜与病灶区域颜色接近，边界处经常是平滑过渡；部分图像还存在阴影、模糊、褶皱和大面积红色黏膜背景。
- NBI 高风险样本中，血管/纹理确实更突出，但反光和暗背景更常见；部分病灶边界仍然依赖很窄的局部灰度变化。
- 两个模态的疑似失败样本都不是单纯“小目标困难”，而是更集中在“边界弱 + 低对比 + 背景干扰”的组合问题。

## 失败风险类型

基于 mask 和图像统计，当前高风险样本可先分为以下几类：

| 风险类型 | WL | NBI | 说明 |
| --- | ---: | ---: | --- |
| 低对比度/边界模糊型 | 2256 | 2249 | 主类型，说明边界弱是跨模态共性问题 |
| 反光干扰型 | 290 | 556 | NBI 更明显 |
| 模糊成像型 | 83 | 0 | WL 更明显 |
| 小目标漏检型 | 3 | 0 | 数量少，但对 Recall/Dice 可能影响明显 |
| 形状不规则型 | 34 | 10 | 数量不多，适合作为边界鲁棒性案例展示 |

这些类别不是模型预测失败类别，而是“数据难度风险类型”。真正的失败案例分析需要加入模型预测结果，例如：

```text
predictions/
  unet/
    WL/*.png
    NBI/*.png
  swin_unet/
    WL/*.png
    NBI/*.png
```

有预测后，应继续统计：

- 预测偏大：pred mask 明显覆盖到正常黏膜。
- 预测偏小：病灶内部或边缘被漏掉。
- 边界错误：Dice 可能尚可，但 Boundary IoU、HD95 或 ASSD 差。
- 完全漏检：尤其关注小病灶、低对比、弱边界样本。
- U-Net 和 Swin-Unet 是否在同一批图上失败：如果重叠度高，说明困难主要来自数据本身；如果失败类型不同，才更适合做结构改进。

## 问题-方法映射

主痛点：病灶边界模糊、局部边界对比弱。

可对应的方法方向：

- 引入边界监督，例如 boundary loss、distance transform loss、Dice + BCE + boundary loss。
- 增加边界分支或边缘辅助任务，让模型显式学习病灶轮廓。
- 在评价指标中加入 Boundary IoU、HD95 或 ASSD，不只报告 Dice/IoU。
- 对低边界差、低对比样本做 hard example mining 或难例重加权。

辅痛点：WL 更容易低对比度/模糊，NBI 更容易反光干扰。

可对应的方法方向：

- 做模态感知增强：WL 加强颜色扰动、局部对比度增强、轻微模糊鲁棒训练；NBI 加入高亮反光模拟和遮挡鲁棒性增强。
- 做 WL/NBI 对比分析：证明 NBI 在颜色差异和清晰度上更有优势，而 WL 更依赖上下文和边界建模。
- 如果论文需要一个更聚焦的创新点，可以围绕“边界感知 + 模态自适应增强”展开，而不是盲目堆叠更复杂主干网络。

## 数据集划分

运行：

```bash
python scripts/split_dataset.py
```

划分策略：

- 比例：train 70%，val 15%，test 15%。
- 随机种子：`20260429`。
- 分层依据：按 mask 前景占比分为小病灶、中等病灶、大病灶，尽量保持各集合比例接近。
- 划分清单：`analysis_outputs/split_manifest.csv`。

划分结果：

| 模态 | train | val | test | 总数 |
| --- | ---: | ---: | ---: | ---: |
| WL | 1960 | 420 | 420 | 2800 |
| NBI | 1960 | 420 | 420 | 2800 |

各子集 image/mask 配对复核：

| 模态 | 集合 | images | masks | 缺失 mask | 缺失 image |
| --- | --- | ---: | ---: | ---: | ---: |
| WL | train | 1960 | 1960 | 0 | 0 |
| WL | val | 420 | 420 | 0 | 0 |
| WL | test | 420 | 420 | 0 | 0 |
| NBI | train | 1960 | 1960 | 0 | 0 |
| NBI | val | 420 | 420 | 0 | 0 |
| NBI | test | 420 | 420 | 0 | 0 |

当前文件名没有患者 ID 或病例号，因此本次无法做患者级划分。如果能恢复患者 ID，应优先改为患者级 train/val/test，避免同一患者图像同时进入训练集和测试集。

## 训练与验证指标约定

后续无论使用 U-Net v2，还是替换为其他网络结构，训练验证和最终测试都建议统一报告以下指标：

| 指标 | 趋势 | 用途 |
| --- | --- | --- |
| Dice | 越高越好 | 衡量预测区域与真实 mask 的整体重叠程度。 |
| IoU | 越高越好 | 衡量预测区域与真实 mask 的交并比，可与 Dice 互相补充。 |
| Boundary IoU | 越高越好 | 衡量预测边界带与真实边界带的重合程度，重点反映边界质量。 |
| HD95 | 越低越好 | 95% Hausdorff Distance，单位为像素，用于衡量边界距离误差，能暴露局部边界偏移或边界糊的问题。 |
| MAE | 越低越好 | 衡量二值预测 mask 与真实 mask 的平均像素误差。 |

本数据集的主痛点是“弱边界、低对比、背景干扰”，不是单纯区域覆盖问题。因此模型对比时不要只看 Dice/IoU，应同时查看 `Boundary IoU` 和 `HD95`。如果 Dice 提升但 Boundary IoU 没有提升，或 HD95 变差，说明模型可能只是扩大/缩小了区域覆盖，并没有真正改善病灶边界定位。

当前 U-Net v2 代码中，训练验证阶段和测试阶段均已使用同一套指标：

```text
Dice / IoU / Boundary IoU / HD95 / MAE
```

## 实验终端输出与记录约定

后续无论使用 U-Net v2、Swin-Unet、普通 U-Net，还是替换为其他网络结构，训练和测试脚本都应尽量遵守同一套终端输出与本地记录规范。这样 WL、NBI 以及不同模型之间的结果才能直接横向比较。

训练启动阶段应先输出数据完整性表，至少包含以下字段：

```text
modal / split / images / masks / missing_masks / missing_images / bad_size / bad_values
```

随后输出本次实验的基础信息：

- 运行设备：例如 `Using device: cuda` 或 `Using device: cpu`。
- 模型名称：例如 `UNetV2`、`SwinUnet`、`UNet`。
- 实验名称：用于区分模态、epoch、batch size、输入尺寸、学习率、优化器、是否增强等设置。
- 输出目录：本次实验所有 checkpoint、CSV 和预测结果的保存位置。
- 优化器信息：至少能看出优化器类型、学习率和 weight decay。
- 训练开始标记：`#################### Start Training ####################`。

每个 epoch 训练阶段应使用 `tqdm` 进度条显示当前 epoch、step 进度、实时学习率和实时 loss，例如：

```text
Epoch 001/100: 100%|████████████████| 123/123 [03:20<00:00,  1.63s/it, lr=1.00e-04, loss=1.2345]
```

每个 epoch 结束后必须分别输出 `TRAIN` 和 `VAL` 汇总行。两行格式应保持一致，并包含：

```text
epoch / dataset / dice / iou / b_iou / hd95 / mae / lr / loss
```

其中 `b_iou` 是终端显示中的缩写，对应 CSV 和论文表格中的 `Boundary IoU` 或 `boundary_iou`。推荐格式如下：

```text
TRAIN | epoch=001/100 | dataset=WL       | dice= 0.7123 | iou= 0.5542 | b_iou= 0.3381 | hd95=  21.4827 | mae= 0.0921 | lr= 1.00e-04 | loss=  1.2345
VAL   | epoch=001/100 | dataset=WL       | dice= 0.7356 | iou= 0.5812 | b_iou= 0.3510 | hd95=  18.9341 | mae= 0.0842 | lr= 1.00e-04 | loss=  0.9876
BEST  | epoch=001/100 best_dice=0.7356
```

当当前 epoch 的验证集 Dice 超过历史最好结果时，应输出 `BEST` 行，并保存最优 checkpoint。

本地记录文件建议统一保存为以下结构：

```text
outputs/
  <ModelName>/
    <WL or NBI>/
      config.json
      history.csv
      metrics_val.csv
      checkpoints/
        best.pth
        latest.pth
      predictions/
        test/
          *.png
      metrics_test.csv
      metrics_test_per_image.csv
```

各文件含义如下：

- `config.json`：保存本次实验参数、模型名、设备、预训练权重路径、输入尺寸、batch size、学习率、优化器、指标列表等信息。
- `history.csv`：按 epoch 保存 `train_loss`、`val_loss`、train/val 的 Dice、IoU、Boundary IoU、HD95、MAE、当前 best Dice，以及该 epoch 是否为最优。
- `metrics_val.csv`：保存验证集逐 epoch 汇总结果，便于快速画曲线或筛选最优 epoch。
- `checkpoints/best.pth`：验证集 Dice 最优的模型权重。
- `checkpoints/latest.pth`：最后一次保存的模型权重，便于中断后恢复或排查。
- `metrics_test.csv`：最终测试集汇总指标。
- `metrics_test_per_image.csv`：测试集逐图指标，用于分析失败样本。
- `predictions/test/*.png`：测试集预测 mask，文件名应与原图 stem 对齐。

默认不要求使用 wandb 或其他在线实验平台；即使使用在线记录，也必须保留上述本地文件，保证实验离线可复现。
