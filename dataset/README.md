# 胃镜病灶分割数据集诊断

本目录包含胃镜病灶分割数据集的两个成像模态：`WL` 与 `NBI`。当前版本保留为未划分的原始配对结构，每个模态包含 2800 张图像及其对应分割 mask，共 5600 对样本。

本次诊断时间：2026-04-29。诊断目标不是比较模型指标，而是先回答：这个数据集真正困难在哪里，后续论文方法应该围绕什么问题展开。

## 当前目录结构

```text
dataset/
  WL/
    images/  2800 张 jpg
    masks/   2800 张 png
  NBI/
    images/  2800 张 jpg
    masks/   2800 张 png
  scripts/
    dataset_diagnosis.py
  analysis_outputs/
    dataset_metrics.csv
    summary.md
    visual_samples.csv
    visual_check/
```

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

## 建议的数据集划分

当前目录尚未划分。若需要复现实验，建议生成独立划分清单，而不是直接移动原始文件。

推荐策略：

- 比例：train 70%，val 15%，test 15%。
- 随机种子固定，例如 `20260429`。
- 分层依据：按 mask 前景占比分为小病灶、中等病灶、大病灶，尽量保持各集合比例接近。
- 输出清单建议保存为 `analysis_outputs/split_manifest.csv`，字段至少包含 `modality`、`image_path`、`mask_path`、`split`、`foreground_ratio`、`size_bucket`。

按当前样本数，70/15/15 划分后每个模态应为：

| 模态 | train | val | test | 总数 |
| --- | ---: | ---: | ---: | ---: |
| WL | 1960 | 420 | 420 | 2800 |
| NBI | 1960 | 420 | 420 | 2800 |

如果能恢复患者 ID，应优先做患者级 train/val/test，避免同一患者图像同时进入训练集和测试集。

