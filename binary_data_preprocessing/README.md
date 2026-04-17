# 胃部 NBI / WL 图像二分类项目说明

## 1. 项目目标

使用 `origin_data` 中已经人工分好的胃部图像，训练一个二分类模型，将 `pending` 文件夹中混合在一起的图像自动分为：

- `NBI`
- `WL`

## 2. 当前目录结构

```text
.
|-- origin_data/
|   |-- NBI/                    # 已标注 NBI 图像，500 张
|   `-- WL/                     # 已标注 WL 图像，500 张
|-- pending/                    # 待分类图像，当前共 9627 张
|-- prepared_data/              # 当前使用的训练/验证集
|   |-- train/
|   |   |-- NBI/
|   |   `-- WL/
|   |-- val/
|   |   |-- NBI/
|   |   `-- WL/
|   `-- split_manifest.csv      # 当前拆分清单
|-- sorted_output_t090/         # 第一次阈值 0.90 的推理结果
|   |-- NBI/
|   |-- WL/
|   `-- uncertain/
|-- artifacts/
|   `-- resnet18/
|       |-- checkpoints/
|       |-- reports/
|       `-- inference_t090/     # 第一次阈值 0.90 的推理报告
|-- .torch/                     # 预训练权重缓存
|-- prepare_dataset.py          # 数据准备脚本
|-- train_resnet18.py           # ResNet18 训练脚本
|-- predict_pending.py          # pending 批量推理脚本
`-- README.md
```

## 3. 总体流程

### Step 1. 规范目录结构

- 保留原始数据目录 `origin_data/NBI` 和 `origin_data/WL`
- 保留待分类目录 `pending`
- 推理结果不再固定写入同一个目录，而是按阈值自动生成新目录
- 例如：
  - `sorted_output_t090`
  - `sorted_output_t080`
  - `sorted_output_t075`

状态：已完成

### Step 2. 划分训练集和验证集

- 从 `origin_data/NBI` 中划分训练集和验证集
- 从 `origin_data/WL` 中划分训练集和验证集
- 当前采用 `8:2` 拆分
- 当前使用的随机种子：`20260416`
- 当前拆分结果：
  - `train/NBI`: 400 张
  - `val/NBI`: 100 张
  - `train/WL`: 400 张
  - `val/WL`: 100 张
- 输出拆分清单 `prepared_data/split_manifest.csv`
- `prepare_dataset.py` 支持 `--clean`，可安全重建拆分目录

状态：已完成

### Step 3. 选择基础模型

- 已确定首版基础模型为 `ResNet18`
- 采用 `ImageNet` 预训练权重，在当前任务上做迁移学习
- 最后一层分类头替换为 2 类输出：
  - `NBI`
  - `WL`

选择 `ResNet18` 的原因：

- 模型结构成熟稳定，适合先建立可靠 baseline
- 相比更大的模型更不容易在 1000 张标注数据上过拟合
- 训练和推理速度较快，便于快速迭代
- 在迁移学习场景中实现简单，后续便于与其他模型对比

后续如果效果不理想，可再对比：

- `EfficientNet-B0`
- `DenseNet121`

状态：已完成

### Step 4. 编写训练脚本

- 已编写 `train_resnet18.py`
- 读取 `prepared_data/train` 和 `prepared_data/val`
- 定义训练增强和验证预处理
- 加载 `ImageNet` 预训练 `ResNet18` 并替换最后分类层为 2 类
- 执行训练并在每个 epoch 后进行验证
- 保存以下产物到 `artifacts/resnet18/`：
  - `checkpoints/best_model.pt`
  - `checkpoints/last_model.pt`
  - `reports/history.csv`
  - `reports/confusion_matrix.csv`
  - `reports/summary.json`

脚本特性：

- 支持自定义 `epochs`、`batch_size`、`lr`、`image_size`
- 支持 `--freeze-backbone` 和 `--full-finetune`
- 在 `CPU` 环境下默认使用更合适的参数：
  - `epochs=8`
  - `batch_size=16`
  - 默认冻结 backbone
- 自动选择 `cuda` 或 `cpu`
- 在终端中实时打印每个 epoch 的训练信息
- 输出验证集准确率、混淆矩阵和分类报告

状态：已完成

### Step 5. 验证模型效果

- 已完成当前拆分版本的一轮训练与验证
- 本次环境使用 `CPU`
- 本次训练配置：
  - `epochs=8`
  - `batch_size=16`
  - `freeze_backbone=true`
- 训练样本数：800
- 验证样本数：200
- 最佳验证集准确率：`0.975`
- 最佳 epoch：`8`
- 验证集混淆矩阵：
  - 实际 `NBI` 预测为 `NBI`: 99
  - 实际 `NBI` 预测为 `WL`: 1
  - 实际 `WL` 预测为 `NBI`: 4
  - 实际 `WL` 预测为 `WL`: 96
- 当前验证集分类报告：
  - `NBI` precision: `0.9612`
  - `NBI` recall: `0.9900`
  - `WL` precision: `0.9897`
  - `WL` recall: `0.9600`

验证结果文件：

- `artifacts/resnet18/checkpoints/best_model.pt`
- `artifacts/resnet18/checkpoints/last_model.pt`
- `artifacts/resnet18/reports/history.csv`
- `artifacts/resnet18/reports/confusion_matrix.csv`
- `artifacts/resnet18/reports/summary.json`

说明：

- 这次 `97.5%` 的结果比之前旧拆分下的 `100%` 更可信
- 当前模型已经具备可用性，但进入 `pending` 批量推理时，仍建议保留低置信度分流到 `uncertain`
- 如果后续实际分拣效果不如验证集表现，可以进一步追加 `test` 集，或按病例/序列级别重新划分数据

状态：已完成

### Step 6. 编写批量推理脚本

- 已编写 `predict_pending.py`
- 读取 `pending` 中全部图像
- 加载 `artifacts/resnet18/checkpoints/best_model.pt`
- 对每张图输出：
  - 预测类别
  - 预测置信度
  - 最终输出目录标签
- 如果不手动指定输出目录，脚本会按阈值自动生成目录名
- 当前第一次推理使用的置信度阈值：`0.9`
- 当前第一次推理报告目录：
  - `artifacts/resnet18/inference_t090`

状态：已完成

### Step 7. 输出分类结果

- 已完成一次 `pending` 全量推理与复制分流
- `pending` 总图像数：`9627`
- 第一次推理结果位于：
  - `sorted_output_t090/NBI`
  - `sorted_output_t090/WL`
  - `sorted_output_t090/uncertain`
- 第一次推理结果统计：
  - `NBI`: `333`
  - `WL`: `1524`
  - `uncertain`: `7770`
- 当前结果说明：
  - `0.9` 阈值较保守，因此 `uncertain` 比例较高
  - 人工抽查显示，当前高置信度输出的 `NBI` 和 `WL` 精度很好
  - 后续可以尝试降低阈值重新推理，例如 `0.8`

状态：已完成

### Step 8. 人工抽样复核

- 从高置信度输出中抽样检查一部分图片
- 重点确认：
  - 是否存在大批量误分
  - 是否某一类系统性偏多或偏少
- 重点检查 `uncertain` 中的边界样本

状态：进行中

### Step 9. 迭代优化

- 将人工确认正确的新样本补充回训练集
- 重新训练模型
- 对 `pending` 结果再次推理
- 持续提高分类稳定性

状态：待完成

## 4. 已完成内容

- 已创建训练和推理所需目录
- 已使用 `prepare_dataset.py` 完成 `8:2` 数据拆分
- 已重新生成一版新的拆分用于当前实验
- 已确定首版基础模型为 `ResNet18`
- 已完成 `train_resnet18.py` 训练脚本
- 已完成当前版本的 `ResNet18` 训练与验证
- 已修复训练脚本中 `summary.json` 的导出问题
- 已将训练脚本调整为更适合 `CPU` 的默认参数
- 已完成 `predict_pending.py` 批量推理脚本
- 已将推理脚本改为按阈值自动生成新目录
- 已将第一次 `0.9` 的推理结果重命名为阈值命名版本

## 5. 使用方式

训练模型可执行：

```powershell
$env:TORCH_HOME=(Resolve-Path '.').Path + '\.torch'
python train_resnet18.py
```

如果想强制进行完整微调：

```powershell
$env:TORCH_HOME=(Resolve-Path '.').Path + '\.torch'
python train_resnet18.py --full-finetune
```

如果想显式只训练最后分类头：

```powershell
$env:TORCH_HOME=(Resolve-Path '.').Path + '\.torch'
python train_resnet18.py --freeze-backbone
```

如果想调整参数，例如训练 12 个 epoch：

```powershell
$env:TORCH_HOME=(Resolve-Path '.').Path + '\.torch'
python train_resnet18.py --epochs 12 --batch-size 16
```

批量推理 `pending` 可执行：

```powershell
python predict_pending.py
```

说明：

- 默认会根据阈值自动生成新目录
- 例如 `--threshold 0.8` 会生成类似 `sorted_output_t080` 和 `artifacts/resnet18/inference_t080`
- 如果同名目录已存在，脚本会自动追加序号，避免覆盖旧结果

如果想调整阈值，例如改为 `0.8`：

```powershell
python predict_pending.py --threshold 0.8
```

如果想手动指定输出目录，也可以显式传参：

```powershell
python predict_pending.py --threshold 0.8 --output-dir my_output --report-dir artifacts/resnet18/my_report
```

## 6. 重要约束

- 不修改 `origin_data` 中的任何原始文件
- 不修改 `pending` 中的任何原始文件
- 对这两个目录只允许读取或复制，不允许移动、重命名、删除或覆盖

## 7. 下一步建议

下一步继续推进 Step 8：

- 对 `sorted_output_t090/NBI` 和 `sorted_output_t090/WL` 继续抽样人工复核
- 重点检查 `sorted_output_t090/uncertain` 中的边界样本
- 如果高置信度精度持续稳定，可以尝试运行：

```powershell
python predict_pending.py --threshold 0.8
```

