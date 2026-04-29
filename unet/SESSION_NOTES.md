# SESSION NOTES

## 1. 本次对话目标概览

本次工作围绕胃部医学图像分割 baseline 的完整搭建与验证展开，核心目标包括：

- 确认数据集目录结构与数据划分是否正确
- 为当前项目编写和重组 `README.md`
- 搭建 baseline 工程结构
- 实现数据读取、预处理增强、标准 `U-Net`
- 实现训练脚本与测试脚本
- 在本机和服务器环境上验证可运行性
- 记录 `WL only` baseline 的训练情况
- 明确后续 `NBI only` 的运行方式

## 2. 数据集与目录相关结论

### 2.1 数据目录

当前项目目录为：

- `F:\code\git\segmentation\baseline`

原始数据位于：

- `sorted_075/`

结构为：

- `sorted_075/WL/{train,val,test}/{images,masks}`
- `sorted_075/NBI/{train,val,test}/{images,masks}`

### 2.2 数据划分

已确认每种模态各 `2000` 张图像，划分如下：

- `train = 1400`
- `val = 300`
- `test = 300`

### 2.3 图像与标签配对

已检查并确认：

- `images` 和 `masks` 数量一致
- 图像与 mask 文件名一一对应
- 原图尺寸与对应 mask 尺寸一致

### 2.4 mask 编码检查结论

对大量 `masks/*.png` 做了检查，结论是：

- mask 不是 `0/1`
- 也不是 `0/255`
- 实际像素值为 `0/2`
- `0` 表示背景
- `2` 表示前景

这也解释了为什么直接打开时看起来几乎全黑：前景像素值只有 `2`，肉眼不明显。

训练代码中已在读取时动态映射为：

- 背景：`0`
- 前景：`1`

### 2.5 关于 `sorted_075`

已明确约束：

- `sorted_075` 下所有文件都不能改动
- 后续所有处理都必须只读访问
- mask 的 `0/2 -> 0/1` 转换只在内存中完成，不回写原文件

## 3. README 与文档整理

### 3.1 README 工作

先后完成了：

- 初始英文版 `README.md`
- 改写为中文版
- 重新组织为“项目执行路线图”风格
- 补入完整工程目录结构
- 补入后续实验步骤说明

### 3.2 outputs/note.md

建立并维护了：

- `outputs/note.md`

当前该文件用于记录每次 baseline 训练的精简结果，最终保留：

- 实验名称
- 最优验证结果
- 训练过程简要观察

## 4. 工程结构搭建情况

已搭建的工程结构包括：

- `configs/`
- `datasets/`
- `models/`
- `utils/`
- `outputs/`
- `scripts/`
- `train.py`
- `test.py`
- `requirements.txt`

各目录职责：

- `sorted_075/`：原始数据，只读
- `configs/`：预留配置文件
- `datasets/`：数据读取、mask 映射、预处理与增强
- `models/`：模型定义
- `utils/`：训练辅助工具
- `outputs/`：训练结果、日志、模型权重
- `scripts/`：预留辅助脚本

## 5. 已实现的代码模块

### 5.1 数据集模块

实现文件：

- `datasets/gastric_segmentation_dataset.py`

已实现功能：

- 从 `sorted_075/<modality>/<split>/` 自动读取数据
- 根据文件名自动配对图像和 mask
- 检查配对完整性
- 检查图像与 mask 尺寸是否一致
- 图像转为 `torch.float32` tensor
- mask 从原始 `0/2` 动态映射成训练使用的 `0/1`
- 支持 `WL` / `NBI`
- 支持 `train / val / test`

### 5.2 预处理与增强模块

实现文件：

- `datasets/transforms.py`

已实现内容：

- `ResizePair`
- `RandomHorizontalFlipPair`
- `RandomVerticalFlipPair`
- `RandomRotatePair`
- `RandomColorJitterPair`
- `NormalizeTensor`
- `build_transforms`

设计原则：

- 图像和 mask 同步做空间变换
- mask 使用最近邻插值，避免标签污染
- 训练集可开启增强
- 验证集和测试集不启用随机增强

### 5.3 模型模块

实现文件：

- `models/unet.py`

实现内容：

- 标准 `U-Net`
- `DoubleConv`
- `DownBlock`
- `UpBlock`
- `OutputBlock`
- `UNet`
- `build_unet`

已做前向验证：

- 输入 `(1, 3, 512, 512)`
- 输出 `(1, 1, 512, 512)`

### 5.4 工具模块

实现文件：

- `utils/common.py`
- `utils/losses.py`
- `utils/metrics.py`

实现内容：

- 目录创建
- JSON / CSV 保存
- 随机种子固定
- `DiceLoss`
- `BCEDiceLoss`
- `Dice / IoU / Precision / Recall`

### 5.5 训练脚本

实现文件：

- `train.py`

当前功能：

- 单模态训练：`WL` 或 `NBI`
- 训练集与验证集 DataLoader
- 标准 `U-Net` 训练
- `BCE + Dice` 联合损失
- 训练与验证指标统计
- 保存 `latest.pt` 与 `best.pt`
- 导出 `config.json`、`history.csv`、`best_metrics.json`、`summary.json`
- 支持 `--device cuda`
- 支持 `--amp`
- 支持 `--dry-run`

### 5.6 测试脚本

实现文件：

- `test.py`

当前功能：

- 加载训练好的 checkpoint
- 在 test 集上推理
- 计算 `Dice / IoU / Precision / Recall`
- 导出 `test_metrics.json`
- 可选保存预测 mask
- 支持 `--device cuda`
- 支持 `--amp`

## 6. 兼容性与环境问题处理

### 6.1 Python 版本兼容

曾遇到问题：

- `test` conda 环境是 `Python 3.8`
- 代码里使用了部分 `3.10+` 类型注解

已处理：

- 改写为兼容 `Python 3.8+` 的类型注解

### 6.2 numpy 报错

服务器训练时出现：

- `RuntimeError: Numpy is not available`

判断与建议：

- 常见原因是没装 `numpy`
- 或者装了与当前 `torch` 不兼容的 `numpy 2.x`

推荐修复方式：

```bash
pip uninstall -y numpy
pip install numpy==1.26.4
```

## 7. 本机与服务器环境验证

### 7.1 本机 `test` 环境

已验证：

- `Python 3.8.20`
- `torch 2.4.1+cpu`
- CPU 版本可跑通 baseline

### 7.2 服务器 GPU 环境建议

推荐服务器基础环境选择：

- `Miniconda`
- `Python 3.10`
- `Ubuntu 22.04`
- `CUDA 11.8`

服务器中建议创建环境：

- conda 环境名：`lyc`

### 7.3 服务器环境安装建议

如果使用 `pip` 安装 GPU 版 PyTorch，已建议：

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

普通依赖建议：

```bash
pip install numpy==1.26.4 pillow tqdm pyyaml matplotlib
```

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 8. 训练命令与测试命令

### 8.1 WL only 训练

已验证有效的训练命令：

```bash
python train.py --modality WL --device cuda --amp --epochs 100 --batch-size 8 --num-workers 8 --image-size 512 512 --scheduler cosine
```

### 8.2 WL only 测试

```bash
python test.py --modality WL --checkpoint outputs/wl_unet/checkpoints/best.pt --device cuda --amp --batch-size 8 --num-workers 8 --image-size 512 512 --save-predictions
```

### 8.3 NBI only 训练

当前建议直接沿用和 `WL` 相同的 baseline 参数：

```bash
python train.py --modality NBI --device cuda --amp --epochs 100 --batch-size 8 --num-workers 8 --image-size 512 512 --scheduler cosine
```

### 8.4 NBI only 测试

```bash
python test.py --modality NBI --checkpoint outputs/nbi_unet/checkpoints/best.pt --device cuda --amp --batch-size 8 --num-workers 8 --image-size 512 512 --save-predictions
```

### 8.5 Dry-run 建议

上服务器或换环境后建议先用小测试检查训练链路：

```bash
python train.py --modality WL --device cuda --amp --epochs 1 --batch-size 2 --num-workers 4 --dry-run
```

## 9. WL only 实验结果总结

### 9.1 50 epoch 版本

最好结果：

- `best_epoch = 43`
- `best_val_dice = 0.6854`
- `best_val_iou = 0.5559`
- `best_val_precision = 0.6803`
- `best_val_recall = 0.7989`

结论：

- 训练收敛正常
- 没有明显严重过拟合
- `100 epoch` 有进一步提升空间

### 9.2 100 epoch 版本

最好结果：

- `best_epoch = 94`
- `best_val_dice = 0.7095`
- `best_val_iou = 0.5841`
- `best_val_precision = 0.7388`
- `best_val_recall = 0.7728`

对比 `50 epoch`：

- `val_dice` 提升约 `0.024`
- 说明延长训练到 `100 epoch` 是有效的

训练过程解读：

- 前期收敛较快
- 中期稳定提升
- 后期进入平台区
- 最优结果集中在 `90-98` 轮附近

判断：

- 当前没有明显严重过拟合
- `100 epoch` 的 `WL only` 可作为较可靠的 baseline

## 10. 关于是否继续调参的建议

已讨论并形成的判断：

- `epoch` 从 `50` 提升到 `100` 是有效的
- 再继续增加到 `120` 可以试，但收益可能有限
- `batch size` 不建议因为显存大就盲目拉太高
- 如果要尝试，建议优先试：
  - `batch size = 8`
  - 或 `12/16`

关于输入尺寸：

- `512x512` 是合理 baseline
- 如需进一步探索，可尝试：
  - `640x640`

更推荐的调参顺序：

1. 先固定当前 `100 epoch + 512x512 + batch 8`
2. 跑完 `NBI only`
3. 再试更大输入尺寸或更长训练

## 11. 训练日志怎么看

训练过程中终端最核心的监控指标：

- `train_loss`
- `train_dice`
- `val_loss`
- `val_dice`
- `lr`

其中最关键的是：

- `val_dice`
- `val_loss`

因为最终最优模型是按验证集 `Dice` 选出来的。

更完整的日志信息保存在：

- `outputs/<experiment>/logs/history.csv`
- `outputs/<experiment>/best_metrics.json`

## 12. 环境打包建议

已讨论过可以将环境完整打包，推荐：

- conda 环境名：`lyc`
- 使用 `conda-pack`

相关命令：

```bash
conda install -n base -c conda-forge conda-pack -y
conda pack -n lyc -o lyc_env.tar.gz
conda env export -n lyc --no-builds > lyc_environment.yml
pip freeze > lyc_requirements_freeze.txt
```

建议保留文件：

- `lyc_env.tar.gz`
- `lyc_environment.yml`
- `lyc_requirements_freeze.txt`

## 13. 当前状态与下一步

当前已经完成：

- 数据检查
- README 组织
- 工程结构搭建
- 数据集模块
- 预处理增强模块
- 标准 `U-Net`
- 训练脚本
- 测试脚本
- `WL only` baseline 训练与分析

当前建议的下一步：

- 按与 `WL only` 相同的参数运行 `NBI only`
- 跑完后分析 `history.csv` 与 `best_metrics.json`
- 再决定是否做更大输入尺寸或更长 epoch 的进一步实验
