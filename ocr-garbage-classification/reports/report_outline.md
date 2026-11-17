# 课程设计报告大纲：手写数字识别与垃圾分类

> 建议控制在 25-30 页以内。表格中的结果先留空，训练完成后再填入。

## 1. 实验背景

- 深度学习图像分类任务简介。
- LeNet5 在 MNIST 手写数字识别中的教学意义。
- 从手写数字识别扩展到真实图像垃圾分类的必要性。
- Fine-Tune 思想：利用预训练模型降低小数据集训练难度。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 主机系统 | Windows，待填 |
| GPU | NVIDIA，待填 |
| Docker Desktop | 待填 |
| WSL2 发行版 | Ubuntu，待填 |
| MindSpore 镜像 | 待填 |
| Python | 待填 |
| CUDA 运行环境 | 待填 |

## 3. Docker + MindSpore GPU 环境配置

- Windows 安装 NVIDIA 驱动。
- 启用 WSL2 和 Docker Desktop WSL2 backend。
- 使用 `nvidia/cuda:12.2.0-base-ubuntu22.04` 验证 `nvidia-smi`。
- 拉取 MindSpore GPU 镜像。
- 使用 `docker compose` 挂载 Windows 项目目录到容器 `/workspace`。
- 使用 `mindspore.run_check()` 验证 MindSpore GPU。

## 4. 数据集介绍

- 课程垃圾分类数据集来源。
- 分类类别数量：26 类。
- 数据目录组织方式：按类别文件夹存放图片。
- 训练/验证划分比例：默认 8:2。
- 数据预处理：Resize、Normalize、随机翻转。
- 样本统计表：

| 类别编号 | 类别名称 | 样本数量 |
|---:|---|---:|
| 0 | 待填 | 待填 |
| ... | ... | ... |

## 5. 开源项目复现说明

- 说明参考了 LeNet5、MobileNetV2、Fine-Tune 和 LoRA 的公开思想。
- 本项目没有直接复制开源项目代码，而是按课程任务重新组织 MindSpore 数据管道、模型、训练和评估脚本。
- 对比复现目标：跑通训练流程、保存最优模型、输出可复现实验日志。

## 6. 自己代码实现说明

- `src/dataset.py`：数据扫描、训练/验证划分、图像预处理、类别统计。
- `src/model.py`：LeNet5、轻量 CNN、MobileNetV2。
- `src/train.py`：训练循环、三种微调模式、日志和 best checkpoint。
- `src/eval.py`：加载 checkpoint 并评估验证集准确率。
- `src/lora.py`：LoRA Conv2d 和 Dense 模块，其中主实验使用 LoRA Conv2d。
- `src/utils.py`：随机种子、参数统计、GPU 显存统计、JSON/CSV 记录。

## 7. 模型结构

- LeNet5：用于说明 MNIST 手写数字识别入门案例。
- MobileNetV2：用于垃圾分类主实验。
- 倒残差结构：1x1 expansion、depthwise 3x3、1x1 projection。
- 分类头：全局平均池化 + Dropout + Dense。

## 8. 三种微调策略原理

### 8.1 冻结微调

- 冻结 backbone，只训练分类头。
- 可训练参数少，训练速度快。
- 适合数据较少或算力有限场景。

### 8.2 全量微调

- backbone 和分类头全部参与训练。
- 学习能力最强，但显存和时间成本最高。
- 需要较小学习率和更谨慎的正则化。

### 8.3 LoRA 微调

- 冻结原始权重，加入低秩分支。
- 本项目把 LoRA 加到 MobileNetV2 的 1x1 pointwise conv，分类头正常训练。
- 在可训练参数较少的前提下增强模型适应能力。

## 9. 实验设计

- 数据划分：训练集 80%，验证集 20%。
- 输入尺寸：224 x 224。
- 优化器：Adam。
- 损失函数：SoftmaxCrossEntropy。
- 评价指标：Top-1 Accuracy。
- 保持相同 batch size、epoch、输入尺寸和随机种子，比较不同微调策略。

## 10. 实验结果对比

| 微调策略 | 冻结层 | LoRA 位置 | 可训练参数量 | 总训练时间 | 峰值显存 | 最优验证准确率 |
|---|---|---|---:|---:|---:|---:|
| 冻结微调 | backbone | 无 | 待填 | 待填 | 待填 | 待填 |
| 全量微调 | 无 | 无 | 待填 | 待填 | 待填 | 待填 |
| LoRA 微调 | 原始主分支 | 1x1 conv | 待填 | 待填 | 待填 | 待填 |

可加入训练曲线图：

- `results/train_freeze.png`
- `results/train_full.png`
- `results/train_lora.png`

## 11. 模型改进实验

改进方案：加强数据增强。

| 实验 | 数据增强 | 其他超参数 | 验证准确率 |
|---|---|---|---:|
| 基线 | Resize + Normalize + RandomHorizontalFlip | 相同 | 待填 |
| 改进 | 增加颜色扰动和随机旋转 | 相同 | 待填 |

原因分析：

- 垃圾分类图像存在光照、角度和背景差异。
- 更强数据增强可以减少模型对固定拍摄条件的依赖。
- 若增强过强导致准确率下降，需要说明可能破坏了类别关键纹理或训练轮数不足。

## 12. 总结与不足

- 总结 MindSpore 图像分类完整流程。
- 比较三种微调策略在准确率、速度和显存上的权衡。
- 说明 Docker + WSL2 对 Windows GPU 环境配置的价值。
- 不足：数据规模有限、未做更多模型对比、LoRA 插入位置仍可优化、未做混淆矩阵分析。
- 后续改进：加入混淆矩阵、类别不平衡处理、学习率调度、更多预训练模型。
