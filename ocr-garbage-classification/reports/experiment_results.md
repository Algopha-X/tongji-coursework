# LeNet5 实验结果记录

本文件按《深度学习模块：手写数字识别与垃圾分类》要求记录 LeNet5 主线实验。所有实验使用 MindSpore 实现，垃圾分类图片统一转灰度、resize 到 32x32、归一化后输入 LeNet5。

## 数据集

| 数据集 | 划分 | 样本数 | 类别数 |
|---|---|---:|---:|
| MNIST | train | 60000 | 10 |
| MNIST | test | 10000 | 10 |
| 垃圾分类 | train | 2593 | 26 |
| 垃圾分类 | test/val | 260 | 26 |

## 完整 LeNet5 结果

| 实验 | 预训练 | 冻结层 | LoRA 位置 | 可训练参数量 | 总训练时间 | 峰值显存 | 最优验证准确率 | 最优 epoch | 曲线 |
|---|---|---|---|---:|---:|---:|---|
| LeNet5 + MNIST 从零训练 | 无 | 无 | 无 | 61684 | 266.77 s | 3698 MB | 98.60% | 5 | `results/curve_lenet_mnist.png` |
| LeNet5 + 垃圾分类从零训练 | 无 | 无 | 无 | 63044 | 362.80 s | 3698 MB | 3.85% | 1 | `results/curve_lenet_garbage_scratch.png` |
| LeNet5 + MNIST 预训练 + 冻结微调垃圾分类 | MNIST | `conv1, conv2, fc1, fc2` | 无 | 2210 | 376.04 s | 3610 MB | 15.38% | 14 | `results/curve_lenet_garbage_freeze.png` |
| LeNet5 + MNIST 预训练 + 全量微调垃圾分类 | MNIST | 无 | 无 | 63044 | 362.99 s | 3698 MB | 64.23% | 24 | `results/curve_lenet_garbage_full.png` |
| LeNet5 + MNIST 预训练 + LoRA 微调垃圾分类 | MNIST | 原始主干 | `fc1, fc2` | 5106 | 379.66 s | 3610 MB | 25.38% | 24 | `results/curve_lenet_garbage_lora.png` |
| LeNet5 改进实验：强数据增强 | MNIST | 无 | 无 | 63044 | 364.80 s | 3698 MB | 38.85% | 24 | `results/curve_lenet_garbage_full_aug.png` |

以上数值来自本轮运行 `scripts/run_lenet_experiments.ps1` 后生成的 `results/summary_*.json` 和 `logs/train_*.csv`。

## 对比维度

正式报告中的主对比表应包含：

| 策略 | 冻结层 | 训练层 | LoRA 位置 | 可训练参数量 | 总训练时间 | 峰值显存/内存 | 最优验证准确率 |
|---|---|---|---|---:|---:|---:|---:|
| 从零训练 | 无 | 全部 LeNet5 | 无 | 63044 | 362.80 s | 3698 MB | 3.85% |
| 冻结微调 | `conv1, conv2, fc1, fc2` | `fc3` | 无 | 2210 | 376.04 s | 3610 MB | 15.38% |
| 全量微调 | 无 | 全部 LeNet5 | 无 | 63044 | 362.99 s | 3698 MB | 64.23% |
| LoRA 微调 | 原始 `conv1, conv2, fc1, fc2` 主干 | `fc1/fc2` LoRA + `fc3` | `fc1, fc2` | 5106 | 379.66 s | 3610 MB | 25.38% |

LoRA 结果说明：在 LeNet5 这种仅 6 万级参数的小模型上，LoRA 的主要价值是验证低秩适配器插入与主干冻结流程。它没有体现出大模型 PEFT 中常见的训练时间或显存优势，准确率也明显低于全量微调。

模型改进实验需要保持全量微调和强数据增强两组超参数一致，只改变 `--strong_aug`：

| 实验 | 改动 | 其他超参数 | 最优验证准确率 |
|---|---|---|---:|
| 基线 | 默认预处理 | 相同 | 64.23% |
| 改进 | `--strong_aug` | 相同 | 38.85% |

强数据增强结果说明：该实验是负向消融。`results/preprocess_lenet_preview.png` 显示随机旋转在 32x32 灰度输入中引入了明显黑色边框，可能形成与类别无关的伪特征。后续应调整旋转填充值、插值方式或增强顺序，而不是简单认为数据增强思想本身无效。

## LeNet5 局限

- 垃圾分类图片转灰度会丢失颜色信息，而颜色常是区分垃圾类别的重要线索。
- 输入尺寸从真实图片压缩到 32x32，会损失纹理和细节。
- LeNet5 参数量小、层数浅，对 26 类细粒度垃圾分类的表达能力有限。
- MNIST 与垃圾分类数据分布差异很大，冻结微调只训练最后分类层时适应能力明显不足。
