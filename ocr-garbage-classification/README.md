# 手写数字识别与垃圾分类课程设计

本项目按课程 PDF 要求改造成 **LeNet5 主线**：先使用 MindSpore 在 MNIST 上从零训练 LeNet5 并保存预训练权重，再加载该权重到 26 类垃圾分类数据集上完成从零训练、冻结微调、全量微调、LoRA 微调和模型改进实验。MobileNetV2 相关代码保留为拓展/附录，不再作为正文主实验。

## 1. 实验主线

必做实验如下：

| 编号 | 实验 | 说明 |
|---:|---|---|
| 1 | LeNet5 + MNIST 从零训练 | 生成 `checkpoints/lenet_mnist.ckpt` |
| 2 | LeNet5 + 垃圾分类从零训练 | 不加载 MNIST 权重，作为垃圾分类 scratch 基线 |
| 3 | LeNet5 + MNIST 预训练 + 冻结微调 | 冻结 `conv1`、`conv2`、`fc1`、`fc2`，只训练 `fc3` |
| 4 | LeNet5 + MNIST 预训练 + 全量微调 | 所有层参与训练 |
| 5 | LeNet5 + MNIST 预训练 + LoRA 微调 | 在 `fc1`、`fc2` 加 LoRA，只训练 LoRA 参数和 `fc3` |
| 6 | 模型改进实验 | 默认使用更强数据增强，保持其他超参数一致 |

垃圾分类图片适配 LeNet5 的流程为：读取 RGB 图片后转灰度，resize 到 `32x32`，缩放到 `[0, 1]`，使用 MNIST 均值/方差归一化，再转为 MindSpore 使用的 `CHW` 格式。

## 2. 项目结构

```text
.
├── data/
│   ├── mnist/                    # MNIST gz 文件，可由脚本自动下载
│   └── garbage_official/          # 26 类垃圾分类数据集
│       ├── train/
│       └── test/
├── checkpoints/                   # 预训练权重与最优 checkpoint
├── logs/                          # CSV 训练日志
├── results/                       # JSON summary、类别索引、曲线 PNG
├── reports/
│   ├── report.tex
│   └── experiment_results.md
├── scripts/
│   └── run_lenet_experiments.ps1
└── src/
    ├── model.py                   # LeNet5、MobileNetV2、SmallCNN
    ├── train_mnist_lenet.py        # MNIST 预训练
    ├── train_lenet_garbage.py      # LeNet5 垃圾分类主实验
    ├── eval_lenet.py               # LeNet5 专用评估
    ├── lora.py
    ├── plot_curves.py
    └── utils.py
```

## 3. 环境

推荐在 Windows + Docker Desktop + WSL2 + NVIDIA GPU 下运行 MindSpore GPU 容器。

```bash
docker compose build
docker compose run --rm mindspore-gpu python -c "import mindspore; mindspore.run_check()"
```

如需验证 Docker 能否访问 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## 4. 数据集

垃圾分类数据集推荐使用官方 train/test 划分：

```text
data/garbage_official/
├── train/
│   ├── Banana Peel/
│   ├── Basketball/
│   └── ...
└── test/
    ├── Banana Peel/
    ├── Basketball/
    └── ...
```

查看数据集统计：

```bash
docker compose run --rm mindspore-gpu python -c "from src.dataset import print_dataset_summary; print_dataset_summary('data/garbage_official')"
```

## 5. 一键运行 LeNet5 实验

PowerShell 下执行：

```powershell
.\scripts\run_lenet_experiments.ps1
```

该脚本会依次运行 MNIST 预训练、垃圾分类从零训练、冻结微调、全量微调、LoRA 微调、增强改进实验，并为每个实验生成曲线。

## 6. 分步运行命令

MNIST 上从零训练 LeNet5：

```bash
docker compose run --rm mindspore-gpu python src/train_mnist_lenet.py \
  --run_name lenet_mnist \
  --epochs 5 \
  --batch_size 64 \
  --image_size 32 \
  --lr 1e-3
```

绘制 MNIST 曲线：

```bash
docker compose run --rm mindspore-gpu python src/plot_curves.py \
  --log logs/train_lenet_mnist.csv \
  --out results/curve_lenet_mnist.png
```

LeNet5 垃圾分类从零训练：

```bash
docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py \
  --data_dir data/garbage_official \
  --mode scratch \
  --run_name lenet_garbage_scratch \
  --epochs 24 \
  --batch_size 32 \
  --image_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4
```

MNIST 预训练 + 冻结微调：

```bash
docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py \
  --data_dir data/garbage_official \
  --pretrained checkpoints/lenet_mnist.ckpt \
  --mode freeze \
  --run_name lenet_garbage_freeze \
  --epochs 24 \
  --batch_size 32 \
  --image_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4
```

MNIST 预训练 + 全量微调：

```bash
docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py \
  --data_dir data/garbage_official \
  --pretrained checkpoints/lenet_mnist.ckpt \
  --mode full \
  --run_name lenet_garbage_full \
  --epochs 24 \
  --batch_size 32 \
  --image_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4
```

MNIST 预训练 + LoRA 微调：

```bash
docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py \
  --data_dir data/garbage_official \
  --pretrained checkpoints/lenet_mnist.ckpt \
  --mode lora \
  --run_name lenet_garbage_lora \
  --epochs 24 \
  --batch_size 32 \
  --image_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --lora_rank 4 \
  --lora_alpha 8
```

改进实验：只增加更强数据增强，其他超参数与全量微调一致。

```bash
docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py \
  --data_dir data/garbage_official \
  --pretrained checkpoints/lenet_mnist.ckpt \
  --mode full \
  --run_name lenet_garbage_full_aug \
  --epochs 24 \
  --batch_size 32 \
  --image_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --strong_aug
```

评估指定 checkpoint：

```bash
docker compose run --rm mindspore-gpu python src/eval_lenet.py \
  --data_dir data/garbage_official \
  --ckpt checkpoints/best_lenet_garbage_full.ckpt \
  --num_classes 26 \
  --out results/eval_lenet_garbage_full.json
```

绘制垃圾分类曲线：

```bash
docker compose run --rm mindspore-gpu python src/plot_curves.py \
  --log logs/train_lenet_garbage_full.csv \
  --out results/curve_lenet_garbage_full.png
```

## 7. 输出文件

每次训练会生成：

- `logs/train_{run_name}.csv`：逐 epoch 记录 loss、accuracy、耗时、显存和可训练参数量。
- `results/summary_{run_name}.json`：记录最优准确率、训练时间、参数量、冻结策略、LoRA 配置、checkpoint 路径。
- `checkpoints/best_{run_name}.ckpt`：验证准确率最高的 checkpoint。
- `results/curve_{run_name}.png`：训练损失和准确率曲线。

本轮完整运行结果：

| 实验 | 最优准确率 | 可训练参数量 | 总训练时间 | 峰值显存 |
|---|---:|---:|---:|---:|
| LeNet5 + MNIST 从零训练 | 98.60% | 61684 | 266.77 s | 3698 MB |
| LeNet5 + 垃圾分类从零训练 | 3.85% | 63044 | 362.80 s | 3698 MB |
| MNIST 预训练 + 冻结微调 | 15.38% | 2210 | 376.04 s | 3610 MB |
| MNIST 预训练 + 全量微调 | 64.23% | 63044 | 362.99 s | 3698 MB |
| MNIST 预训练 + LoRA 微调 | 25.38% | 5106 | 379.66 s | 3610 MB |
| 全量微调 + 强数据增强 | 38.85% | 63044 | 364.80 s | 3698 MB |

结果解读要点：

- LoRA 在 LeNet5 这种小模型上主要用于验证 MindSpore 中低秩适配器插入、主干冻结和增量参数训练流程；它并没有带来实际训练时间优势，准确率也低于全量微调。
- 强数据增强是一次负向消融。`results/preprocess_lenet_preview.png` 显示随机旋转会在 32x32 灰度输入中引入明显黑色边框，可能形成伪特征，因此后续应调整旋转填充值、插值方式或增强顺序。

## 8. 策略说明

冻结微调：加载 MNIST 预训练权重，冻结 `conv1`、`conv2`、`fc1`、`fc2`，只训练最后分类层 `fc3`。该策略可训练参数最少，训练速度快，但垃圾分类和 MNIST 差异较大，准确率上限可能受限。

全量微调：加载 MNIST 预训练权重后，所有层都参与训练。该策略适应新任务能力最强，但训练时间和内存占用更高，也更容易过拟合。

LoRA 微调：在 LeNet5 的 `fc1` 和 `fc2` 上加入低秩分支 `B(A(x)) * alpha / rank`，冻结原始主干权重，只训练 LoRA 参数和 `fc3`。它比冻结微调更灵活，又比全量微调训练参数更少。

模型改进实验：默认使用 `--strong_aug`，即在灰度图上增加随机旋转和随机水平翻转。为了公平对比，学习率、epoch、batch size、权重衰减、输入尺寸和随机种子保持不变。

## 9. MobileNetV2 拓展

`src/train.py`、`src/train_official_mobilenet.py`、`src/finetune.py` 中的 MobileNetV2 代码仍保留，可作为附录或拓展实验使用。课程 PDF 和本次报告主线以 LeNet5 为准。
