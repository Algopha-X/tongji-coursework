from typing import List, Optional

try:
    import mindspore.nn as nn
    import mindspore.ops as ops
except ImportError:
    nn = None
    ops = None

try:
    from src.lora import LoRAConv2d, LoRADenseFromBase
except ImportError:
    from lora import LoRAConv2d, LoRADenseFromBase


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Cell if nn else object):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        use_lora=False,
        lora_rank=4,
        lora_alpha=8.0,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to build models.")

        padding = (kernel_size - 1) // 2
        if use_lora and kernel_size == 1 and groups == 1:
            conv = LoRAConv2d(in_channels, out_channels, rank=lora_rank, alpha=lora_alpha)
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="pad",
                padding=padding,
                group=groups,
                has_bias=False,
            )
        self.block = nn.SequentialCell([
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        ])

    def construct(self, x):
        return self.block(x)


class InvertedResidual(nn.Cell if nn else object):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        use_lora=False,
        lora_rank=4,
        lora_alpha=8.0,
    ):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNReLU(
                in_channels,
                hidden_dim,
                kernel_size=1,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            ))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
        ])

        if use_lora:
            pointwise = LoRAConv2d(hidden_dim, out_channels, rank=lora_rank, alpha=lora_alpha)
        else:
            pointwise = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=False)
        layers.extend([pointwise, nn.BatchNorm2d(out_channels)])
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Cell if nn else object):
    def __init__(
        self,
        num_classes=26,
        width_mult=1.0,
        dropout=0.2,
        use_lora=False,
        lora_rank=4,
        lora_alpha=8.0,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to build models.")

        block = InvertedResidual
        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * max(1.0, width_mult))
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(
                    input_channel,
                    output_channel,
                    stride,
                    expand_ratio=t,
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                ))
                input_channel = output_channel
        features.append(ConvBNReLU(
            input_channel,
            last_channel,
            kernel_size=1,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        ))
        self.features = nn.SequentialCell(features)
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.classifier = nn.SequentialCell([nn.Dropout(p=dropout), nn.Dense(last_channel, num_classes)])

    def construct(self, x):
        x = self.features(x)
        x = self.reduce_mean(x, (2, 3))
        return self.classifier(x)


class SmallCNN(nn.Cell if nn else object):
    def __init__(self, num_classes=26, dropout=0.3):
        super().__init__()
        self.features = nn.SequentialCell([
            ConvBNReLU(3, 32, stride=2),
            ConvBNReLU(32, 64, stride=2),
            ConvBNReLU(64, 128, stride=2),
            ConvBNReLU(128, 256, stride=2),
        ])
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.classifier = nn.SequentialCell([nn.Dropout(p=dropout), nn.Dense(256, num_classes)])

    def construct(self, x):
        x = self.features(x)
        x = self.reduce_mean(x, (2, 3))
        return self.classifier(x)


class LeNet5(nn.Cell if nn else object):
    """LeNet5 backbone used as the main line for MNIST pretraining and garbage fine-tuning."""

    def __init__(
        self,
        num_classes=10,
        dropout: float = 0.0,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to build models.")

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode="valid")
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        fc1 = nn.Dense(16 * 5 * 5, 120)
        fc2 = nn.Dense(120, 84)
        if use_lora:
            fc1 = LoRADenseFromBase(fc1, rank=lora_rank, alpha=lora_alpha, name_prefix="fc1")
            fc2 = LoRADenseFromBase(fc2, rank=lora_rank, alpha=lora_alpha, name_prefix="fc2")
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = nn.Dense(84, num_classes)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.use_lora = use_lora

    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc3(x)


def build_model(
    model_name: str = "mobilenetv2",
    num_classes: int = 26,
    dropout: float = 0.2,
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
):
    name = model_name.lower()
    if name == "mobilenetv2":
        return MobileNetV2(
            num_classes=num_classes,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    if name == "lenet5":
        return LeNet5(
            num_classes=num_classes,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
    raise ValueError(f"Unsupported model: {model_name}")
