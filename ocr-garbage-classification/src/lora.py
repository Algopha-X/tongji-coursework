from typing import Optional

try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore.common.initializer import HeUniform, Zero, initializer
except ImportError:
    ms = None
    nn = None
    ops = None
    HeUniform = None
    Zero = None
    initializer = None


class LoRAConv2d(nn.Cell if nn else object):
    """LoRA wrapper for 1x1 Conv2d.

    The base convolution keeps its pretrained weights frozen. The trainable low-rank
    branch learns B(A(x)) * alpha / rank and is added to the base output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 4,
        alpha: float = 8.0,
        has_bias: bool = False,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to use LoRAConv2d.")

        self.base = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            pad_mode="valid",
            has_bias=has_bias,
        )
        self.lora_a = nn.Conv2d(
            in_channels,
            rank,
            kernel_size=1,
            stride=1,
            pad_mode="valid",
            has_bias=False,
            weight_init=HeUniform(),
        )
        self.lora_b = nn.Conv2d(
            rank,
            out_channels,
            kernel_size=1,
            stride=1,
            pad_mode="valid",
            has_bias=False,
            weight_init=Zero(),
        )
        self.scale = alpha / float(rank)
        self.base.weight.requires_grad = False
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad = False

    def construct(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class LoRAConv2dFromBase(nn.Cell if nn else object):
    """LoRA wrapper that keeps an existing 1x1 convolution as the frozen base."""

    def __init__(self, base_conv, rank: int = 4, alpha: float = 8.0, name_prefix: Optional[str] = None):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to use LoRAConv2dFromBase.")

        weight_shape = tuple(base_conv.weight.shape)
        if len(weight_shape) != 4 or weight_shape[2:] != (1, 1):
            raise ValueError(f"LoRAConv2dFromBase only supports 1x1 Conv2d, got shape={weight_shape}")

        out_channels, in_channels = int(weight_shape[0]), int(weight_shape[1])
        self.base = base_conv
        self.lora_a = nn.Conv2d(
            in_channels,
            rank,
            kernel_size=1,
            stride=1,
            pad_mode="valid",
            has_bias=False,
            weight_init=HeUniform(),
        )
        self.lora_b = nn.Conv2d(
            rank,
            out_channels,
            kernel_size=1,
            stride=1,
            pad_mode="valid",
            has_bias=False,
            weight_init=Zero(),
        )
        if name_prefix:
            safe_prefix = name_prefix.replace(".", "_")
            self.base.weight.name = f"{safe_prefix}.base.weight"
            if getattr(self.base, "bias", None) is not None:
                self.base.bias.name = f"{safe_prefix}.base.bias"
            self.lora_a.weight.name = f"{safe_prefix}.lora_a.weight"
            self.lora_b.weight.name = f"{safe_prefix}.lora_b.weight"
        self.scale = alpha / float(rank)
        self.base.weight.requires_grad = False
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad = False

    def construct(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class LoRADense(nn.Cell if nn else object):
    def __init__(self, in_channels: int, out_channels: int, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to use LoRADense.")

        self.base = nn.Dense(in_channels, out_channels)
        self.lora_a = nn.Dense(in_channels, rank, has_bias=False, weight_init=HeUniform())
        self.lora_b = nn.Dense(rank, out_channels, has_bias=False, weight_init=Zero())
        self.scale = alpha / float(rank)
        self.base.weight.requires_grad = False
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad = False

    def construct(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class LoRADenseFromBase(nn.Cell if nn else object):
    """LoRA wrapper that keeps an existing Dense layer as the frozen base."""

    def __init__(self, base_dense, rank: int = 4, alpha: float = 8.0, name_prefix: Optional[str] = None):
        super().__init__()
        if nn is None:
            raise ImportError("MindSpore is required to use LoRADenseFromBase.")

        weight_shape = tuple(base_dense.weight.shape)
        if len(weight_shape) != 2:
            raise ValueError(f"LoRADenseFromBase only supports Dense, got shape={weight_shape}")

        out_channels, in_channels = int(weight_shape[0]), int(weight_shape[1])
        self.base = base_dense
        self.lora_a = nn.Dense(in_channels, rank, has_bias=False, weight_init=HeUniform())
        self.lora_b = nn.Dense(rank, out_channels, has_bias=False, weight_init=Zero())
        if name_prefix:
            safe_prefix = name_prefix.replace(".", "_")
            self.base.weight.name = f"{safe_prefix}.base.weight"
            if getattr(self.base, "bias", None) is not None:
                self.base.bias.name = f"{safe_prefix}.base.bias"
            self.lora_a.weight.name = f"{safe_prefix}.lora_a.weight"
            self.lora_b.weight.name = f"{safe_prefix}.lora_b.weight"
        self.scale = alpha / float(rank)
        self.base.weight.requires_grad = False
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad = False

    def construct(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale
