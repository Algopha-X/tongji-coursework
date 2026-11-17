import csv
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import mindspore as ms
    from mindspore import context
    from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
except ImportError:
    ms = None
    context = None
    load_checkpoint = None
    load_param_into_net = None
    save_checkpoint = None


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if ms is not None:
        ms.set_seed(seed)


def setup_context(device_target: str = "GPU", mode: str = "GRAPH"):
    if context is None:
        raise ImportError("MindSpore is required to setup context.")
    graph_mode = context.GRAPH_MODE if mode.upper() == "GRAPH" else context.PYNATIVE_MODE
    context.set_context(mode=graph_mode, device_target=device_target)


def load_pretrained_if_available(network, ckpt_path: str = ""):
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    params = load_checkpoint(str(path))
    not_loaded = load_param_into_net(network, params, strict_load=False)
    print(f"Loaded pretrained checkpoint: {path}")
    if not_loaded:
        print(f"Parameters not loaded: {not_loaded}")


def _is_classifier_param(name: str) -> bool:
    lowered = name.lower()
    return "classifier" in lowered or "dense" in lowered or "head" in lowered


def load_pretrained_flexible(
    network,
    ckpt_path: str = "",
    min_loaded: int = 5,
    strategy: str = "name",
):
    """Load pretrained weights with a name-first and shape-fallback strategy.

    Official MobileNetV2 checkpoints do not always use the same parameter names as
    this course project. The safe default is name-only loading. Shape fallback is
    available for experiments, but it can corrupt the network when many tensors
    share identical shapes, such as BatchNorm parameters.
    """
    if not ckpt_path:
        return {"loaded_by_name": 0, "loaded_by_shape": 0, "skipped": 0, "strategy": strategy}
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    strategy = strategy.lower()
    if strategy not in {"name", "shape"}:
        raise ValueError(f"Unsupported pretrained loading strategy: {strategy}")

    ckpt_params = load_checkpoint(str(path))
    before = {p.name: p.asnumpy().copy() for p in network.get_parameters()}
    load_param_into_net(network, ckpt_params, strict_load=False)
    loaded_by_name = 0
    for param in network.get_parameters():
        if param.name in before and not np.array_equal(before[param.name], param.asnumpy()):
            loaded_by_name += 1

    if strategy == "name":
        skipped = sum(
            1
            for param in network.get_parameters()
            if not _is_classifier_param(param.name)
            and (param.name not in before or np.array_equal(before[param.name], param.asnumpy()))
        )
        report = {
            "loaded_by_name": loaded_by_name,
            "loaded_by_shape": 0,
            "skipped": skipped,
            "strategy": strategy,
        }
        print(f"Loaded pretrained checkpoint with name-only strategy: {path}")
        print(f"Pretrained load report: {report}")
        if loaded_by_name < min_loaded:
            print("Warning: very few pretrained parameters were loaded. Check checkpoint compatibility.")
        return report

    loaded_names = set()
    target_params = []
    for param in network.get_parameters():
        if _is_classifier_param(param.name):
            continue
        if param.name in before and not np.array_equal(before[param.name], param.asnumpy()):
            continue
        target_params.append(param)

    source_by_shape: Dict[Tuple[int, ...], List] = {}
    for name, tensor in ckpt_params.items():
        if _is_classifier_param(name):
            continue
        source_by_shape.setdefault(tuple(tensor.shape), []).append((name, tensor))

    loaded_by_shape = 0
    skipped = 0
    for param in target_params:
        candidates = source_by_shape.get(tuple(param.shape), [])
        chosen = None
        while candidates:
            cand_name, cand_tensor = candidates.pop(0)
            if cand_name not in loaded_names:
                chosen = (cand_name, cand_tensor)
                break
        if chosen is None:
            skipped += 1
            continue
        cand_name, cand_tensor = chosen
        param.set_data(cand_tensor)
        loaded_names.add(cand_name)
        loaded_by_shape += 1

    report = {
        "loaded_by_name": loaded_by_name,
        "loaded_by_shape": loaded_by_shape,
        "skipped": skipped,
        "strategy": strategy,
    }
    print(f"Loaded pretrained checkpoint with shape-fallback strategy: {path}")
    print(f"Pretrained load report: {report}")
    if loaded_by_name + loaded_by_shape < min_loaded:
        print("Warning: very few pretrained parameters were loaded. Check checkpoint compatibility.")
    return report


def build_lr(
    base_lr: float,
    total_steps: int,
    steps_per_epoch: int,
    schedule: str = "constant",
    min_lr: float = 1e-6,
    warmup_epochs: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
):
    if ms is None:
        raise ImportError("MindSpore is required to build a learning rate schedule.")
    schedule = schedule.lower()
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    values: List[float] = []

    for step in range(total_steps):
        if warmup_steps > 0 and step < warmup_steps:
            lr = base_lr * float(step + 1) / float(warmup_steps)
        elif schedule == "constant":
            lr = base_lr
        elif schedule == "cosine":
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / denom
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
        elif schedule == "step":
            epoch = step // max(steps_per_epoch, 1)
            lr = base_lr * (gamma ** (epoch // max(step_size, 1)))
            lr = max(lr, min_lr)
        else:
            raise ValueError(f"Unsupported lr schedule: {schedule}")
        values.append(float(lr))

    if schedule == "constant" and warmup_steps == 0:
        return base_lr
    return ms.Tensor(np.array(values, dtype=np.float32), ms.float32)


def set_finetune_mode(network, mode: str):
    mode = mode.lower()
    for param in network.get_parameters():
        param.requires_grad = True

    if mode == "freeze":
        for param in network.get_parameters():
            param.requires_grad = "classifier" in param.name
    elif mode == "full":
        pass
    elif mode == "lora":
        for param in network.get_parameters():
            param.requires_grad = ("lora_" in param.name) or ("classifier" in param.name)
    else:
        raise ValueError(f"Unsupported finetune mode: {mode}")


def parameter_report(network) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in network.get_parameters():
        count = int(np.prod(param.shape))
        total += count
        if param.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def append_csv(path: str, row: Dict):
    ensure_dir(str(Path(path).parent))
    write_header = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: str, obj: Dict):
    ensure_dir(str(Path(path).parent))
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def gpu_memory_mb() -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [int(x.strip()) for x in out.splitlines() if x.strip()]
        return max(values) if values else -1
    except Exception:
        return -1


def top1_accuracy(logits, labels) -> Tuple[int, int]:
    pred = np.argmax(logits.asnumpy(), axis=1)
    gt = labels.asnumpy()
    return int((pred == gt).sum()), int(gt.shape[0])


def evaluate_network(network, dataset) -> Dict[str, float]:
    network.set_train(False)
    correct = 0
    total = 0
    for images, labels in dataset.create_tuple_iterator():
        logits = network(images)
        batch_correct, batch_total = top1_accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
    return {"acc": correct / max(total, 1), "total": total}
