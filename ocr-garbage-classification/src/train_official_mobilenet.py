import argparse
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from src.dataset import create_dataset, write_class_index
from src.lora import LoRAConv2dFromBase
from src.utils import (
    append_csv,
    build_lr,
    ensure_dir,
    evaluate_network,
    gpu_memory_mb,
    parameter_report,
    save_json,
    set_seed,
    setup_context,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Official pretrained MobileNetV2 experiments with MindCV")
    parser.add_argument("--data_dir", default="data/garbage_official")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--mode", default="full", choices=["freeze", "full", "lora"])
    parser.add_argument("--model_name", default="mobilenet_v2_100")
    parser.add_argument("--pretrained", action="store_true", help="Load official MindCV pretrained weights")
    parser.add_argument("--checkpoint_path", default="", help="Optional MindCV-compatible checkpoint path")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine", "step"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument(
        "--lora_max_layers",
        type=int,
        default=12,
        help="Number of 1x1 convolution layers, counted from the end, to wrap with LoRA. Use 0 for all.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device_target", default="GPU")
    parser.add_argument("--context_mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


def create_official_model(model_name: str, num_classes: int, pretrained: bool, checkpoint_path: str = ""):
    try:
        from mindcv.models import create_model
        from mindcv.models import registry
    except ImportError as exc:
        raise ImportError(
            "MindCV is required for official pretrained MobileNet experiments. "
            "Install it in the container with: python -m pip install mindcv==0.2.2"
        ) from exc

    try:
        registered = registry.list_models()
    except Exception:
        registered = []

    mobilenet_names = [name for name in registered if "mobilenet" in name.lower()]
    print(f"MindCV MobileNet candidates: {mobilenet_names}")
    aliases = [
        model_name,
        "mobilenet_v2_100",
        "mobilenetv2_100",
        "mobilenetv2",
        "mobilenet_v2",
        "mobilenet_v2_035_224",
        "mobilenet_v2_050_224",
        "mobilenet_v2_075_224",
        "mobilenet_v2_100_224",
    ] + mobilenet_names
    last_error = None
    for candidate in dict.fromkeys(aliases):
        kwargs = {
            "model_name": candidate,
            "num_classes": num_classes,
            "pretrained": pretrained,
        }
        try:
            network = create_model(**kwargs)
            print(f"Created MindCV model: {candidate}")
            return network
        except Exception as exc:
            last_error = exc
            print(f"Failed to create MindCV model '{candidate}': {exc}")
    raise RuntimeError("Could not create an official MobileNetV2 model with MindCV.") from last_error


def _candidate_param_names(name: str):
    names = [name]
    if name.startswith("backbone."):
        names.append(name[len("backbone."):])
    if name.startswith("head.classifier."):
        tail = name[len("head.classifier."):]
        names.extend([
            f"classifier.{tail}",
            f"head.{tail}",
            f"dense.{tail}",
            f"fc.{tail}",
        ])
    cleaned = []
    for item in names:
        if item.startswith("moments.") or item in {"step", "global_step", "learning_rate", "momentum"}:
            continue
        cleaned.append(item)
    return cleaned


def load_mindcv_checkpoint_clean(network, checkpoint_path: str = ""):
    if not checkpoint_path:
        return {"loaded": 0, "skipped": 0, "checkpoint_path": ""}

    raw_params = load_checkpoint(checkpoint_path)
    target_shapes = {param.name: tuple(param.shape) for param in network.get_parameters()}
    mapped = {}
    skipped = 0

    for ckpt_name, tensor in raw_params.items():
        if ckpt_name.startswith("moments.") or ckpt_name in {"step", "global_step", "learning_rate", "momentum"}:
            skipped += 1
            continue
        loaded = False
        for target_name in _candidate_param_names(ckpt_name):
            if target_name not in target_shapes:
                continue
            if tuple(tensor.shape) != target_shapes[target_name]:
                skipped += 1
                loaded = True
                break
            mapped[target_name] = tensor
            loaded = True
            break
        if not loaded:
            skipped += 1

    not_loaded = load_param_into_net(network, mapped, strict_load=False)
    report = {
        "loaded": len(mapped),
        "skipped": skipped,
        "not_loaded": len(not_loaded) if not_loaded else 0,
        "checkpoint_path": checkpoint_path,
    }
    print(f"Loaded cleaned checkpoint: {report}")
    if report["loaded"] < 20:
        print("Warning: few checkpoint parameters were loaded; checkpoint may be incompatible.")
    return report


def _is_lora_target_conv(cell):
    if not isinstance(cell, nn.Conv2d):
        return False
    weight_shape = tuple(cell.weight.shape)
    if len(weight_shape) != 4 or weight_shape[2:] != (1, 1):
        return False
    group = getattr(cell, "group", 1)
    return int(group) == 1


def _collect_lora_targets(cell, prefix=""):
    targets = []
    for child_name, child in cell.name_cells().items():
        if child_name == "":
            continue
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if _is_lora_target_conv(child):
            targets.append((cell, child_name, full_name, child))
        if isinstance(child, nn.Cell):
            targets.extend(_collect_lora_targets(child, full_name))
    return targets


def apply_lora_to_official_model(network, rank: int = 4, alpha: float = 8.0, max_layers: int = 12):
    targets = _collect_lora_targets(network)
    if max_layers > 0:
        targets = targets[-max_layers:]

    for parent, child_name, full_name, conv in targets:
        lora_cell = LoRAConv2dFromBase(conv, rank=rank, alpha=alpha, name_prefix=full_name)
        if isinstance(parent, nn.SequentialCell) and child_name.isdigit():
            parent[int(child_name)] = lora_cell
        else:
            parent.insert_child_to_cell(child_name, lora_cell)

    print(f"Applied LoRA to {len(targets)} official MobileNet 1x1 conv layers.")
    if targets:
        print("LoRA target layers:")
        for _, _, full_name, conv in targets:
            print(f"  - {full_name}: shape={tuple(conv.weight.shape)}")
    return [full_name for _, _, full_name, _ in targets]


def set_official_finetune_mode(network, mode: str):
    for param in network.get_parameters():
        param.requires_grad = True

    if mode == "full":
        return
    if mode == "lora":
        classifier_keywords = ("classifier", "head", "fc", "dense")
        for param in network.get_parameters():
            lowered = param.name.lower()
            param.requires_grad = ("lora_" in lowered) or any(key in lowered for key in classifier_keywords)
        return
    if mode != "freeze":
        raise ValueError(f"Unsupported mode: {mode}")

    classifier_keywords = ("classifier", "head", "fc", "dense")
    for param in network.get_parameters():
        lowered = param.name.lower()
        param.requires_grad = any(key in lowered for key in classifier_keywords)

    if network.trainable_params():
        return

    # Fallback for model definitions with unusual names: keep the last two tensors trainable.
    params = list(network.get_parameters())
    for param in params:
        param.requires_grad = False
    for param in params[-2:]:
        param.requires_grad = True


def ensure_unique_parameter_names(network):
    seen = {}
    for param in network.get_parameters():
        original_name = param.name
        count = seen.get(original_name, 0)
        if count:
            param.name = f"{original_name}_{count}"
        seen[original_name] = count + 1


def train_one_epoch(network, dataset, loss_fn, optimizer):
    network.set_train(True)

    def forward_fn(images, labels):
        logits = network(images)
        loss = loss_fn(logits, labels)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    for images, labels in dataset.create_tuple_iterator():
        (loss, logits), grads = grad_fn(images, labels)
        optimizer(grads)
        total_loss += float(loss.asnumpy())
        pred = ops.Argmax(axis=1)(logits).asnumpy()
        gt = labels.asnumpy()
        correct += int((pred == gt).sum())
        total += int(gt.shape[0])
        steps += 1

    return {"loss": total_loss / max(steps, 1), "acc": correct / max(total, 1)}


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_context(args.device_target, args.context_mode)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.log_dir)
    ensure_dir(args.results_dir)

    train_ds, class_names, stats, train_count, val_count = create_dataset(
        args.data_dir,
        split="train",
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_ds, _, _, _, _ = create_dataset(
        args.data_dir,
        split="val",
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    write_class_index(class_names, str(Path(args.results_dir) / "class_index.json"))

    network = create_official_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=args.pretrained,
        checkpoint_path="",
    )
    checkpoint_report = load_mindcv_checkpoint_clean(network, args.checkpoint_path)
    lora_targets = []
    if args.mode == "lora":
        lora_targets = apply_lora_to_official_model(
            network,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            max_layers=args.lora_max_layers,
        )
    set_official_finetune_mode(network, args.mode)
    ensure_unique_parameter_names(network)
    params = parameter_report(network)

    print(f"Dataset: {len(class_names)} classes, train={train_count}, val={val_count}")
    print(f"Samples per class: {stats}")
    print(f"Official model: {args.model_name}, pretrained={args.pretrained}, checkpoint_path={args.checkpoint_path}")
    print(f"Parameters: {params}")

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    steps_per_epoch = train_ds.get_dataset_size()
    lr = build_lr(
        base_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        steps_per_epoch=steps_per_epoch,
        schedule=args.lr_schedule,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        step_size=args.step_size,
        gamma=args.gamma,
    )
    optimizer = nn.Adam(network.trainable_params(), learning_rate=lr, weight_decay=args.weight_decay)

    log_path = str(Path(args.log_dir) / f"train_{args.run_name}.csv")
    best_acc = -1.0
    best_path = str(Path(args.ckpt_dir) / f"best_{args.run_name}.ckpt")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(network, train_ds, loss_fn, optimizer)
        val_metrics = evaluate_network(network, val_ds)
        used_mem = gpu_memory_mb()
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "mode": args.mode,
            "run_name": args.run_name,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "epoch_seconds": round(elapsed, 2),
            "gpu_memory_mb": used_mem,
            "trainable_params": params["trainable"],
        }
        append_csv(log_path, row)
        print(row)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(network, best_path)
            print(f"Saved best checkpoint to {best_path}")

    summary = {
        "mode": args.mode,
        "run_name": args.run_name,
        "model": args.model_name,
        "official_model": True,
        "pretrained": args.pretrained,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_report": checkpoint_report,
        "classes": class_names,
        "train_samples": train_count,
        "val_samples": val_count,
        "best_val_acc": best_acc,
        "total_seconds": round(time.perf_counter() - start_time, 2),
        "parameters": params,
        "lr_schedule": args.lr_schedule,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "lora_rank": args.lora_rank if args.mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        "lora_targets": lora_targets,
        "best_ckpt": best_path,
    }
    save_json(str(Path(args.results_dir) / f"summary_{args.run_name}.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
