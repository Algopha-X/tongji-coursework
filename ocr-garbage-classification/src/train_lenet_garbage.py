import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from src.dataset import has_official_split, scan_image_folder, scan_official_split, split_samples, write_class_index
from src.model import LeNet5
from src.utils import append_csv, ensure_dir, gpu_memory_mb, parameter_report, save_json, set_seed, setup_context


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and fine-tune LeNet5 on 26-class garbage classification")
    parser.add_argument("--data_dir", default="data/garbage_official")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--mode", default="freeze", choices=["scratch", "freeze", "full", "lora"])
    parser.add_argument("--run_name", default="")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device_target", default="GPU")
    parser.add_argument("--context_mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


class GrayImageList:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        return np.asarray(image)[..., None], np.int32(label)


def collect_samples(data_dir: str, val_ratio: float, seed: int):
    if has_official_split(data_dir):
        train_samples, val_samples, class_names, stats = scan_official_split(data_dir)
        return train_samples, val_samples, class_names, stats
    samples, class_names, stats = scan_image_folder(data_dir)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)
    return train_samples, val_samples, class_names, stats


def create_gray_dataset(samples, split: str, image_size: int, batch_size: int, num_workers: int, stronger_aug: bool = False):
    dataset = ds.GeneratorDataset(
        GrayImageList(samples),
        column_names=["image", "label"],
        shuffle=(split == "train"),
        num_parallel_workers=num_workers,
    )
    image_ops = [vision.Resize((image_size, image_size))]
    if split == "train" and stronger_aug:
        image_ops.extend([
            vision.RandomRotation(12),
            vision.RandomHorizontalFlip(prob=0.5),
        ])
    image_ops.extend([
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW(),
    ])
    dataset = dataset.map(
        operations=image_ops,
        input_columns=["image"],
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(mstype.int32),
        input_columns=["label"],
        num_parallel_workers=num_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=(split == "train"))
    return dataset


def load_lenet_pretrained(network, ckpt_path: str):
    if not ckpt_path:
        return {"loaded": 0, "skipped": 0, "not_loaded": 0, "checkpoint_path": ""}
    raw_params = load_checkpoint(ckpt_path)
    target_shapes = {param.name: tuple(param.shape) for param in network.get_parameters()}
    alias_map = {
        "conv1.weight": ["net.0.weight"],
        "conv1.bias": ["net.0.bias"],
        "conv2.weight": ["net.3.weight"],
        "conv2.bias": ["net.3.bias"],
        "fc1.weight": ["net.7.weight", "fc1.base.weight"],
        "fc1.bias": ["net.7.bias", "fc1.base.bias"],
        "fc2.weight": ["net.9.weight", "fc2.base.weight"],
        "fc2.bias": ["net.9.bias", "fc2.base.bias"],
        "fc3.weight": ["net.11.weight"],
        "fc3.bias": ["net.11.bias"],
        "fc1.base.weight": ["fc1.weight", "net.7.weight"],
        "fc1.base.bias": ["fc1.bias", "net.7.bias"],
        "fc2.base.weight": ["fc2.weight", "net.9.weight"],
        "fc2.base.bias": ["fc2.bias", "net.9.bias"],
    }
    mapped = {}
    used_sources = set()
    skipped = 0
    for target_name, target_shape in target_shapes.items():
        candidates = [target_name] + alias_map.get(target_name, [])
        chosen = None
        for source_name in candidates:
            if source_name in used_sources or source_name not in raw_params:
                continue
            tensor = raw_params[source_name]
            if tuple(tensor.shape) == target_shape:
                chosen = (source_name, tensor)
                break
        if chosen is None:
            continue
        source_name, tensor = chosen
        mapped[target_name] = tensor
        used_sources.add(source_name)
    skipped = len(raw_params) - len(used_sources)

    not_loaded = load_param_into_net(network, mapped, strict_load=False)
    report = {
        "loaded": len(mapped),
        "skipped": skipped,
        "not_loaded": len(not_loaded) if not_loaded else 0,
        "checkpoint_path": ckpt_path,
    }
    print(f"Loaded LeNet pretrained checkpoint: {report}")
    return report


def set_lenet_finetune_mode(network, mode: str):
    for param in network.get_parameters():
        param.requires_grad = True
    if mode in {"scratch", "full"}:
        return
    if mode == "freeze":
        for param in network.get_parameters():
            param.requires_grad = param.name.startswith("fc3.")
        return
    if mode == "lora":
        for param in network.get_parameters():
            name = param.name.lower()
            param.requires_grad = ("lora_" in name) or name.startswith("fc3.")
        return
    raise ValueError(f"Unsupported LeNet5 fine-tune mode: {mode}")


def trainable_parameter_names(network):
    return [param.name for param in network.get_parameters() if param.requires_grad]


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
    argmax = ops.Argmax(axis=1)

    for images, labels in dataset.create_tuple_iterator():
        (loss, logits), grads = grad_fn(images, labels)
        optimizer(grads)
        total_loss += float(loss.asnumpy())
        pred = argmax(logits).asnumpy()
        gt = labels.asnumpy()
        correct += int((pred == gt).sum())
        total += int(gt.shape[0])
        steps += 1
    return {"loss": total_loss / max(steps, 1), "acc": correct / max(total, 1)}


def evaluate(network, dataset):
    network.set_train(False)
    correct = 0
    total = 0
    argmax = ops.Argmax(axis=1)
    for images, labels in dataset.create_tuple_iterator():
        logits = network(images)
        pred = argmax(logits).asnumpy()
        gt = labels.asnumpy()
        correct += int((pred == gt).sum())
        total += int(gt.shape[0])
    return {"acc": correct / max(total, 1), "total": total}


def main():
    args = parse_args()
    run_tag = args.run_name.strip() or f"lenet_garbage_{args.mode}"
    set_seed(args.seed)
    setup_context(args.device_target, args.context_mode)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.log_dir)
    ensure_dir(args.results_dir)

    train_samples, val_samples, class_names, stats = collect_samples(args.data_dir, args.val_ratio, args.seed)
    train_ds = create_gray_dataset(
        train_samples,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        stronger_aug=args.strong_aug,
    )
    val_ds = create_gray_dataset(val_samples, "val", args.image_size, args.batch_size, args.num_workers)
    write_class_index(class_names, str(Path(args.results_dir) / "class_index.json"))

    use_lora = args.mode == "lora"
    network = LeNet5(
        num_classes=len(class_names),
        dropout=args.dropout,
        use_lora=use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    pretrained_path = args.pretrained.strip()
    if args.mode in {"freeze", "full", "lora"} and not pretrained_path:
        pretrained_path = str(Path(args.ckpt_dir) / "lenet_mnist.ckpt")
    pretrained_report = load_lenet_pretrained(network, pretrained_path) if args.mode != "scratch" else {
        "loaded": 0,
        "skipped": 0,
        "not_loaded": 0,
        "checkpoint_path": "",
    }
    set_lenet_finetune_mode(network, args.mode)
    params = parameter_report(network)
    trainable_names = trainable_parameter_names(network)
    print(f"Dataset: {len(class_names)} classes, train={len(train_samples)}, val={len(val_samples)}")
    print(f"Samples per class: {stats}")
    print(f"Parameters: {params}")
    print(f"Trainable parameters: {trainable_names}")

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)

    log_path = str(Path(args.log_dir) / f"train_{run_tag}.csv")
    if Path(log_path).exists():
        Path(log_path).unlink()
    best_acc = -1.0
    best_path = str(Path(args.ckpt_dir) / f"best_{run_tag}.ckpt")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(network, train_ds, loss_fn, optimizer)
        val_metrics = evaluate(network, val_ds)
        row = {
            "epoch": epoch,
            "mode": args.mode,
            "run_name": run_tag,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "epoch_seconds": round(time.perf_counter() - epoch_start, 2),
            "gpu_memory_mb": gpu_memory_mb(),
            "trainable_params": params["trainable"],
            "strong_aug": int(args.strong_aug),
            "dropout": args.dropout,
        }
        append_csv(log_path, row)
        print(row)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(network, best_path)
            print(f"Saved best checkpoint to {best_path}")

    summary = {
        "mode": args.mode,
        "run_name": run_tag,
        "model": "lenet5",
        "pretrained": pretrained_path,
        "pretrained_report": pretrained_report,
        "input_adapter": "RGB images are converted to grayscale, resized to 32x32, rescaled to [0,1], normalized by MNIST mean/std, and converted from HWC to CHW.",
        "frozen_strategy": {
            "scratch": "no pretrained checkpoint; all LeNet5 parameters are trainable",
            "freeze": "load MNIST pretrained weights; freeze conv1, conv2, fc1 and fc2; train fc3 only",
            "full": "load MNIST pretrained weights; train all layers",
            "lora": "load MNIST pretrained weights; freeze original backbone; train fc1/fc2 LoRA parameters and fc3",
        }[args.mode],
        "lora": {
            "enabled": use_lora,
            "layers": ["fc1", "fc2"] if use_lora else [],
            "rank": args.lora_rank if use_lora else 0,
            "alpha": args.lora_alpha if use_lora else 0,
        },
        "dropout": args.dropout,
        "strong_aug": args.strong_aug,
        "trainable_parameter_names": trainable_names,
        "classes": class_names,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_acc": best_acc,
        "total_seconds": round(time.perf_counter() - start_time, 2),
        "parameters": params,
        "best_ckpt": best_path,
    }
    save_json(str(Path(args.results_dir) / f"summary_{run_tag}.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
