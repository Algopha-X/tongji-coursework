import argparse
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import save_checkpoint

from src.dataset import create_dataset, write_class_index
from src.model import build_model
from src.utils import (
    append_csv,
    build_lr,
    ensure_dir,
    evaluate_network,
    gpu_memory_mb,
    load_pretrained_flexible,
    parameter_report,
    save_json,
    set_finetune_mode,
    set_seed,
    setup_context,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MindSpore garbage classification training")
    parser.add_argument("--data_dir", default="data/garbage", help="ImageFolder style dataset root")
    parser.add_argument("--mode", default="freeze", choices=["freeze", "full", "lora"])
    parser.add_argument("--run_name", default="", help="Optional suffix for ckpt/log/summary files")
    parser.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "smallcnn"])
    parser.add_argument("--pretrained", default="", help="Optional MindSpore checkpoint path")
    parser.add_argument("--pretrained_strategy", default="name", choices=["name", "shape"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine", "step"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--strong_aug", action="store_true", help="Use stronger augmentation for improvement experiment")
    parser.add_argument("--device_target", default="GPU")
    parser.add_argument("--context_mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


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
    run_tag = args.run_name.strip() or args.mode
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
        stronger_aug=args.strong_aug,
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

    network = build_model(
        args.model,
        num_classes=len(class_names),
        dropout=args.dropout,
        use_lora=(args.mode == "lora"),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    pretrained_report = load_pretrained_flexible(
        network,
        args.pretrained,
        strategy=args.pretrained_strategy,
    )
    set_finetune_mode(network, args.mode)
    params = parameter_report(network)
    print(f"Dataset: {len(class_names)} classes, train={train_count}, val={val_count}")
    print(f"Samples per class: {stats}")
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

    log_path = str(Path(args.log_dir) / f"train_{run_tag}.csv")
    best_acc = -1.0
    best_path = str(Path(args.ckpt_dir) / f"best_{run_tag}.ckpt")
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
        "run_name": run_tag,
        "model": args.model,
        "classes": class_names,
        "train_samples": train_count,
        "val_samples": val_count,
        "best_val_acc": best_acc,
        "total_seconds": round(time.perf_counter() - start_time, 2),
        "parameters": params,
        "pretrained": args.pretrained,
        "pretrained_strategy": args.pretrained_strategy,
        "pretrained_report": pretrained_report,
        "lr_schedule": args.lr_schedule,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "best_ckpt": best_path,
    }
    save_json(str(Path(args.results_dir) / f"summary_{run_tag}.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
