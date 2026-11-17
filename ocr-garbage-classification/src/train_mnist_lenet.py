import argparse
import gzip
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore import save_checkpoint

from src.model import LeNet5
from src.utils import (
    append_csv,
    ensure_dir,
    gpu_memory_mb,
    parameter_report,
    save_json,
    set_seed,
    setup_context,
)


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LeNet5 on MNIST and save a pretrained checkpoint")
    parser.add_argument("--data_dir", default="data/mnist")
    parser.add_argument("--run_name", default="lenet_mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device_target", default="GPU")
    parser.add_argument("--context_mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


def download_mnist(root: Path):
    ensure_dir(str(root))
    for name, url in MNIST_URLS.items():
        target = root / f"{name}.gz"
        if target.exists():
            continue
        print(f"Downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)


def read_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols, 1)


def read_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int32)


class ArrayDataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def create_mnist_dataset(root: str, split: str, image_size: int, batch_size: int, num_workers: int):
    root_path = Path(root)
    download_mnist(root_path)
    if split == "train":
        images = read_images(root_path / "train_images.gz")
        labels = read_labels(root_path / "train_labels.gz")
    else:
        images = read_images(root_path / "test_images.gz")
        labels = read_labels(root_path / "test_labels.gz")

    dataset = ds.GeneratorDataset(
        ArrayDataset(images, labels),
        column_names=["image", "label"],
        shuffle=(split == "train"),
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=[
            vision.Resize((image_size, image_size)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=(0.1307,), std=(0.3081,)),
            vision.HWC2CHW(),
        ],
        input_columns=["image"],
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(mstype.int32),
        input_columns=["label"],
        num_parallel_workers=num_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=(split == "train"))
    return dataset, len(labels)


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
    set_seed(args.seed)
    setup_context(args.device_target, args.context_mode)
    ensure_dir(args.ckpt_dir)
    ensure_dir(args.log_dir)
    ensure_dir(args.results_dir)

    train_ds, train_count = create_mnist_dataset(args.data_dir, "train", args.image_size, args.batch_size, args.num_workers)
    test_ds, test_count = create_mnist_dataset(args.data_dir, "test", args.image_size, args.batch_size, args.num_workers)
    network = LeNet5(num_classes=10, dropout=args.dropout)
    params = parameter_report(network)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)

    log_path = str(Path(args.log_dir) / f"train_{args.run_name}.csv")
    if Path(log_path).exists():
        Path(log_path).unlink()
    best_acc = -1.0
    best_path = str(Path(args.ckpt_dir) / "lenet_mnist.ckpt")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(network, train_ds, loss_fn, optimizer)
        test_metrics = evaluate(network, test_ds)
        row = {
            "epoch": epoch,
            "mode": "scratch",
            "run_name": args.run_name,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "val_acc": round(test_metrics["acc"], 6),
            "epoch_seconds": round(time.perf_counter() - epoch_start, 2),
            "gpu_memory_mb": gpu_memory_mb(),
            "trainable_params": params["trainable"],
            "dropout": args.dropout,
        }
        append_csv(log_path, row)
        print(row)
        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            save_checkpoint(network, best_path)
            print(f"Saved best checkpoint to {best_path}")

    summary = {
        "run_name": args.run_name,
        "model": "lenet5",
        "dataset": "mnist",
        "train_samples": train_count,
        "val_samples": test_count,
        "best_val_acc": best_acc,
        "total_seconds": round(time.perf_counter() - start_time, 2),
        "parameters": params,
        "dropout": args.dropout,
        "best_ckpt": best_path,
    }
    save_json(str(Path(args.results_dir) / f"summary_{args.run_name}.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
