import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.log, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, marker="o", label="train")
    plt.plot(epochs, val_acc, marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy")
    plt.legend()

    out = args.out or str(Path("results") / (Path(args.log).stem + ".png"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved curve to {out}")


if __name__ == "__main__":
    main()
