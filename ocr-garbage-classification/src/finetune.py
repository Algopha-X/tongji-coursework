import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run comparable fine-tuning experiments")
    parser.add_argument("--data_dir", default="data/garbage")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args appended to train.py")
    return parser.parse_args()


def run(mode: str, args):
    cmd = [
        sys.executable,
        "src/train.py",
        "--mode",
        mode,
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--image_size",
        str(args.image_size),
        "--lr",
        str(args.lr),
    ]
    if args.pretrained:
        cmd.extend(["--pretrained", args.pretrained])
    cmd.extend(args.extra)
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    args = parse_args()
    for mode in ["freeze", "full", "lora"]:
        run(mode, args)


if __name__ == "__main__":
    main()
