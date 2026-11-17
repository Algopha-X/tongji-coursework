import argparse
from pathlib import Path

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

from src.train_lenet_garbage import collect_samples, create_gray_dataset
from src.model import LeNet5
from src.utils import evaluate_network, parameter_report, save_json, setup_context


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LeNet5 checkpoint on garbage classification")
    parser.add_argument("--data_dir", default="data/garbage_official")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_classes", type=int, default=26)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device_target", default="GPU")
    parser.add_argument("--context_mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_context(args.device_target, args.context_mode)
    _, val_samples, class_names, _ = collect_samples(args.data_dir, args.val_ratio, args.seed)
    dataset = create_gray_dataset(val_samples, "val", args.image_size, args.batch_size, args.num_workers)

    network = LeNet5(
        num_classes=args.num_classes or len(class_names),
        dropout=args.dropout,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    params = load_checkpoint(args.ckpt)
    not_loaded = load_param_into_net(network, params, strict_load=False)
    metrics = evaluate_network(network, dataset)
    result = {
        "model": "lenet5",
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "classes": class_names,
        "val_samples": len(val_samples),
        "metrics": metrics,
        "parameters": parameter_report(network),
        "not_loaded": [str(item) for item in not_loaded] if not_loaded else [],
    }
    out = args.out or str(Path("results") / f"eval_{Path(args.ckpt).stem}.json")
    save_json(out, result)
    print(result)


if __name__ == "__main__":
    main()
