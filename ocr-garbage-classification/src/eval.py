import argparse
from pathlib import Path

from mindspore import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.model import build_model
from src.utils import evaluate_network, save_json, set_seed, setup_context


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate garbage classification checkpoint")
    parser.add_argument("--data_dir", default="data/garbage")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "smallcnn"])
    parser.add_argument("--mode", default="full", choices=["freeze", "full", "lora"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--device_target", default="GPU")
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_context(args.device_target)
    val_ds, class_names, _, _, val_count = create_dataset(
        args.data_dir,
        split="val",
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    network = build_model(
        args.model,
        num_classes=len(class_names),
        dropout=args.dropout,
        use_lora=(args.mode == "lora"),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    params = load_checkpoint(args.ckpt)
    load_param_into_net(network, params, strict_load=False)
    metrics = evaluate_network(network, val_ds)
    result = {"ckpt": args.ckpt, "val_samples": val_count, "val_acc": metrics["acc"]}
    print(result)
    save_json(str(Path(args.results_dir) / "eval_result.json"), result)


if __name__ == "__main__":
    main()
