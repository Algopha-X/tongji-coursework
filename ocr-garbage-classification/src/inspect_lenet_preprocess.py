import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore import dtype as mstype

from src.train_lenet_garbage import GrayImageList, collect_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Save a contact sheet of LeNet5 garbage preprocessing outputs")
    parser.add_argument("--data_dir", default="data/garbage_official")
    parser.add_argument("--out", default="results/preprocess_lenet_preview.png")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


def make_dataset(samples, image_size, stronger_aug, num_images):
    image_ops = [vision.Resize((image_size, image_size))]
    if stronger_aug:
        image_ops.extend([
            vision.RandomRotation(12),
            vision.RandomHorizontalFlip(prob=0.5),
        ])
    image_ops.extend([
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW(),
    ])

    dataset = ds.GeneratorDataset(
        GrayImageList(samples[:num_images]),
        column_names=["image", "label"],
        shuffle=False,
        num_parallel_workers=1,
    )
    dataset = dataset.map(operations=image_ops, input_columns=["image"], num_parallel_workers=1)
    dataset = dataset.map(operations=transforms.TypeCast(mstype.int32), input_columns=["label"], num_parallel_workers=1)
    return dataset.batch(num_images, drop_remainder=False)


def denormalize(batch):
    images = batch.asnumpy()
    images = images[:, 0, :, :] * 0.3081 + 0.1307
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    return images


def draw_row(canvas, images, labels, class_names, y, title, cell):
    draw = ImageDraw.Draw(canvas)
    draw.text((8, y + 8), title, fill=(0, 0, 0))
    x0 = 150
    for idx, image in enumerate(images):
        tile = Image.fromarray(image, mode="L").resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
        x = x0 + idx * (cell + 18)
        canvas.paste(tile, (x, y))
        label = class_names[int(labels[idx])]
        draw.text((x, y + cell + 4), label[:14], fill=(0, 0, 0))


def main():
    args = parse_args()
    train_samples, _, class_names, _ = collect_samples(args.data_dir, args.val_ratio, args.seed)
    selected = train_samples[:args.num_images]
    base_ds = make_dataset(selected, args.image_size, False, args.num_images)
    aug_ds = make_dataset(selected, args.image_size, True, args.num_images)

    base_images, base_labels = next(base_ds.create_tuple_iterator())
    aug_images, aug_labels = next(aug_ds.create_tuple_iterator())
    base_np = denormalize(base_images)
    aug_np = denormalize(aug_images)

    cell = 96
    width = 150 + args.num_images * (cell + 18)
    height = 2 * (cell + 40) + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw_row(canvas, base_np, base_labels.asnumpy(), class_names, 12, "baseline", cell)
    draw_row(canvas, aug_np, aug_labels.asnumpy(), class_names, 12 + cell + 58, "strong_aug", cell)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved preprocessing preview to {out}")


if __name__ == "__main__":
    main()
