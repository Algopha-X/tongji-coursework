import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import mindspore.dataset.transforms as transforms
    from mindspore import dtype as mstype
except ImportError:  # Allows local syntax checks outside the MindSpore container.
    ds = None
    vision = None
    transforms = None
    mstype = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageList:
    def __init__(self, samples: Sequence[Tuple[str, int]]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return np.asarray(image), np.int32(label)


def scan_image_folder(data_dir: str) -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class folders found under {root}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    stats = {name: 0 for name in class_names}

    for class_name in class_names:
        class_dir = root / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path), class_to_idx[class_name]))
                stats[class_name] += 1

    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples, class_names, stats


def has_official_split(data_dir: str) -> bool:
    root = Path(data_dir)
    return (root / "train").is_dir() and (root / "test").is_dir()


def _scan_with_class_names(data_dir: Path, class_names: Sequence[str]) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    stats = {name: 0 for name in class_names}

    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path), class_to_idx[class_name]))
                stats[class_name] += 1
    return samples, stats


def scan_official_split(data_dir: str):
    root = Path(data_dir)
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected train/ and test/ under {root}")

    train_classes = {p.name for p in train_root.iterdir() if p.is_dir()}
    test_classes = {p.name for p in test_root.iterdir() if p.is_dir()}
    class_names = sorted(train_classes | test_classes)
    if not class_names:
        raise RuntimeError(f"No class folders found under {train_root} or {test_root}")

    train_samples, train_stats = _scan_with_class_names(train_root, class_names)
    test_samples, test_stats = _scan_with_class_names(test_root, class_names)
    if not train_samples:
        raise RuntimeError(f"No training images found under {train_root}")
    if not test_samples:
        raise RuntimeError(f"No test images found under {test_root}")

    stats = {
        name: {
            "train": train_stats[name],
            "test": test_stats[name],
            "total": train_stats[name] + test_stats[name],
        }
        for name in class_names
    }
    return train_samples, test_samples, class_names, stats


def split_samples(
    samples: Sequence[Tuple[str, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    rng = random.Random(seed)

    for label, items in by_class.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _vision_ops(image_size: int, training: bool, stronger_aug: bool = False):
    if vision is None:
        raise ImportError("MindSpore is required to build datasets.")

    ops = [vision.Resize((image_size, image_size))]
    if training:
        if stronger_aug:
            ops.extend([
                vision.RandomColorAdjust(brightness=0.25, contrast=0.25, saturation=0.2),
                vision.RandomRotation(12),
            ])
        ops.append(vision.RandomHorizontalFlip(prob=0.5))

    ops.extend([
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        vision.HWC2CHW(),
    ])
    return ops


def create_dataset(
    data_dir: str,
    split: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    stronger_aug: bool = False,
):
    if ds is None:
        raise ImportError("MindSpore is required to build datasets.")

    if has_official_split(data_dir):
        train_samples, val_samples, class_names, stats = scan_official_split(data_dir)
    else:
        samples, class_names, stats = scan_image_folder(data_dir)
        train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)

    selected = train_samples if split == "train" else val_samples
    dataset = ds.GeneratorDataset(
        ImageList(selected),
        column_names=["image", "label"],
        shuffle=(split == "train"),
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=_vision_ops(image_size, training=(split == "train"), stronger_aug=stronger_aug),
        input_columns=["image"],
        num_parallel_workers=num_workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(mstype.int32),
        input_columns=["label"],
        num_parallel_workers=num_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=(split == "train"))
    return dataset, class_names, stats, len(train_samples), len(val_samples)


def write_class_index(class_names: Sequence[str], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {str(i): name for i, name in enumerate(class_names)}
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def print_dataset_summary(data_dir: str, val_ratio: float = 0.2, seed: int = 42):
    if has_official_split(data_dir):
        train_samples, val_samples, class_names, stats = scan_official_split(data_dir)
        print("split: official train/test")
        print(f"classes: {len(class_names)}")
        print(f"total samples: {len(train_samples) + len(val_samples)}")
        print(f"train samples: {len(train_samples)}")
        print(f"test samples: {len(val_samples)}")
        for name in class_names:
            item = stats[name]
            print(f"{name}: train={item['train']}, test={item['test']}, total={item['total']}")
        return

    samples, class_names, stats = scan_image_folder(data_dir)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)
    print("split: random train/val")
    print(f"classes: {len(class_names)}")
    print(f"total samples: {len(samples)}")
    print(f"train samples: {len(train_samples)}")
    print(f"val samples: {len(val_samples)}")
    for name in class_names:
        print(f"{name}: {stats[name]}")
