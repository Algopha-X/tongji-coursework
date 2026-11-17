import argparse
import ssl
import urllib.request
from pathlib import Path


PRETRAINED_URLS = {
    "mindvision_imagenet": "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt",
    "official_openimage_cpu_gpu": "https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2_cpu_gpu.ckpt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download MobileNetV2 pretrained checkpoint")
    parser.add_argument(
        "--source",
        default="mindvision_imagenet",
        choices=sorted(PRETRAINED_URLS),
        help="Pretrained checkpoint source",
    )
    parser.add_argument("--out", default="checkpoints/pretrained_mobilenetv2.ckpt")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    return parser.parse_args()


def main():
    args = parse_args()
    url = PRETRAINED_URLS[args.source]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.source}")
    print(url)
    print(f"Saving to {out}")
    if args.insecure:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response:
            out.write_bytes(response.read())
    else:
        urllib.request.urlretrieve(url, out)
    print(f"Done: {out} ({out.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
