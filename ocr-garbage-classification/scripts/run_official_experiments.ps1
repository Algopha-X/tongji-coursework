$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..")

$CkptUrl = "https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_100-d5532038.ckpt"
$CkptPath = "checkpoints/mindcv_mobilenet_v2_100.ckpt"

function Run-Step {
  param(
    [string]$Name,
    [scriptblock]$Command
  )
  Write-Host "== $Name =="
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name (exit code $LASTEXITCODE)"
  }
}

Run-Step "Checking MindCV" {
  docker compose run --rm mindspore-gpu python -m pip show mindcv
}

if (-not (Test-Path $CkptPath)) {
  Run-Step "Downloading MindCV MobileNetV2-1.0 checkpoint" {
    docker compose run --rm mindspore-gpu python -c "import urllib.request; urllib.request.urlretrieve('$CkptUrl', '$CkptPath'); print('download ok')"
  }
}

Run-Step "Smoke test: MobileNetV2-1.0 checkpoint loading" {
  docker compose run --rm mindspore-gpu python src/train_official_mobilenet.py `
    --data_dir data/garbage_official `
    --model_name mobilenet_v2_100_224 `
    --checkpoint_path $CkptPath `
    --mode freeze `
    --run_name mindcv_mobilenet_v2_100_smoke `
    --epochs 1 `
    --batch_size 16 `
    --lr 1e-3
}

Run-Step "MindCV MobileNetV2-1.0: freeze" {
  docker compose run --rm mindspore-gpu python src/train_official_mobilenet.py `
    --data_dir data/garbage_official `
    --model_name mobilenet_v2_100_224 `
    --checkpoint_path $CkptPath `
    --mode freeze `
    --run_name mindcv_mobilenet_v2_100_freeze `
    --epochs 24 `
    --batch_size 16 `
    --lr 1e-3
}

Run-Step "MindCV MobileNetV2-1.0: full" {
  docker compose run --rm mindspore-gpu python src/train_official_mobilenet.py `
    --data_dir data/garbage_official `
    --model_name mobilenet_v2_100_224 `
    --checkpoint_path $CkptPath `
    --mode full `
    --run_name mindcv_mobilenet_v2_100_full `
    --epochs 24 `
    --batch_size 16 `
    --lr 1e-4
}

Run-Step "MindCV MobileNetV2-1.0: full + cosine" {
  docker compose run --rm mindspore-gpu python src/train_official_mobilenet.py `
    --data_dir data/garbage_official `
    --model_name mobilenet_v2_100_224 `
    --checkpoint_path $CkptPath `
    --mode full `
    --run_name mindcv_mobilenet_v2_100_full_cosine `
    --epochs 24 `
    --batch_size 16 `
    --lr 1e-4 `
    --lr_schedule cosine `
    --min_lr 1e-6 `
    --warmup_epochs 2
}

Run-Step "MindCV MobileNetV2-1.0: LoRA" {
  docker compose run --rm mindspore-gpu python src/train_official_mobilenet.py `
    --data_dir data/garbage_official `
    --model_name mobilenet_v2_100_224 `
    --checkpoint_path $CkptPath `
    --mode lora `
    --run_name mindcv_mobilenet_v2_100_lora_real `
    --epochs 24 `
    --batch_size 16 `
    --lr 1e-3 `
    --lora_rank 4 `
    --lora_alpha 8 `
    --lora_max_layers 12
}

Run-Step "Plot full cosine curve" {
  docker compose run --rm mindspore-gpu python src/plot_curves.py `
    --log logs/train_mindcv_mobilenet_v2_100_full_cosine.csv `
    --out results/curve_mindcv_mobilenet_v2_100_full_cosine.png
}

Run-Step "Plot LoRA curve" {
  docker compose run --rm mindspore-gpu python src/plot_curves.py `
    --log logs/train_mindcv_mobilenet_v2_100_lora_real.csv `
    --out results/curve_mindcv_mobilenet_v2_100_lora_real.png
}

Write-Host "== Summaries =="
Get-Content results\summary_mindcv_mobilenet_v2_100_freeze.json -ErrorAction Stop
Get-Content results\summary_mindcv_mobilenet_v2_100_full.json -ErrorAction Stop
Get-Content results\summary_mindcv_mobilenet_v2_100_full_cosine.json -ErrorAction Stop
Get-Content results\summary_mindcv_mobilenet_v2_100_lora_real.json -ErrorAction Stop
