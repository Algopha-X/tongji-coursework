$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..")

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

$CommonGarbageArgs = @(
  "--data_dir", "data/garbage_official",
  "--pretrained", "checkpoints/lenet_mnist.ckpt",
  "--epochs", "24",
  "--batch_size", "32",
  "--image_size", "32",
  "--lr", "1e-3",
  "--weight_decay", "1e-4"
)

Run-Step "LeNet5 + MNIST from scratch" {
  docker compose run --rm mindspore-gpu python src/train_mnist_lenet.py `
    --run_name lenet_mnist `
    --epochs 5 `
    --batch_size 64 `
    --image_size 32 `
    --lr 1e-3
}

Run-Step "Plot MNIST curve" {
  docker compose run --rm mindspore-gpu python src/plot_curves.py `
    --log logs/train_lenet_mnist.csv `
    --out results/curve_lenet_mnist.png
}

Run-Step "LeNet5 + garbage from scratch" {
  docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py `
    --data_dir data/garbage_official `
    --mode scratch `
    --run_name lenet_garbage_scratch `
    --epochs 24 `
    --batch_size 32 `
    --image_size 32 `
    --lr 1e-3 `
    --weight_decay 1e-4
}

Run-Step "LeNet5 + MNIST pretrained + frozen garbage fine-tune" {
  docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py `
    @CommonGarbageArgs `
    --mode freeze `
    --run_name lenet_garbage_freeze
}

Run-Step "LeNet5 + MNIST pretrained + full garbage fine-tune" {
  docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py `
    @CommonGarbageArgs `
    --mode full `
    --run_name lenet_garbage_full
}

Run-Step "LeNet5 + MNIST pretrained + LoRA garbage fine-tune" {
  docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py `
    @CommonGarbageArgs `
    --mode lora `
    --run_name lenet_garbage_lora `
    --lora_rank 4 `
    --lora_alpha 8
}

Run-Step "LeNet5 improvement experiment: full fine-tune + stronger augmentation" {
  docker compose run --rm mindspore-gpu python src/train_lenet_garbage.py `
    @CommonGarbageArgs `
    --mode full `
    --run_name lenet_garbage_full_aug `
    --strong_aug
}

$CurveRuns = @(
  "lenet_garbage_scratch",
  "lenet_garbage_freeze",
  "lenet_garbage_full",
  "lenet_garbage_lora",
  "lenet_garbage_full_aug"
)

foreach ($Run in $CurveRuns) {
  Run-Step "Plot $Run curve" {
    docker compose run --rm mindspore-gpu python src/plot_curves.py `
      --log "logs/train_$Run.csv" `
      --out "results/curve_$Run.png"
  }
}

Write-Host "== Summaries =="
$Summaries = @(
  "results/summary_lenet_mnist.json",
  "results/summary_lenet_garbage_scratch.json",
  "results/summary_lenet_garbage_freeze.json",
  "results/summary_lenet_garbage_full.json",
  "results/summary_lenet_garbage_lora.json",
  "results/summary_lenet_garbage_full_aug.json"
)
foreach ($Summary in $Summaries) {
  Write-Host "-- $Summary --"
  Get-Content $Summary -ErrorAction Stop
}
