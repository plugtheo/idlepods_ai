#Requires -Version 5.1
<#
.SYNOPSIS
  One-time bootstrap: build the training image and train LoRA adapters on the GPU.

.DESCRIPTION
  Builds the training Docker image then runs train_gpu_simple.py inside it with
  GPU access and internet (to download the Qwen3-8B base model + HF datasets).
  Any extra arguments are passed through to train_gpu_simple.py.

.EXAMPLE
  .\run_bootstrap.ps1                                 # train all 6 capabilities
  .\run_bootstrap.ps1 --capability coding             # single capability
  .\run_bootstrap.ps1 --capability coding --validate  # train + smoke test
  .\run_bootstrap.ps1 --validate                      # all + validate each
  .\run_bootstrap.ps1 --fresh --validate              # re-train from scratch

.NOTES
  Prerequisites:
    - Docker Desktop with WSL2 GPU support (NVIDIA Container Toolkit)
    - $env:HF_TOKEN set to your HuggingFace token (needed for Qwen3-8B)
    - ~20 GB free disk space for model weights + datasets

  Outputs:   data\lora_checkpoints\<capability>_lora\
  Datasets:  data\training_data_curated\
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$Image       = "idlepods-training:bootstrap"

# Ensure output directories exist before bind-mounting (Docker creates missing
# bind-mount targets as directories, which would make them root-owned).
New-Item -ItemType Directory -Force "$ProjectRoot\data\lora_checkpoints"   | Out-Null
New-Item -ItemType Directory -Force "$ProjectRoot\data\training_data_curated" | Out-Null

Write-Host "==> Building training image (context: $ProjectRoot)..."
docker build --tag $Image --file "$ProjectRoot\training\Dockerfile" $ProjectRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$HfCache = if ($env:HF_CACHE_DIR) { $env:HF_CACHE_DIR } else { "$env:USERPROFILE\.cache\huggingface" }
$HfToken = if ($env:HF_TOKEN)     { $env:HF_TOKEN }     else { "" }

$RunArgs = @(
    "run", "--rm",
    "--runtime=nvidia",
    "--gpus", "all",
    "-v", "${ProjectRoot}\data:/data",
    "-v", "${ProjectRoot}\data\lora_checkpoints:/data/lora_checkpoints",
    "-v", "${HfCache}:/root/.cache/huggingface",
    "-v", "${ProjectRoot}\models.yaml:/config/models.yaml:ro",
    "-e", "HF_TOKEN=$HfToken",
    "-e", "TRAINING__OUTPUT_DIR=/data/lora_checkpoints",
    "-e", "MODELS_YAML_PATH=/config/models.yaml",
    "-e", "HF_HUB_OFFLINE=0",
    "-e", "TRANSFORMERS_OFFLINE=0"
)

if (Test-Path "$ProjectRoot\recipes.yaml") {
    $RunArgs += "-v", "${ProjectRoot}\recipes.yaml:/config/recipes.yaml:ro"
    $RunArgs += "-e", "RECIPES_YAML_PATH=/config/recipes.yaml"
}

$RunArgs += $Image
$RunArgs += "python", "/app/training/bootstrap/train_gpu_simple.py"
$RunArgs += $args   # pass-through any CLI flags (--capability, --validate, etc.)

Write-Host ""
Write-Host "==> Running bootstrap training (GPU)..."
docker @RunArgs
exit $LASTEXITCODE
