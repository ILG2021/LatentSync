param(
    [string]$CheckpointRoot = ".\debug\unet\stage2_512_full_5090_offload",
    [string]$ReportPath = ".\debug\diagnose_unet\weights_report.json",
    [switch]$TraceInference,
    [string]$VideoPath = "",
    [string]$AudioPath = "",
    [string]$ConfigPath = ".\configs\unet\stage2_512_full_5090_offload.yaml",
    [string]$VideoOutputPath = ".\debug\diagnose_unet\checkpoint1000.mp4"
)

$ErrorActionPreference = "Stop"

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$diagnosticScript = Join-Path $PSScriptRoot "tools\diagnose_unet_checkpoint.py"
$checkpointSearchRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $CheckpointRoot))

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $diagnosticScript -PathType Leaf)) {
    throw "Diagnostic script not found: $diagnosticScript"
}
if (-not (Test-Path -LiteralPath $checkpointSearchRoot -PathType Container)) {
    throw "Checkpoint directory not found: $checkpointSearchRoot"
}

$checkpoint1000 = Get-ChildItem -LiteralPath $checkpointSearchRoot -Recurse -File -Filter "checkpoint-1000.pt" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $checkpoint1000) {
    throw "No checkpoint-1000.pt found under: $checkpointSearchRoot"
}

# A resumed run gets a new timestamp directory. Select the newest checkpoint-500
# that existed no later than the selected checkpoint-1000, which follows that run's
# most likely resume lineage without requiring both files to share a directory.
$checkpoint500 = Get-ChildItem -LiteralPath $checkpointSearchRoot -Recurse -File -Filter "checkpoint-500.pt" |
    Where-Object { $_.LastWriteTime -le $checkpoint1000.LastWriteTime } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $checkpoint500) {
    throw "No checkpoint-500.pt preceding $($checkpoint1000.FullName) was found."
}

Write-Host "Selected checkpoint 500 : $($checkpoint500.FullName)"
Write-Host "Selected checkpoint 1000: $($checkpoint1000.FullName)"

$arguments = @(
    $diagnosticScript,
    "--checkpoints",
    $checkpoint500.FullName,
    $checkpoint1000.FullName,
    "--report-json",
    [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $ReportPath))
)

if ($TraceInference) {
    if ([string]::IsNullOrWhiteSpace($VideoPath) -or [string]::IsNullOrWhiteSpace($AudioPath)) {
        throw "-TraceInference requires both -VideoPath and -AudioPath."
    }

    $arguments += @(
        "--trace-checkpoint",
        $checkpoint1000.FullName,
        "--unet-config-path",
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $ConfigPath)),
        "--video-path",
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $VideoPath)),
        "--audio-path",
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $AudioPath)),
        "--video-out-path",
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $VideoOutputPath)),
        "--inference-steps",
        "20",
        "--guidance-scale",
        "1.0",
        "--seed",
        "1247",
        "--dtype",
        "fp16"
    )
}

& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Diagnostic process failed with exit code $LASTEXITCODE"
}

Write-Host "Diagnostic completed: $ReportPath"
