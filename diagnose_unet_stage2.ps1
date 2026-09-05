param(
    [string]$CheckpointRoot = ".\debug\unet\stage2_512_full_5090_offload",
    [string]$ReportPath = "",
    [switch]$TraceInference,
    [string]$VideoPath = "",
    [string]$AudioPath = "",
    [string]$ConfigPath = ".\configs\unet\stage2_512_full_5090_offload.yaml",
    [string]$VideoOutputPath = "",
    [ValidateRange(1, 2147483647)][int]$BaselineStep = 1000,
    [ValidateRange(1, 2147483647)][int]$TargetStep = 1500,
    [ValidateSet("fp16", "fp32")][string]$Dtype = "fp16",
    [double]$GuidanceScale = 1.5
)

$ErrorActionPreference = "Stop"

if ($BaselineStep -ge $TargetStep) {
    throw "BaselineStep must be less than TargetStep."
}
if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $suffix = if ($TraceInference) { $Dtype } else { "weights" }
    $ReportPath = ".\debug\diagnose_unet\report_${BaselineStep}_${TargetStep}_${suffix}.json"
}
if ([string]::IsNullOrWhiteSpace($VideoOutputPath)) {
    $VideoOutputPath = ".\debug\diagnose_unet\checkpoint_${TargetStep}_${Dtype}.mp4"
}

function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Path))
}

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$diagnosticScript = Join-Path $PSScriptRoot "tools\diagnose_unet_checkpoint.py"
$checkpointSearchRoot = (Resolve-ProjectPath $CheckpointRoot)

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $diagnosticScript -PathType Leaf)) {
    throw "Diagnostic script not found: $diagnosticScript"
}
if (-not (Test-Path -LiteralPath $checkpointSearchRoot -PathType Container)) {
    throw "Checkpoint directory not found: $checkpointSearchRoot"
}

$targetCheckpoint = Get-ChildItem -LiteralPath $checkpointSearchRoot -Recurse -File -Filter "checkpoint-$TargetStep.pt" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $targetCheckpoint) {
    throw "No checkpoint-$TargetStep.pt found under: $checkpointSearchRoot"
}

# A resumed run gets a new timestamp directory. Select the newest checkpoint-$BaselineStep
# that existed no later than the selected checkpoint-$TargetStep, which follows that run's
# most likely resume lineage without requiring both files to share a directory.
$baselineCheckpoint = Get-ChildItem -LiteralPath $checkpointSearchRoot -Recurse -File -Filter "checkpoint-$BaselineStep.pt" |
    Where-Object { $_.LastWriteTime -le $targetCheckpoint.LastWriteTime } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $baselineCheckpoint) {
    throw "No checkpoint-$BaselineStep.pt preceding $($targetCheckpoint.FullName) was found."
}

Write-Host "Selected baseline ${BaselineStep}: $($baselineCheckpoint.FullName)"
Write-Host "Selected target ${TargetStep}: $($targetCheckpoint.FullName)"

$arguments = @(
    $diagnosticScript,
    "--checkpoints",
    $baselineCheckpoint.FullName,
    $targetCheckpoint.FullName,
    "--report-json",
    (Resolve-ProjectPath $ReportPath)
)

if ($TraceInference) {
    if ([string]::IsNullOrWhiteSpace($VideoPath) -or [string]::IsNullOrWhiteSpace($AudioPath)) {
        throw "-TraceInference requires both -VideoPath and -AudioPath."
    }

    $arguments += @(
        "--trace-checkpoint",
        $targetCheckpoint.FullName,
        "--unet-config-path",
        (Resolve-ProjectPath $ConfigPath),
        "--video-path",
        (Resolve-ProjectPath $VideoPath),
        "--audio-path",
        (Resolve-ProjectPath $AudioPath),
        "--video-out-path",
        (Resolve-ProjectPath $VideoOutputPath),
        "--inference-steps",
        "20",
        "--guidance-scale",
        $GuidanceScale.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--seed",
        "1247",
        "--dtype",
        $Dtype
    )
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonPath @arguments
    $diagnosticExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($diagnosticExitCode -ne 0) {
    throw "Diagnostic process failed with exit code $diagnosticExitCode"
}

Write-Host "Diagnostic completed: $ReportPath"

