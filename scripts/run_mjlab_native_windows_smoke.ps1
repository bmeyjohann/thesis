param(
    [ValidateSet("native", "viser")]
    [string]$Viewer = "native",
    [ValidateSet("zero", "random")]
    [string]$Agent = "zero",
    [string]$Task = "Mjlab-Velocity-Flat-Unitree-G1",
    [int]$NumEnvs = 1,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$python = "C:\Users\benja\miniconda3\envs\mjlab_win_native\python.exe"
$warpCache = "C:\home\benjamin\thesis\.tmp_warp_win_native"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing isolated native MJLab environment: $python"
}
New-Item -ItemType Directory -Force -Path $warpCache | Out-Null
$env:PYTHONUTF8 = "1"
$env:WARP_CACHE_PATH = $warpCache

& $python -c "import torch, mjlab, mujoco_warp, viser, warp; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0)); print('This is a native-Windows MJLab smoke task, not the thesis target terrain.')"
if ($LASTEXITCODE -ne 0) {
    throw "Native MJLab dependency probe failed."
}

Write-Host "Starting native-Windows MJLab compatibility test" -ForegroundColor Cyan
Write-Host "task:       $Task"
Write-Host "viewer:     $Viewer"
Write-Host "agent:      $Agent"
Write-Host "device:     $Device"
if ($Viewer -eq "viser") {
    Write-Host "browser:    http://localhost:8080"
}

& $python -m mjlab.scripts.play $Task `
    --agent $Agent `
    --viewer $Viewer `
    --num-envs $NumEnvs `
    --device $Device `
    --no-terminations True
exit $LASTEXITCODE
