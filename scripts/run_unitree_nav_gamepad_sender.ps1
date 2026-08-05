param(
    [string]$Host = $(if ($env:GAMEPAD_BIND_HOST) { $env:GAMEPAD_BIND_HOST } else { "0.0.0.0" }),
    [int]$Port = $(if ($env:GAMEPAD_PORT) { [int]$env:GAMEPAD_PORT } else { 8794 }),
    [double]$SampleHz = $(if ($env:GAMEPAD_SAMPLE_HZ) { [double]$env:GAMEPAD_SAMPLE_HZ } else { 60.0 }),
    [int]$GamepadDeviceIndex = $(if ($env:GAMEPAD_DEVICE_INDEX) { [int]$env:GAMEPAD_DEVICE_INDEX } else { 0 })
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sender = Join-Path $PSScriptRoot "unitree_nav_gamepad_sender.py"
$requirements = Join-Path $PSScriptRoot "safetygym_gamepad_sender_requirements.txt"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is not installed or not on PATH." }
if (-not (Test-Path $sender)) { throw "Missing sender: $sender" }
if (-not (Test-Path $requirements)) { throw "Missing requirements: $requirements" }

Write-Host "Starting Unitree navigation gamepad publisher" -ForegroundColor Cyan
Write-Host "host:   $Host"
Write-Host "port:   $Port"
Write-Host "hz:     $SampleHz"
Write-Host "device: $GamepadDeviceIndex"
Write-Host "Hold RB to override the policy once the WSL evaluator is running."

Push-Location $repoRoot
try {
    & uv run --with-requirements $requirements python $sender `
        --host $Host `
        --port $Port `
        --sample-hz $SampleHz `
        --gamepad-device-index $GamepadDeviceIndex
}
finally { Pop-Location }

