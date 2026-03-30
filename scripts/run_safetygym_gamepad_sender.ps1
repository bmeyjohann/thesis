param(
    [string]$EnvName = $(if ($env:ENV_NAME) { $env:ENV_NAME } else { "SafetyCarGoal2-v0" }),
    [int]$ActionDim = $(if ($env:ACTION_DIM) { [int]$env:ACTION_DIM } else { 2 }),
    [string]$Host = $(if ($env:HOST) { $env:HOST } else { "0.0.0.0" }),
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8793 }),
    [double]$SampleHz = $(if ($env:SAMPLE_HZ) { [double]$env:SAMPLE_HZ } else { 60.0 }),
    [int]$GamepadDeviceIndex = $(if ($env:GAMEPAD_DEVICE_INDEX) { [int]$env:GAMEPAD_DEVICE_INDEX } else { 0 }),
    [string]$GamepadConfigPath = $(if ($env:GAMEPAD_CONFIG_PATH) { $env:GAMEPAD_CONFIG_PATH } else { "$HOME\.config\thesis\safetygym_gamepad\mapping_profile.json" }),
    [switch]$NoSavedConfig
)

$ErrorActionPreference = "Stop"
$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:SenderScript = Join-Path $PSScriptRoot "safetygym_gamepad_sender.py"
$script:Requirements = Join-Path $PSScriptRoot "safetygym_gamepad_sender_requirements.txt"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not on PATH."
}

if (-not (Test-Path $script:SenderScript)) {
    throw "Missing sender script: $script:SenderScript"
}

if (-not (Test-Path $script:Requirements)) {
    throw "Missing requirements file: $script:Requirements"
}

$cmd = @(
    "run",
    "--with-requirements", $script:Requirements,
    "python", $script:SenderScript,
    "--env_name", $EnvName,
    "--action_dim", "$ActionDim",
    "--host", $Host,
    "--port", "$Port",
    "--sample_hz", "$SampleHz",
    "--gamepad_config_path", $GamepadConfigPath,
    "--gamepad_device_index", "$GamepadDeviceIndex"
)

if ($NoSavedConfig) {
    $cmd += "--no_gamepad_use_saved_config"
}

Write-Host "Starting SafetyGym gamepad sender" -ForegroundColor Cyan
Write-Host "repo:   $script:RepoRoot"
Write-Host "env:    $EnvName"
Write-Host "host:   $Host"
Write-Host "port:   $Port"
Write-Host "hz:     $SampleHz"
Write-Host "device: $GamepadDeviceIndex"
if (-not $NoSavedConfig) {
    Write-Host "config: $GamepadConfigPath"
}

Push-Location $script:RepoRoot
try {
    & uv @cmd
}
finally {
    Pop-Location
}
