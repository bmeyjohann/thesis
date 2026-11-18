param(
    [string]$EnvName = $(if ($env:ENV_NAME) { $env:ENV_NAME } else { "SafetyCarGoal2-v0" }),
    [int]$ActionDim = $(if ($env:ACTION_DIM) { [int]$env:ACTION_DIM } else { 2 }),
    # `$Host` is a read-only automatic PowerShell variable, so do not use it
    # as a parameter name even though it is the network term used by Python.
    [string]$ListenHost = $(if ($env:HOST) { $env:HOST } else { "0.0.0.0" }),
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8793 }),
    [double]$SampleHz = $(if ($env:SAMPLE_HZ) { [double]$env:SAMPLE_HZ } else { 60.0 }),
    [double]$DebugHz = $(if ($env:DEBUG_HZ) { [double]$env:DEBUG_HZ } else { 2.0 }),
    [double]$DebugAxisThreshold = $(if ($env:DEBUG_AXIS_THRESHOLD) { [double]$env:DEBUG_AXIS_THRESHOLD } else { 0.03 }),
    [double]$HotplugRetrySeconds = $(if ($env:GAMEPAD_HOTPLUG_RETRY_SECONDS) { [double]$env:GAMEPAD_HOTPLUG_RETRY_SECONDS } else { 1.0 }),
    [int]$GamepadDeviceIndex = $(if ($env:GAMEPAD_DEVICE_INDEX) { [int]$env:GAMEPAD_DEVICE_INDEX } else { 0 }),
    [string]$GamepadConfigPath = $(if ($env:GAMEPAD_CONFIG_PATH) { $env:GAMEPAD_CONFIG_PATH } else { "$HOME\.config\thesis\safetygym_gamepad\mapping_profile.json" }),
    [string]$PythonExe = $(if ($env:UNITREE_GAMEPAD_PYTHON) { $env:UNITREE_GAMEPAD_PYTHON } else { "C:\Users\benja\miniconda3\envs\thesis\python.exe" }),
    [switch]$NoSavedConfig
)

$ErrorActionPreference = "Stop"
$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:SenderScript = Join-Path $PSScriptRoot "safetygym_gamepad_sender.py"
if (-not (Test-Path $script:SenderScript)) {
    throw "Missing sender script: $script:SenderScript"
}
if (-not (Test-Path $PythonExe)) {
    throw "Windows thesis Python environment not found: $PythonExe"
}

$cmd = @(
    $script:SenderScript,
    "--env_name", $EnvName,
    "--action_dim", "$ActionDim",
    "--host", $ListenHost,
    "--port", "$Port",
    "--sample_hz", "$SampleHz",
    "--debug_hz", "$DebugHz",
    "--debug_axis_threshold", "$DebugAxisThreshold",
    "--hotplug_retry_seconds", "$HotplugRetrySeconds",
    "--gamepad_config_path", $GamepadConfigPath,
    "--gamepad_device_index", "$GamepadDeviceIndex"
)

if ($NoSavedConfig) {
    $cmd += "--no_gamepad_use_saved_config"
}

Write-Host "Starting SafetyGym gamepad sender" -ForegroundColor Cyan
Write-Host "repo:   $script:RepoRoot"
Write-Host "python: $PythonExe"
Write-Host "env:    $EnvName"
Write-Host "host:   $ListenHost"
Write-Host "port:   $Port"
Write-Host "hz:     $SampleHz"
Write-Host "debug:  $DebugHz Hz (axes >= $DebugAxisThreshold)"
Write-Host "hotplug: re-enumerate every $HotplugRetrySeconds s while disconnected"
Write-Host "device: $GamepadDeviceIndex"
if (-not $NoSavedConfig) {
    Write-Host "config: $GamepadConfigPath"
}

Push-Location $script:RepoRoot
try {
    & $PythonExe @cmd
}
finally {
    Pop-Location
}
