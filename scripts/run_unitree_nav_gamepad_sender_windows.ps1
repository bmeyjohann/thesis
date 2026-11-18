param(
    [int]$Port = $(if ($env:GAMEPAD_PORT) { [int]$env:GAMEPAD_PORT } else { 8794 }),
    [int]$GamepadDeviceIndex = $(if ($env:GAMEPAD_DEVICE_INDEX) { [int]$env:GAMEPAD_DEVICE_INDEX } else { 0 }),
    [double]$DebugHz = $(if ($env:GAMEPAD_DEBUG_HZ) { [double]$env:GAMEPAD_DEBUG_HZ } else { 4.0 }),
    [double]$HotplugRetrySeconds = $(if ($env:GAMEPAD_HOTPLUG_RETRY_SECONDS) { [double]$env:GAMEPAD_HOTPLUG_RETRY_SECONDS } else { 1.0 }),
    [int]$ParentPid = 0,
    [string]$WslHostAddress = '',
    [string]$EndpointFile = $(if ($env:GAMEPAD_ENDPOINT_FILE_WINDOWS) { $env:GAMEPAD_ENDPOINT_FILE_WINDOWS } else { "C:\Data\thesis\local\unitree_gamepad_endpoint.json" })
)

$ErrorActionPreference = "Stop"
$sender = Join-Path $PSScriptRoot "run_safetygym_gamepad_sender.ps1"
if (-not (Test-Path $sender)) {
    throw "Missing Windows gamepad sender: $sender"
}

if (-not $WslHostAddress) {
    $wslAdapter = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.InterfaceAlias -like "vEthernet (WSL*" } |
        Select-Object -First 1
    if (-not $wslAdapter) { throw "Could not discover the Windows WSL adapter address." }
    $WslHostAddress = $wslAdapter.IPAddress
}

$endpointParent = Split-Path -Parent $EndpointFile
New-Item -ItemType Directory -Force -Path $endpointParent | Out-Null
@{
    host = $WslHostAddress
    port = $Port
    generated_at = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json | Set-Content -Path $EndpointFile -Encoding ASCII

Write-Host "Starting standalone Unitree gamepad publisher" -ForegroundColor Cyan
Write-Host "controller: Xbox/SDL device $GamepadDeviceIndex"
Write-Host "hotplug:    retry every $HotplugRetrySeconds s"
Write-Host "Windows:    0.0.0.0:$Port"
Write-Host "WSL target: ${WslHostAddress}:$Port"
Write-Host "endpoint:   $EndpointFile"
Write-Host ""
Write-Host "Keep this window open. In a WSL terminal run:" -ForegroundColor Yellow
Write-Host "  cd /home/benjamin/thesis"
Write-Host "  ./scripts/run_unitree_gamepad_receiver_probe.sh"
Write-Host "Then start collection with:"
Write-Host "  ./scripts/run_unitree_mjlab_nav_human_intervention_local.sh"
Write-Host ""

& $sender `
    -EnvName "SafetyPointGoal1-v0" `
    -ActionDim 3 `
    -ListenHost "0.0.0.0" `
    -Port $Port `
    -GamepadDeviceIndex $GamepadDeviceIndex `
    -DebugHz $DebugHz `
    -HotplugRetrySeconds $HotplugRetrySeconds `
    -ParentPid $ParentPid `
    -NoSavedConfig
