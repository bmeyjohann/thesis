param(
    [string]$Distro = "Ubuntu",
    [string]$WslRepo = "/home/benjamin/thesis",
    [string]$LowLevelPolicyPath = "",
    [int]$GamepadPort = 8794,
    [int]$GamepadDeviceIndex = 0,
    [double]$GamepadDebugHz = 4.0,
    [ValidateSet("pygame", "viser")]
    [string]$ViewerBackend = "viser",
    [int]$ViserPort = 8080,
    [ValidateSet("chase", "bird", "free")]
    [string]$ViserCameraMode = "chase",
    [int]$ViserRenderStride = 1,
    [int]$VideoWidth = 192,
    [int]$VideoHeight = 144,
    [int]$RgbEvery = 4
)

$ErrorActionPreference = "Stop"
$sender = Join-Path $PSScriptRoot "run_unitree_nav_gamepad_sender_windows.ps1"
if (-not (Test-Path -LiteralPath $sender)) {
    throw "Missing native Windows gamepad publisher: $sender"
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe is unavailable."
}

$existingListener = Get-NetTCPConnection -LocalPort $GamepadPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($existingListener) {
    throw "Port $GamepadPort is already owned by process $($existingListener.OwningProcess). Close the stale publisher first."
}

$wslAdapter = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.InterfaceAlias -like "vEthernet (WSL*" } |
    Select-Object -First 1
if (-not $wslAdapter) {
    throw "Could not discover the Windows WSL adapter address."
}
$gamepadHost = $wslAdapter.IPAddress

Write-Host "Starting target-terrain manual gamepad evaluation" -ForegroundColor Cyan
Write-Host "locomotion: $(if ($LowLevelPolicyPath) { $LowLevelPolicyPath } else { '<latest gentle checkpoint>' })"
Write-Host "terrain:    balanced; sand drag 120 N/(m/s); ice friction 0.30"
Write-Host "episode:    continuous goals; 3600 s environment horizon"
Write-Host "gamepad:    Windows :$GamepadPort -> WSL ${gamepadHost}:$GamepadPort"
Write-Host "render:     ${VideoWidth}x${VideoHeight}, every $RgbEvery simulation steps"
Write-Host "viewer:     $ViewerBackend$(if ($ViewerBackend -eq 'viser') { " at http://localhost:$ViserPort" } else { '' })"
Write-Host "camera:     $ViserCameraMode; scene update every $ViserRenderStride frame(s)"
Write-Host "training:   disabled"
Write-Host "recording:  disabled"

$senderArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $sender,
    "-Port", "$GamepadPort",
    "-GamepadDeviceIndex", "$GamepadDeviceIndex",
    "-DebugHz", "$GamepadDebugHz"
)
$senderProcess = $null
try {
    $senderProcess = Start-Process -FilePath "powershell.exe" -ArgumentList $senderArgs -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    $ready = $false
    while (-not $ready -and [DateTime]::UtcNow -lt $deadline) {
        if ($senderProcess.HasExited) {
            throw "Gamepad publisher exited during startup. Inspect its PowerShell window."
        }
        $probe = [System.Net.Sockets.TcpClient]::new()
        try {
            $pending = $probe.BeginConnect("127.0.0.1", $GamepadPort, $null, $null)
            if ($pending.AsyncWaitHandle.WaitOne(500)) {
                $probe.EndConnect($pending)
                $ready = $true
            }
        }
        catch {
            # Native Pygame may still be importing.
        }
        finally {
            $probe.Dispose()
        }
        if (-not $ready) { Start-Sleep -Milliseconds 250 }
    }
    if (-not $ready) {
        throw "Gamepad publisher did not listen on localhost:$GamepadPort within 20 seconds."
    }

    & wsl.exe -d $Distro --cd $WslRepo -- env `
        "LOW_LEVEL_POLICY_PATH=$LowLevelPolicyPath" `
        "GAMEPAD_HOST=$gamepadHost" `
        "GAMEPAD_PORT=$GamepadPort" `
        "VIEWER_BACKEND=$ViewerBackend" `
        "VISER_PORT=$ViserPort" `
        "VISER_CAMERA_MODE=$ViserCameraMode" `
        "VISER_RENDER_STRIDE=$ViserRenderStride" `
        "VIDEO_WIDTH=$VideoWidth" `
        "VIDEO_HEIGHT=$VideoHeight" `
        "RGB_EVERY=$RgbEvery" `
        ./scripts/run_unitree_target_terrain_manual_gamepad_local.sh
    if ($LASTEXITCODE -ne 0) {
        throw "WSL evaluator exited with code $LASTEXITCODE"
    }
}
finally {
    if ($senderProcess -and -not $senderProcess.HasExited) {
        Start-Process -FilePath "$env:SystemRoot\System32\taskkill.exe" `
            -ArgumentList @("/PID", "$($senderProcess.Id)", "/T", "/F") `
            -WindowStyle Hidden `
            -Wait
    }
}
