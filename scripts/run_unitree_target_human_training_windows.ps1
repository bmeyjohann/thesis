param(
    [ValidateSet('pref_bc_rl', 'pref_rl', 'bc_rl', 'bc_only', 'pref_only', 'rl_only')]
    [string]$Mode = 'pref_bc_rl',
    [int]$Steps = 4500,
    [ValidateSet('height_scan', 'rgb', 'rgb_height', 'depth', 'depth_height', 'rgb_depth', 'stereo_rgb', 'stereo_rgb_height', 'php_depth')]
    [string]$Sensors = 'height_scan',
    [string]$EncoderCheckpoint = '',
    [ValidateSet('legacy', 'stabilized_goal_v1', 'stabilized_goal_v2', 'stabilized_goal_v3')]
    [string]$ImageProfile = 'stabilized_goal_v3',
    [int]$Seed = 7,
    [ValidatePattern('^[A-Za-z0-9_-]*$')]
    [string]$RunName = '',
    [string]$ReceiptPath = '',
    [switch]$StudyProfile,
    [switch]$CheckSenderOnly
)
$ErrorActionPreference = 'Stop'
function Receive-UnitreeStartupPacket {
    param([string]$Address = '127.0.0.1', [int]$Port = 8794, [int]$TimeoutMs = 1000,
          [scriptblock]$ClientFactory = { [Net.Sockets.TcpClient]::new() })
    $client = & $ClientFactory
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        $connect = $client.ConnectAsync($Address, $Port)
        if (-not $connect.Wait($TimeoutMs)) { return $null }
        $remaining = $TimeoutMs - [int]$timer.ElapsedMilliseconds
        if ($remaining -le 0) { return $null }
        $reader = [IO.StreamReader]::new($client.GetStream())
        $read = $reader.ReadLineAsync()
        if (-not $read.Wait($remaining)) { return $null }
        return ($read.Result | ConvertFrom-Json)
    } catch { return $null }
    finally { $client.Dispose() }
}
$publisher = Join-Path $PSScriptRoot 'run_unitree_nav_gamepad_sender_windows.ps1'
if (-not (Test-Path -LiteralPath $publisher)) { throw "Missing publisher: $publisher" }
if (Get-NetTCPConnection -LocalPort 8794 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8794 is already in use. Close the previous gamepad sender first.'
}
Write-Host '[startup 1/3] Checking WSL and receipt path...'
& wsl.exe -d Ubuntu --exec true
if ($LASTEXITCODE -ne 0) { throw 'Could not start WSL.' }
# Direct execution preserves Windows backslashes and environment values with spaces.
$linuxReceipt = ''
if ($ReceiptPath) {
    $linuxReceipt = & wsl.exe -d Ubuntu --exec wslpath -u ([IO.Path]::GetFullPath($ReceiptPath))
    if ($LASTEXITCODE -ne 0 -or -not $linuxReceipt) { throw 'Cannot resolve receipt path in WSL.' }
    $linuxReceipt = $linuxReceipt.Trim()
}
$adapter = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -like 'vEthernet (WSL*' } | Select-Object -First 1
if (-not $adapter) { throw 'Windows WSL network adapter was not found.' }
$logRoot = Join-Path (Split-Path -Parent $PSScriptRoot) ('local\unitree_gamepad_launcher_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + "_$PID")
[void][IO.Directory]::CreateDirectory($logRoot)
$senderOut = Join-Path $logRoot 'sender.stdout.log'
$senderErr = Join-Path $logRoot 'sender.stderr.log'
$senderProcess = Start-Process powershell.exe -PassThru -WindowStyle Hidden -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $publisher + '"'), '-ParentPid', "$PID",
    '-WslHostAddress', "$($adapter.IPAddress)"
) -RedirectStandardOutput $senderOut -RedirectStandardError $senderErr
try {
    Write-Host '[startup 2/3] Waiting for a gamepad JSON packet (at most 30 seconds)...'
    $startupTimer = [Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    while ($startupTimer.Elapsed.TotalSeconds -lt 30 -and -not $ready) {
        $senderProcess.Refresh()
        if ($senderProcess.HasExited) { throw "Gamepad publisher exited during startup (code $($senderProcess.ExitCode))." }
        $remaining = [Math]::Max(1, [Math]::Min(1000, 30000 - [int]$startupTimer.ElapsedMilliseconds))
        $packet = Receive-UnitreeStartupPacket -TimeoutMs $remaining
        $ready = $null -ne $packet
        if (-not $ready) { Start-Sleep -Milliseconds 100 }
    }
    if (-not $ready) { throw 'Gamepad publisher did not deliver a JSON packet within 30 seconds; training was not started.' }
    if ($CheckSenderOnly) {
        Write-Host 'Gamepad sender startup and JSON packet receipt: PASS'
        return
    }
    Write-Host '[startup 3/3] Gamepad packet received. Initializing simulator; the page is not ready until initialization finishes.'
    Write-Host 'Then open http://localhost:8080 and press Pause / Run. Stop and save ends training.'
    $extraEnvironment = @()
    if ($StudyProfile) {
        $extraEnvironment += @('NAVIGATION_FEATURE_LAGS=0 1 25', 'ONLINE_LR_ACTOR=0.0001',
            'ONLINE_LR_CRITIC=0.0003', 'ONLINE_BATCH_SIZE=256', 'ONLINE_ACTOR_BC_WEIGHT=0.1',
            'ONLINE_N_STEP=1', 'ONLINE_ALPHA=0.01', 'ONLINE_ALPHA_MIN=0.00001', 'ONLINE_ALPHA_MAX=0.1',
            'ONLINE_ALPHA_LR=0.0003', 'ONLINE_ALPHA_STATE_SCOPE=student', 'ONLINE_TARGET_ENTROPY=-3',
            'ONLINE_PREF_SAMPLING_MODE=separate', 'ONLINE_PREF_BATCH_SIZE=256',
            'ONLINE_PREF_REPLAY_CAPACITY=100000', 'ONLINE_PREF_LAMBDA_INIT=1',
            'ONLINE_PREF_LAMBDA_LR=0', 'ONLINE_PREF_LAMBDA_MAX=10', 'TARGET_TERRAIN_PRESET=challenge')
    }
    if ($RunName) { $extraEnvironment += "ONLINE_RUN_NAME=$RunName" }
    if ($ReceiptPath) {
        $extraEnvironment += "ONLINE_SESSION_RECEIPT=$linuxReceipt"
    }
    & wsl.exe -d Ubuntu --cd /home/benjamin/thesis --exec env `
        "GAMEPAD_HOST=$($adapter.IPAddress)" 'GAMEPAD_PORT=8794' `
        "ONLINE_OBJECTIVE_MODE=$Mode" "ONLINE_TOTAL_STEPS=$Steps" `
        "NAVIGATION_OBSERVATION_MODE=$Sensors" "NAVIGATION_IMAGE_PROFILE=$ImageProfile" "NAVIGATION_ENCODER_CHECKPOINT=$EncoderCheckpoint" "SEED=$Seed" `
        @extraEnvironment `
        bash scripts/run_unitree_target_human_training_local.sh
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
}
catch {
    $details = @($_.Exception.Message, "Sender logs: $logRoot")
    foreach ($path in @($senderErr, $senderOut)) {
        if (Test-Path -LiteralPath $path) { $details += Get-Content -LiteralPath $path -Tail 30 }
    }
    throw ($details -join [Environment]::NewLine)
}
finally {
    if (-not $senderProcess.HasExited) {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $senderProcess.Id /T /F | Out-Null
    }
}
