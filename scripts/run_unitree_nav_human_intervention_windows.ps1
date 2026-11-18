param(
    [string]$Distro = $(if ($env:WSL_DISTRO_NAME) { $env:WSL_DISTRO_NAME } else { "Ubuntu" }),
    [string]$WslRepo = $(if ($env:UNITREE_WSL_REPO) { $env:UNITREE_WSL_REPO } else { "/home/benjamin/thesis" }),
    [string]$ModelPath = $(if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "/home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_goalonly_obstacles_forward_5k_20260729/final.pt" }),
    [string]$DatasetDir = $(if ($env:HUMAN_DATASET_DIR) { $env:HUMAN_DATASET_DIR } else { "" }),
    [int]$GamepadPort = $(if ($env:GAMEPAD_PORT) { [int]$env:GAMEPAD_PORT } else { 8794 }),
    [string]$GamepadHost = $(if ($env:GAMEPAD_HOST) { $env:GAMEPAD_HOST } else { "" }),
    [int]$GamepadDeviceIndex = $(if ($env:GAMEPAD_DEVICE_INDEX) { [int]$env:GAMEPAD_DEVICE_INDEX } else { 0 }),
    [double]$SimFps = $(if ($env:SIM_FPS) { [double]$env:SIM_FPS } else { 20.0 }),
    [double]$GamepadInterventionThreshold = $(if ($env:GAMEPAD_INTERVENTION_THRESHOLD) { [double]$env:GAMEPAD_INTERVENTION_THRESHOLD } else { 0.05 }),
    [double]$GamepadDebugHz = $(if ($env:GAMEPAD_DEBUG_HZ) { [double]$env:GAMEPAD_DEBUG_HZ } else { 4.0 }),
    [int]$OnlineTrain = $(if ($env:ONLINE_TRAIN) { [int]$env:ONLINE_TRAIN } else { 1 }),
    [string]$OnlineRunName = $(if ($env:ONLINE_RUN_NAME) { $env:ONLINE_RUN_NAME } else { "" }),
    [string]$OnlineObjectiveMode = $(if ($env:ONLINE_OBJECTIVE_MODE) { $env:ONLINE_OBJECTIVE_MODE } else { "pref_bc_rl" }),
    [int]$OnlineTotalSteps = $(if ($env:ONLINE_TOTAL_STEPS) { [int]$env:ONLINE_TOTAL_STEPS } else { 0 }),
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 0 }),
    [double]$OnlinePrefRankWeight = $(if ($env:ONLINE_PREF_RANK_WEIGHT) { [double]$env:ONLINE_PREF_RANK_WEIGHT } else { 1.0 }),
    [double]$OnlineActorBcWeight = $(if ($env:ONLINE_ACTOR_BC_WEIGHT) { [double]$env:ONLINE_ACTOR_BC_WEIGHT } else { 0.2 }),
    [double]$OnlinePrefProxyValueBound = $(if ($env:ONLINE_PREF_PROXY_VALUE_BOUND) { [double]$env:ONLINE_PREF_PROXY_VALUE_BOUND } else { 1.0 }),
    [switch]$FromScratch,
    [double]$HeightScanForwardSize = $(if ($env:HEIGHT_SCAN_FORWARD_SIZE) { [double]$env:HEIGHT_SCAN_FORWARD_SIZE } else { 5.0 }),
    [double]$HeightScanLateralSize = $(if ($env:HEIGHT_SCAN_LATERAL_SIZE) { [double]$env:HEIGHT_SCAN_LATERAL_SIZE } else { 3.0 }),
    [double]$HeightScanResolution = $(if ($env:HEIGHT_SCAN_RESOLUTION) { [double]$env:HEIGHT_SCAN_RESOLUTION } else { 0.25 }),
    [string]$HeightScanPattern = $(if ($env:HEIGHT_SCAN_PATTERN) { $env:HEIGHT_SCAN_PATTERN } else { "grid" }),
    [double]$HeightScanFrustumNear = 0.25,
    [double]$HeightScanFrustumFar = 4.0,
    [double]$HeightScanFrustumFovDeg = 70.0,
    [int]$HeightScanFrustumSide = 17,
    [int]$ScanHistory = $(if ($env:SCAN_HISTORY) { [int]$env:SCAN_HISTORY } else { 5 }),
    [int]$ScanHistoryStride = $(if ($env:SCAN_HISTORY_STRIDE) { [int]$env:SCAN_HISTORY_STRIDE } else { 5 }),
    [int]$ActionHistory = $(if ($env:ACTION_HISTORY) { [int]$env:ACTION_HISTORY } else { 20 }),
    [string]$GoalEncoding = $(if ($env:GOAL_ENCODING) { $env:GOAL_ENCODING } else { "distance_bearing" }),
    [double]$GoalDistanceScale = $(if ($env:GOAL_DISTANCE_SCALE) { [double]$env:GOAL_DISTANCE_SCALE } else { 14.0 }),
    [double]$VelocityScale = $(if ($env:VELOCITY_SCALE) { [double]$env:VELOCITY_SCALE } else { 1.0 }),
    [double]$StartPositionRange = $(if ($env:START_POSITION_RANGE) { [double]$env:START_POSITION_RANGE } else { 0.8 }),
    [string]$NavigationEpisodeMode = $(if ($env:NAVIGATION_EPISODE_MODE) { $env:NAVIGATION_EPISODE_MODE } else { "continuous_goals" }),
    [double]$ContinuousGoalDistanceMin = 8.0,
    [double]$ContinuousGoalDistanceMax = 14.0,
    [string]$ContinuousGoalRegionMode = "terrain_bank",
    [double]$ContinuousGoalBlockedProbability = 0.85,
    [double]$MinGoalObstacleClearance = 1.0,
    [double]$ViewerMapRadius = 7.0,
    [int]$ShowRgb = 1,
    [int]$RgbEvery = 5,
    [int]$VideoWidth = 320,
    [int]$VideoHeight = 240,
    [double]$VideoCameraDistance = 7.0,
    [double]$VideoCameraElevation = -65.0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sender = Join-Path $PSScriptRoot "run_safetygym_gamepad_sender.ps1"
if ($FromScratch) {
    $ModelPath = ""
    if ($HeightScanForwardSize -le 0.0) { $HeightScanForwardSize = 5.0 }
    if ($HeightScanLateralSize -le 0.0) { $HeightScanLateralSize = 3.0 }
    if ($HeightScanResolution -le 0.0) { $HeightScanResolution = 0.25 }
}

function ConvertTo-BashLiteral([string]$Value) {
    # Bash's only safe embedded-single-quote sequence is: '"'"'. Build it
    # numerically so PowerShell never has to parse nested quote delimiters.
    $singleQuote = [string]([char]39)
    $doubleQuote = [string]([char]34)
    $embeddedQuote = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    $escaped = $Value.Replace($singleQuote, $embeddedQuote)
    return $singleQuote + $escaped + $singleQuote
}

if (-not (Test-Path $sender)) {
    throw "Missing existing Windows raw gamepad sender: $sender"
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe is not available on PATH."
}
if ([string]::IsNullOrWhiteSpace($GamepadHost)) {
    $wslAdapter = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.InterfaceAlias -like "vEthernet (WSL*" } |
        Select-Object -First 1
    if (-not $wslAdapter) {
        throw "Could not discover the Windows WSL adapter address. Set GAMEPAD_HOST explicitly."
    }
    $GamepadHost = $wslAdapter.IPAddress
}

$bashRepo = ConvertTo-BashLiteral $WslRepo
$bashModel = ConvertTo-BashLiteral $ModelPath
$bashDataset = ConvertTo-BashLiteral $DatasetDir
$bashGamepadHost = ConvertTo-BashLiteral $GamepadHost
$bashPort = ConvertTo-BashLiteral "$GamepadPort"
$bashFps = ConvertTo-BashLiteral "$SimFps"
$bashInterventionThreshold = ConvertTo-BashLiteral "$GamepadInterventionThreshold"
$bashOnlineTrain = ConvertTo-BashLiteral "$OnlineTrain"
$bashOnlineRunName = ConvertTo-BashLiteral $OnlineRunName
$bashOnlineObjectiveMode = ConvertTo-BashLiteral $OnlineObjectiveMode
$bashOnlineTotalSteps = ConvertTo-BashLiteral "$OnlineTotalSteps"
$bashSeed = ConvertTo-BashLiteral "$Seed"
$bashOnlinePrefRankWeight = ConvertTo-BashLiteral "$OnlinePrefRankWeight"
$bashOnlineActorBcWeight = ConvertTo-BashLiteral "$OnlineActorBcWeight"
$bashOnlinePrefProxyValueBound = ConvertTo-BashLiteral "$OnlinePrefProxyValueBound"
$bashScanForwardSize = ConvertTo-BashLiteral "$HeightScanForwardSize"
$bashScanLateralSize = ConvertTo-BashLiteral "$HeightScanLateralSize"
$bashScanResolution = ConvertTo-BashLiteral "$HeightScanResolution"
$bashScanPattern = ConvertTo-BashLiteral "$HeightScanPattern"
$bashScanFrustumNear = ConvertTo-BashLiteral "$HeightScanFrustumNear"
$bashScanFrustumFar = ConvertTo-BashLiteral "$HeightScanFrustumFar"
$bashScanFrustumFovDeg = ConvertTo-BashLiteral "$HeightScanFrustumFovDeg"
$bashScanFrustumSide = ConvertTo-BashLiteral "$HeightScanFrustumSide"
$bashScanHistory = ConvertTo-BashLiteral "$ScanHistory"
$bashScanHistoryStride = ConvertTo-BashLiteral "$ScanHistoryStride"
$bashActionHistory = ConvertTo-BashLiteral "$ActionHistory"
$bashGoalEncoding = ConvertTo-BashLiteral $GoalEncoding
$bashGoalDistanceScale = ConvertTo-BashLiteral "$GoalDistanceScale"
$bashVelocityScale = ConvertTo-BashLiteral "$VelocityScale"
$bashStartPositionRange = ConvertTo-BashLiteral "$StartPositionRange"
$bashNavigationEpisodeMode = ConvertTo-BashLiteral $NavigationEpisodeMode
$bashContinuousGoalDistanceMin = ConvertTo-BashLiteral "$ContinuousGoalDistanceMin"
$bashContinuousGoalDistanceMax = ConvertTo-BashLiteral "$ContinuousGoalDistanceMax"
$bashContinuousGoalRegionMode = ConvertTo-BashLiteral $ContinuousGoalRegionMode
$bashContinuousGoalBlockedProbability = ConvertTo-BashLiteral "$ContinuousGoalBlockedProbability"
$bashMinGoalObstacleClearance = ConvertTo-BashLiteral "$MinGoalObstacleClearance"
$bashViewerMapRadius = ConvertTo-BashLiteral "$ViewerMapRadius"
$bashShowRgb = ConvertTo-BashLiteral "$ShowRgb"
$bashRgbEvery = ConvertTo-BashLiteral "$RgbEvery"
$bashVideoWidth = ConvertTo-BashLiteral "$VideoWidth"
$bashVideoHeight = ConvertTo-BashLiteral "$VideoHeight"
$bashVideoCameraDistance = ConvertTo-BashLiteral "$VideoCameraDistance"
$bashVideoCameraElevation = ConvertTo-BashLiteral "$VideoCameraElevation"
$bashCommand = @"
set -euo pipefail
test -x $bashRepo/scripts/run_unitree_mjlab_nav_human_intervention_local.sh
cd $bashRepo
MODEL_PATH=$bashModel \
GAMEPAD_HOST=$bashGamepadHost \
GAMEPAD_PORT=$bashPort \
SIM_FPS=$bashFps \
GAMEPAD_INTERVENTION_THRESHOLD=$bashInterventionThreshold \
HUMAN_DATASET_DIR=$bashDataset \
ONLINE_TRAIN=$bashOnlineTrain \
ONLINE_RUN_NAME=$bashOnlineRunName \
ONLINE_OBJECTIVE_MODE=$bashOnlineObjectiveMode \
ONLINE_TOTAL_STEPS=$bashOnlineTotalSteps \
SEED=$bashSeed \
ONLINE_PREF_RANK_WEIGHT=$bashOnlinePrefRankWeight \
ONLINE_ACTOR_BC_WEIGHT=$bashOnlineActorBcWeight \
ONLINE_PREF_PROXY_VALUE_BOUND=$bashOnlinePrefProxyValueBound \
HEIGHT_SCAN_FORWARD_SIZE=$bashScanForwardSize \
HEIGHT_SCAN_LATERAL_SIZE=$bashScanLateralSize \
HEIGHT_SCAN_RESOLUTION=$bashScanResolution \
HEIGHT_SCAN_PATTERN=$bashScanPattern \
HEIGHT_SCAN_FRUSTUM_NEAR=$bashScanFrustumNear \
HEIGHT_SCAN_FRUSTUM_FAR=$bashScanFrustumFar \
HEIGHT_SCAN_FRUSTUM_FOV_DEG=$bashScanFrustumFovDeg \
HEIGHT_SCAN_FRUSTUM_SIDE=$bashScanFrustumSide \
SCAN_HISTORY=$bashScanHistory \
SCAN_HISTORY_STRIDE=$bashScanHistoryStride \
ACTION_HISTORY=$bashActionHistory \
GOAL_ENCODING=$bashGoalEncoding \
GOAL_DISTANCE_SCALE=$bashGoalDistanceScale \
VELOCITY_SCALE=$bashVelocityScale \
START_POSITION_RANGE=$bashStartPositionRange \
NAVIGATION_EPISODE_MODE=$bashNavigationEpisodeMode \
CONTINUOUS_GOAL_DISTANCE_MIN=$bashContinuousGoalDistanceMin \
CONTINUOUS_GOAL_DISTANCE_MAX=$bashContinuousGoalDistanceMax \
CONTINUOUS_GOAL_REGION_MODE=$bashContinuousGoalRegionMode \
CONTINUOUS_GOAL_BLOCKED_PROBABILITY=$bashContinuousGoalBlockedProbability \
MIN_GOAL_OBSTACLE_CLEARANCE=$bashMinGoalObstacleClearance \
VIEWER_MAP_RADIUS=$bashViewerMapRadius \
SHOW_RGB=$bashShowRgb \
RGB_EVERY=$bashRgbEvery \
VIDEO_WIDTH=$bashVideoWidth \
VIDEO_HEIGHT=$bashVideoHeight \
VIDEO_CAMERA_DISTANCE=$bashVideoCameraDistance \
VIDEO_CAMERA_ELEVATION=$bashVideoCameraElevation \
./scripts/run_unitree_mjlab_nav_human_intervention_local.sh
"@
$bashCommand = $bashCommand.Replace("`r`n", "`n")

Write-Host "Starting Unitree human-intervention training" -ForegroundColor Cyan
Write-Host "Windows repo: $repoRoot"
Write-Host "WSL repo:     $WslRepo"
Write-Host "checkpoint:   $ModelPath"
Write-Host "objective:    $OnlineObjectiveMode"
Write-Host "dataset:      $(if ($DatasetDir) { $DatasetDir } else { '<timestamped WSL default>' })"
Write-Host "gamepad:      Windows localhost:$GamepadPort -> WSL target ${GamepadHost}:$GamepadPort"
Write-Host "sender debug: $GamepadDebugHz Hz; move a stick to see named_axes"
Write-Host "sender window: a second PowerShell window stays open for controller diagnostics"
Write-Host "takeover:     stick command norm >= $GamepadInterventionThreshold (RB is not required)"
Write-Host "online train: $OnlineTrain"
Write-Host "scanner:      mode=$HeightScanPattern | grid=${HeightScanForwardSize}x${HeightScanLateralSize}m @ ${HeightScanResolution}m | frustum=${HeightScanFrustumNear}-${HeightScanFrustumFar}m ${HeightScanFrustumFovDeg}deg ${HeightScanFrustumSide}x${HeightScanFrustumSide}"
Write-Host "history:      $ScanHistory scans every $ScanHistoryStride steps; $ActionHistory dense actions"
Write-Host "start area:   +/-${StartPositionRange}m around each terrain-tile origin"
Write-Host "episode mode: $NavigationEpisodeMode"
Write-Host "goal sampling: ${ContinuousGoalDistanceMin}-${ContinuousGoalDistanceMax}m | region=$ContinuousGoalRegionMode | blocked=$ContinuousGoalBlockedProbability"
Write-Host "goal clearance: ${MinGoalObstacleClearance}m | viewer radius=${ViewerMapRadius}m"
Write-Host "render:       RGB=${ShowRgb} ${VideoWidth}x${VideoHeight} every ${RgbEvery} steps | camera elevation=${VideoCameraElevation}"
Write-Host "initialization: $(if ($FromScratch) { 'scratch' } else { 'checkpoint' })"
Write-Host "controls:     move stick to steer | Start pause/run | Back stop | X reset | Y next layout | A step"

# The Windows sender needs native SDL access. It only publishes raw state; all
# Unitree mapping and collection logic remains in the authoritative WSL repo.
$existingListener = Get-NetTCPConnection -LocalPort $GamepadPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($existingListener) {
    throw "Port $GamepadPort is already owned by process $($existingListener.OwningProcess). Close the stale gamepad sender before launching a new collection run."
}
$senderArgs = @(
    "-NoProfile",
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", $sender,
    "-EnvName", "SafetyPointGoal1-v0",
    "-ActionDim", "3",
    "-Port", "$GamepadPort",
    "-GamepadDeviceIndex", "$GamepadDeviceIndex",
    "-DebugHz", "$GamepadDebugHz",
    "-NoSavedConfig"
)
$senderProcess = $null
try {
    $senderProcess = Start-Process -FilePath "powershell.exe" -ArgumentList $senderArgs -PassThru

    # Do not launch Isaac/WSL collection against a dead publisher. A refused
    # WSL connection otherwise looks like a receiver bug even though no
    # Windows port is listening.
    $publisherReady = $false
    $publisherDeadline = [DateTime]::UtcNow.AddSeconds(20)
    while (-not $publisherReady -and [DateTime]::UtcNow -lt $publisherDeadline) {
        if ($senderProcess.HasExited) {
            throw "Gamepad publisher exited during startup. Inspect the sender PowerShell window for its traceback."
        }
        $publisherProbe = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $publisherProbe.BeginConnect("127.0.0.1", $GamepadPort, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(500)) {
                $publisherProbe.EndConnect($connect)
                $publisherReady = $true
            }
        }
        catch {
            # The publisher may still be importing Pygame and project modules.
        }
        finally {
            $publisherProbe.Dispose()
        }
        if (-not $publisherReady) {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $publisherReady) {
        throw "Gamepad publisher did not listen on Windows localhost:${GamepadPort} within 20 seconds. Inspect the sender PowerShell window; WSL collection was not started."
    }
    Write-Host "gamepad publisher: listening on Windows localhost:$GamepadPort" -ForegroundColor Green

    & wsl.exe -d $Distro -- bash -lc $bashCommand
    if ($null -eq $LASTEXITCODE) {
        throw "wsl.exe returned no exit code. This PowerShell was likely launched through WSL interop. Use run_unitree_nav_gamepad_sender_windows.ps1 here and run the WSL collector separately."
    }
    if ($LASTEXITCODE -ne 0) {
        throw "WSL collector exited with code $LASTEXITCODE"
    }
}
finally {
    if ($senderProcess -and -not $senderProcess.HasExited) {
        # Stop the nested PowerShell and its Python publisher child. Killing
        # only the shell leaves an orphan listener on port 8794.
        Start-Process -FilePath "$env:SystemRoot\System32\taskkill.exe" `
            -ArgumentList @("/PID", "$($senderProcess.Id)", "/T", "/F") `
            -WindowStyle Hidden `
            -Wait
    }
}




