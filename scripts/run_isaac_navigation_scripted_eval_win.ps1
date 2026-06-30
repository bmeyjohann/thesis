param(
  [string]$Repo = "C:\Data\thesis",
  [string]$Task = "Isaac-Navigation-NoObstacles-Flat-Go2-v0",
  [string]$Controller = "goal_straight",
  [string]$NavLowLevelCfg = "official",
  [string]$Device = "cpu",
  [string]$LowLevelPolicyPath = "",
  [string]$OutDir = "",
  [int]$NumEpisodes = 10,
  [int]$MaxSteps = 500,
  [int]$Seed = 0,
  [double]$LinGain = 0.35,
  [double]$YawGain = 0.8,
  [double]$MaxLinVel = 0.35,
  [double]$MaxYawVel = 0.6,
  [double]$ForwardHeadingDeadband = 0.35,
  [double]$TeacherObstacleX = 1.0,
  [double]$TeacherObstacleY = 0.0,
  [double]$TeacherObstacleRadius = 0.65,
  [double]$TeacherObstacleMargin = 0.45,
  [string]$TeacherObstacleSide = "left",
  [switch]$Video,
  [int]$VideoLength = 350,
  [string]$VideoCameraMode = "scene",
  [bool]$DryRun = $false
)

$ErrorActionPreference = "Stop"

if ($LowLevelPolicyPath -eq "") {
  $LowLevelPolicyPath = Join-Path $Repo "logs\rsl_rl\unitree_go2_flat\2025-11-23_13-46-53_official_task_ppo_12893598\exported\policy.pt"
}
if ($OutDir -eq "") {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutDir = Join-Path $Repo "logs\isaac_navigation_scripted\win_${Controller}_${stamp}"
}

$condaActivate = "C:\Users\benja\miniconda3\Scripts\activate.bat"
$condaEnv = "C:\Users\benja\miniconda3\envs\env_isaaclab"
$isaacLabBat = Join-Path $Repo "IsaacLab\isaaclab.bat"
$evalScript = Join-Path $Repo "safe-locomotion\scripts\rsl_rl\eval_navigation_scripted.py"

if (!(Test-Path $condaActivate)) { throw "Conda activate not found: $condaActivate" }
if (!(Test-Path $condaEnv)) { throw "Conda env not found: $condaEnv" }
if (!(Test-Path $isaacLabBat)) { throw "IsaacLab bat not found: $isaacLabBat" }
if (!(Test-Path $evalScript)) { throw "Eval script not found: $evalScript" }
if (!(Test-Path $LowLevelPolicyPath)) { throw "Low-level policy not found: $LowLevelPolicyPath" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:NAV_LOW_LEVEL_CFG = $NavLowLevelCfg
$env:MPLCONFIGDIR = Join-Path $env:TEMP "mplconfig_isaac_nav"

$videoArgs = @()
if ($Video) {
  $videoArgs = @("--video", "--video_length", "$VideoLength")
}

Write-Host "Isaac scripted navigation eval (Windows)"
Write-Host "repo:       $Repo"
Write-Host "task:       $Task"
Write-Host "nav cfg:    $NavLowLevelCfg"
Write-Host "device:     $Device"
Write-Host "policy:     $LowLevelPolicyPath"
Write-Host "controller: $Controller"
Write-Host "out:        $OutDir"
Write-Host "video:      $Video"
Write-Host "dry run:    $DryRun"

$evalArgs = $videoArgs + @(
  "--device", $Device,
  "--task", $Task,
  "--low_level_policy_path", $LowLevelPolicyPath,
  "--num_envs", "1",
  "--num_episodes", "$NumEpisodes",
  "--max_steps", "$MaxSteps",
  "--output_dir", $OutDir,
  "--seed", "$Seed",
  "--controller", $Controller,
  "--lin_gain", "$LinGain",
  "--yaw_gain", "$YawGain",
  "--max_lin_vel", "$MaxLinVel",
  "--max_yaw_vel", "$MaxYawVel",
  "--forward_heading_deadband", "$ForwardHeadingDeadband",
  "--teacher_obstacle_x", "$TeacherObstacleX",
  "--teacher_obstacle_y", "$TeacherObstacleY",
  "--teacher_obstacle_radius", "$TeacherObstacleRadius",
  "--teacher_obstacle_margin", "$TeacherObstacleMargin",
  "--teacher_obstacle_side", "$TeacherObstacleSide",
  "--video_camera_mode", "$VideoCameraMode",
  "--stop_on_success"
)

function Quote-CmdArg([string]$Value) {
  if ($Value -match '[\s"]') {
    return '"' + ($Value -replace '"', '\"') + '"'
  }
  return $Value
}

$evalArgString = ($evalArgs | ForEach-Object { Quote-CmdArg "$_" }) -join " "
$cmdString = "call `"$condaActivate`" `"$condaEnv`" && cd /d `"$Repo`" && `"$isaacLabBat`" -p `"$evalScript`" --headless $evalArgString"

Write-Host "cmd:        $cmdString"
if ($DryRun) {
  exit 0
}

$cmdFile = Join-Path $OutDir "run_eval.cmd"
$cmdFileContent = @"
@echo on
set NAV_LOW_LEVEL_CFG=$NavLowLevelCfg
set MPLCONFIGDIR=$env:MPLCONFIGDIR
call "$condaActivate" "$condaEnv"
if errorlevel 1 exit /b %errorlevel%
cd /d "$Repo"
if errorlevel 1 exit /b %errorlevel%
"$isaacLabBat" -p "$evalScript" --headless $evalArgString
exit /b %errorlevel%
"@
$cmdFileContent | Set-Content -Encoding ASCII -Path $cmdFile
Write-Host "cmd file:   $cmdFile"

& $cmdFile
$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
  $exitCode = 1
}
Write-Host "exit code:  $exitCode"
exit $exitCode
