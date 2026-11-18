param(
  [string]$Repo = "C:\Data\thesis",
  [string]$Device = "cpu",
  [int]$NumEpisodes = 10,
  [int]$MaxSteps = 500,
  [switch]$Video,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$launcher = Join-Path $Repo "scripts\run_isaac_navigation_scripted_eval_win.ps1"
if (!(Test-Path $launcher)) {
  throw "Single-run launcher not found: $launcher"
}

$runRoot = Join-Path $Repo "logs\isaac_navigation_scripted"
$straightOut = Join-Path $runRoot "win_straightline_official_probe"
$obstacleOut = Join-Path $runRoot "win_single_obstacle_teacher_probe"

$common = @{
  Repo = $Repo
  NavLowLevelCfg = "official"
  Device = $Device
  NumEpisodes = $NumEpisodes
  MaxSteps = $MaxSteps
}
if ($Video) { $common.Video = $true }
if ($DryRun) { $common.DryRun = $true }

Write-Host "Running Isaac/Go2 validation bundle"
Write-Host "repo:       $Repo"
Write-Host "episodes:   $NumEpisodes"
Write-Host "max steps:  $MaxSteps"
Write-Host "device:     $Device"
Write-Host "video:      $Video"
Write-Host "straight:   $straightOut"
Write-Host "obstacle:   $obstacleOut"

& $launcher @common `
  -Task "Isaac-Navigation-NoObstacles-Flat-Go2-v0" `
  -Controller "goal_straight" `
  -OutDir $straightOut
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $launcher @common `
  -Task "Isaac-Navigation-SingleObstacle-Flat-Go2-v0" `
  -Controller "obstacle_teacher" `
  -OutDir $obstacleOut `
  -TeacherObstacleX 1.0 `
  -TeacherObstacleY 0.0 `
  -TeacherObstacleRadius 0.65 `
  -TeacherObstacleMargin 0.45 `
  -TeacherObstacleSide "left"
exit $LASTEXITCODE
