param(
    [ValidateSet("cpo", "pcpo")]
    [string]$Teacher = $(if ($env:TEACHER) { $env:TEACHER } else { "cpo" }),
    [string]$Repo = $(if ($env:REPO) { $env:REPO } else { "C:\Data\thesis" }),
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$RenderMode = $(if ($env:RENDER_MODE) { $env:RENDER_MODE } else { "human" }),
    [int]$Episodes = $(if ($env:EPISODES) { [int]$env:EPISODES } else { 10 }),
    [int]$Fps = $(if ($env:FPS) { [int]$env:FPS } else { 30 }),
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 1 })
)

$ErrorActionPreference = "Stop"

$teacherDir = Join-Path $Repo "models\safetygym_teachers\safe_rl"
$checkpoint = Join-Path $teacherDir "${Teacher}_model_599.pt"
$config = Join-Path $teacherDir "${Teacher}_eval.yaml"
$script = Join-Path $Repo "scripts\eval_safe_rl_safetygym_teacher.py"
$outDir = Join-Path $Repo "logs\safetygym_safe_rl_teacher_eval"

$env:PYTHONPATH = "$Repo;$Repo\safe_rl;$Repo\safety-gymnasium"

Write-Host "Starting Safe-RL Safety-Gym teacher eval"
Write-Host "repo:       $Repo"
Write-Host "teacher:    $Teacher"
Write-Host "checkpoint: $checkpoint"
Write-Host "config:     $config"
Write-Host "render:     $RenderMode"
Write-Host "episodes:   $Episodes"
Write-Host "fps:        $Fps"

& $Python `
  $script `
  --env_id SafetyCarGoal1-v0 `
  --config $config `
  --checkpoint $checkpoint `
  --episodes $Episodes `
  --device cpu `
  --seed $Seed `
  --render_mode $RenderMode `
  --fps $Fps `
  --out_dir $outDir
