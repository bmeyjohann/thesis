param(
    [string]$RepoRoot = $(if ($env:REPO_ROOT) { $env:REPO_ROOT } else { Split-Path -Parent $PSScriptRoot }),
    [string]$PythonExe = $(if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }),
    [string]$EnvName = $(if ($env:ENV_NAME) { $env:ENV_NAME } else { "SafetyCarGoal1-v0" }),
    [string]$Timestamp = $(if ($env:TIMESTAMP) { $env:TIMESTAMP } else { Get-Date -Format "yyyyMMdd_HHmmss" }),
    [string]$ExpPrefix = $(if ($env:EXP_PREFIX) { $env:EXP_PREFIX } else { "safetycar_goal1_plain_small_ln_dense_human" }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 50000 }),
    [int]$LearningStarts = $(if ($env:LEARNING_STARTS) { [int]$env:LEARNING_STARTS } else { 5000 }),
    [int]$CheckpointInterval = $(if ($env:CHECKPOINT_INTERVAL) { [int]$env:CHECKPOINT_INTERVAL } else { 5000 }),
    [int]$NumUpdates = $(if ($env:NUM_UPDATES) { [int]$env:NUM_UPDATES } else { 2 }),
    [int]$PolicyFrequency = $(if ($env:POLICY_FREQUENCY) { [int]$env:POLICY_FREQUENCY } else { 2 }),
    [string]$EnvFpsLimit = $(if ($env:ENV_FPS_LIMIT) { $env:ENV_FPS_LIMIT } else { "30" }),
    [string]$StepPenalty = $(if ($env:STEP_PENALTY) { $env:STEP_PENALTY } else { "-0.001" }),
    [string]$RenderMode = $(if ($env:RENDER_MODE) { $env:RENDER_MODE } else { "human" }),
    [string]$WandbProject = $(if ($env:WANDB_PROJECT) { $env:WANDB_PROJECT } else { "thesis-safetygym" }),
    [string]$WandbMode = $(if ($env:WANDB_MODE) { $env:WANDB_MODE } else { "online" })
)

$ErrorActionPreference = "Stop"

$runName = if ($env:RUN_NAME) { $env:RUN_NAME } else { "${ExpPrefix}_${Timestamp}" }
$wandbGroup = if ($env:WANDB_GROUP) { $env:WANDB_GROUP } else { "safetycar_goal1_plain_small_ln_dense_human_${Timestamp}" }

if (-not (Test-Path (Join-Path $RepoRoot "train_fast_sac_safetygym_minimal.py"))) {
    throw "Could not find train_fast_sac_safetygym_minimal.py under repo root: $RepoRoot"
}

$cmd = @(
    ".\train_fast_sac_safetygym_minimal.py",
    "--env_name", $EnvName,
    "--exp_name", $runName,
    "--variant", "plain",
    "--seed", "1",
    "--device", "auto",
    "--render_mode", $RenderMode,
    "--total_timesteps", "$TotalTimesteps",
    "--learning_starts", "$LearningStarts",
    "--batch_size", "64",
    "--num_updates", "$NumUpdates",
    "--policy_frequency", "$PolicyFrequency",
    "--env_fps_limit", "$EnvFpsLimit",
    "--gamma", "0.99",
    "--tau", "0.005",
    "--actor_learning_rate", "3e-4",
    "--critic_learning_rate", "3e-4",
    "--actor_hidden_dim", "256",
    "--critic_hidden_dim", "512",
    "--module_impl", "custom",
    "--use_layer_norm",
    "--alpha_init", "1e-3",
    "--alpha_min", "0.0",
    "--alpha_max", "1.0",
    "--critic_loss_reduction", "sum",
    "--obs_normalization",
    "--reward_mode", "dense",
    "--step_penalty", "$StepPenalty",
    "--cost_penalty", "0.0",
    "--car_wheel_command_limit", "2.0",
    "--car_force_scale", "2.0",
    "--car_action_mode", "raw_wheels",
    "--scale_actor_to_env_bounds",
    "--terminate_on_goal",
    "--no_reseed_on_episode_reset",
    "--log_interval", "1000",
    "--eval_interval", "$CheckpointInterval",
    "--num_eval_episodes", "10",
    "--save_interval", "$CheckpointInterval",
    "--use_wandb",
    "--wandb_project", $WandbProject,
    "--wandb_mode", $WandbMode,
    "--wandb_group", $wandbGroup,
    "--wandb_run_name", $runName
)

Write-Host "Starting Safety-Gym Goal1 plain baseline run" -ForegroundColor Cyan
Write-Host "repo:       $RepoRoot"
Write-Host "python:     $PythonExe"
Write-Host "run:        $runName"
Write-Host "group:      $wandbGroup"
Write-Host "render:     $RenderMode"
Write-Host "env:        $EnvName"
Write-Host "model:      raw_wheels / small net / layer norm / dense reward"
Write-Host "variant:    plain"
Write-Host "step pen:   $StepPenalty"
Write-Host "env fps:    $EnvFpsLimit"
Write-Host "updates:    num_updates=$NumUpdates, policy_frequency=$PolicyFrequency"

Push-Location $RepoRoot
try {
    & $PythonExe @cmd
}
finally {
    Pop-Location
}
