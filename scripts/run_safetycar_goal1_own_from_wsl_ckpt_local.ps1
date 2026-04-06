param(
    [string]$RepoRoot = $(if ($env:REPO_ROOT) { $env:REPO_ROOT } else { Split-Path -Parent $PSScriptRoot }),
    [string]$PythonExe = $(if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }),
    [string]$CheckpointPath = $(if ($env:INIT_CHECKPOINT_PATH) { $env:INIT_CHECKPOINT_PATH } else { "\\wsl.localhost\Ubuntu\home\benjamin\thesis\models\safetygym_minimal\safetycar_min_goal1_raw_small_term_base_20260406_000100\final.pt" }),
    [string]$EnvName = $(if ($env:ENV_NAME) { $env:ENV_NAME } else { "SafetyCarGoal1-v0" }),
    [string]$Timestamp = $(if ($env:TIMESTAMP) { $env:TIMESTAMP } else { Get-Date -Format "yyyyMMdd_HHmmss" }),
    [string]$ExpPrefix = $(if ($env:EXP_PREFIX) { $env:EXP_PREFIX } else { "safetycar_min_goal1_own_human_from_wsl_ckpt" }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 10000 }),
    [int]$LearningStarts = $(if ($env:LEARNING_STARTS) { [int]$env:LEARNING_STARTS } else { 1000 }),
    [int]$CheckpointInterval = $(if ($env:CHECKPOINT_INTERVAL) { [int]$env:CHECKPOINT_INTERVAL } else { 2000 }),
    [string]$RenderMode = $(if ($env:RENDER_MODE) { $env:RENDER_MODE } else { "human" }),
    [string]$HumanInputDevice = $(if ($env:HUMAN_INPUT_DEVICE) { $env:HUMAN_INPUT_DEVICE } else { "gamepad" }),
    [string]$GamepadMode = $(if ($env:GAMEPAD_MODE) { $env:GAMEPAD_MODE } else { "local" }),
    [string]$WandbProject = $(if ($env:WANDB_PROJECT) { $env:WANDB_PROJECT } else { "thesis-safetygym" }),
    [string]$WandbMode = $(if ($env:WANDB_MODE) { $env:WANDB_MODE } else { "online" }),
    [int]$GamepadPort = $(if ($env:GAMEPAD_PORT) { [int]$env:GAMEPAD_PORT } else { 8793 })
)

$ErrorActionPreference = "Stop"

$runName = if ($env:RUN_NAME) { $env:RUN_NAME } else { "${ExpPrefix}_${Timestamp}" }
$wandbGroup = if ($env:WANDB_GROUP) { $env:WANDB_GROUP } else { "safetycar_goal1_own_human_probe_${Timestamp}" }
$gamepadHost = if ($env:GAMEPAD_HOST) { $env:GAMEPAD_HOST } else { "127.0.0.1" }

if (-not (Test-Path (Join-Path $RepoRoot "train_fast_sac_safetygym_minimal.py"))) {
    throw "Could not find train_fast_sac_safetygym_minimal.py under repo root: $RepoRoot"
}

if (-not (Test-Path $CheckpointPath)) {
    throw "Checkpoint path does not exist: $CheckpointPath"
}

$cmd = @(
    ".\train_fast_sac_safetygym_minimal.py",
    "--env_name", $EnvName,
    "--exp_name", $runName,
    "--variant", "own",
    "--seed", "1",
    "--device", "auto",
    "--render_mode", $RenderMode,
    "--total_timesteps", "$TotalTimesteps",
    "--learning_starts", "$LearningStarts",
    "--batch_size", "64",
    "--num_updates", "7",
    "--policy_frequency", "2",
    "--gamma", "0.97",
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
    "--step_penalty", "-0.001",
    "--cost_penalty", "0.0",
    "--car_wheel_command_limit", "2.0",
    "--car_force_scale", "2.0",
    "--car_action_mode", "raw_wheels",
    "--scale_actor_to_env_bounds",
    "--terminate_on_goal",
    "--no_reseed_on_episode_reset",
    "--use_intervention",
    "--human_input_device", $HumanInputDevice,
    "--gamepad_mode", $GamepadMode,
    "--gamepad_host", $gamepadHost,
    "--gamepad_port", "$GamepadPort",
    "--human_action_scale", "1.0",
    "--intervention_threshold", "0.02",
    "--intervention_hold_seconds", "0.15",
    "--pref_sampling_mode", "linked",
    "--pref_sample_ratio", "0.5",
    "--pref_rank_weight", "1.0",
    "--pref_loss_type", "lagrangian",
    "--pref_stopgrad_positive",
    "--demo_sample_ratio", "0.0",
    "--init_checkpoint_path", $CheckpointPath,
    "--load_actor_from_checkpoint",
    "--load_critic_from_checkpoint",
    "--load_critic_target_from_checkpoint",
    "--load_alpha_from_checkpoint",
    "--log_interval", "1000",
    "--eval_interval", "$CheckpointInterval",
    "--num_eval_episodes", "10",
    "--save_interval", "$CheckpointInterval",
    "--viz_on_checkpoint",
    "--eval_save_episode_plots",
    "--use_wandb",
    "--wandb_project", $WandbProject,
    "--wandb_mode", $WandbMode,
    "--wandb_group", $wandbGroup,
    "--wandb_run_name", $runName
)

Write-Host "Starting Safety-Gym Goal1 own-method human run from WSL checkpoint" -ForegroundColor Cyan
Write-Host "repo:       $RepoRoot"
Write-Host "python:     $PythonExe"
Write-Host "checkpoint: $CheckpointPath"
Write-Host "run:        $runName"
Write-Host "group:      $wandbGroup"
Write-Host "input:      $HumanInputDevice ($GamepadMode)"

Push-Location $RepoRoot
try {
    & $PythonExe @cmd
}
finally {
    Pop-Location
}
