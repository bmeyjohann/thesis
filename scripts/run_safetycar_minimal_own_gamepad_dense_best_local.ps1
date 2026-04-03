param(
    [string]$RepoRoot = $(if ($env:REPO_ROOT) { $env:REPO_ROOT } else { Split-Path -Parent $PSScriptRoot }),
    [string]$PythonExe = $(if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }),
    [string]$EnvName = $(if ($env:ENV_NAME) { $env:ENV_NAME } else { "SafetyCarGoal2-v0" }),
    [string]$Timestamp = $(if ($env:TIMESTAMP) { $env:TIMESTAMP } else { Get-Date -Format "yyyyMMdd_HHmmss" }),
    [string]$ExpPrefix = $(if ($env:EXP_PREFIX) { $env:EXP_PREFIX } else { "safetycar_min_own_gamepad_dense_best" }),
    [string]$RenderMode = $(if ($env:RENDER_MODE) { $env:RENDER_MODE } else { "human" }),
    [string]$HumanInputDevice = $(if ($env:HUMAN_INPUT_DEVICE) { $env:HUMAN_INPUT_DEVICE } else { "gamepad" }),
    [string]$GamepadMode = $(if ($env:GAMEPAD_MODE) { $env:GAMEPAD_MODE } else { "local" }),
    [string]$WandbProject = $(if ($env:WANDB_PROJECT) { $env:WANDB_PROJECT } else { "thesis-safetygym" }),
    [string]$WandbMode = $(if ($env:WANDB_MODE) { $env:WANDB_MODE } else { "online" }),
    [int]$GamepadPort = $(if ($env:GAMEPAD_PORT) { [int]$env:GAMEPAD_PORT } else { 8793 })
)

$ErrorActionPreference = "Stop"

$runName = if ($env:RUN_NAME) { $env:RUN_NAME } else { "${ExpPrefix}_${Timestamp}" }
$wandbGroup = if ($env:WANDB_GROUP) { $env:WANDB_GROUP } else { "safetycar_dense_own_human_${Timestamp}" }
$gamepadHost = if ($env:GAMEPAD_HOST) { $env:GAMEPAD_HOST } else { "127.0.0.1" }

if (-not (Test-Path (Join-Path $RepoRoot "train_fast_sac_safetygym_minimal.py"))) {
    throw "Could not find train_fast_sac_safetygym_minimal.py under repo root: $RepoRoot"
}

$cmd = @(
    ".\train_fast_sac_safetygym_minimal.py",
    "--env_name", $EnvName,
    "--exp_name", $runName,
    "--variant", "own",
    "--seed", "1",
    "--device", "auto",
    "--render_mode", $RenderMode,
    "--total_timesteps", "50000",
    "--learning_starts", "1000",
    "--batch_size", "64",
    "--num_updates", "2",
    "--policy_frequency", "2",
    "--gamma", "0.99",
    "--tau", "0.005",
    "--actor_learning_rate", "3e-4",
    "--critic_learning_rate", "3e-4",
    "--actor_hidden_dim", "512",
    "--critic_hidden_dim", "1024",
    "--module_impl", "custom",
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
    "--scale_actor_to_env_bounds",
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
    "--store_intervened_in_demo_buffer",
    "--demo_sample_ratio", "0.25",
    "--log_interval", "1000",
    "--eval_interval", "5000",
    "--num_eval_episodes", "10",
    "--save_interval", "5000",
    "--viz_on_checkpoint",
    "--eval_save_episode_plots",
    "--use_wandb",
    "--wandb_project", $WandbProject,
    "--wandb_mode", $WandbMode,
    "--wandb_group", $wandbGroup,
    "--wandb_run_name", $runName
)

Write-Host "Starting rebuilt Safety-Gym dense own-method human run" -ForegroundColor Cyan
Write-Host "repo:   $RepoRoot"
Write-Host "python: $PythonExe"
Write-Host "run:    $runName"
Write-Host "group:  $wandbGroup"
Write-Host "input:  $HumanInputDevice ($GamepadMode)"

Push-Location $RepoRoot
try {
    & $PythonExe @cmd
}
finally {
    Pop-Location
}
