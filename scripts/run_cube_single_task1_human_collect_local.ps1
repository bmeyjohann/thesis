param(
    [string]$RepoRoot = $(if ($env:REPO_ROOT) { $env:REPO_ROOT } else { Split-Path -Parent $PSScriptRoot }),
    [string]$PythonExe = $(if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }),
    [string]$EnvName = $(if ($env:ENV_NAME) { $env:ENV_NAME } else { "cube-single-singletask-task1-v0" }),
    [string]$VrHost = $(if ($env:VR_HOST) { $env:VR_HOST } else { "127.0.0.1" }),
    [int]$VrPort = $(if ($env:VR_PORT) { [int]$env:VR_PORT } else { 8765 }),
    [string]$Project = $(if ($env:PROJECT) { $env:PROJECT } else { "ogbench-manip-human-data" }),
    [string]$WandbMode = $(if ($env:WANDB_MODE) { $env:WANDB_MODE } else { "online" }),
    [string]$Timestamp = $(if ($env:TIMESTAMP) { $env:TIMESTAMP } else { Get-Date -Format "yyyyMMdd_HHmmss" }),
    [string]$ExpPrefix = $(if ($env:EXP_PREFIX) { $env:EXP_PREFIX } else { "cube_single_task1_human_collect_norot_fixedalpha1e3_win" }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 30000 }),
    [int]$MaxEpisodeSteps = $(if ($env:MAX_EPISODE_STEPS) { [int]$env:MAX_EPISODE_STEPS } else { 1000 }),
    [int]$BatchSize = $(if ($env:BATCH_SIZE) { [int]$env:BATCH_SIZE } else { 256 }),
    [int]$NumUpdates = $(if ($env:NUM_UPDATES) { [int]$env:NUM_UPDATES } else { 1 }),
    [int]$CtaRatio = $(if ($env:CTA_RATIO) { [int]$env:CTA_RATIO } else { 2 }),
    [int]$LearningStarts = $(if ($env:LEARNING_STARTS) { [int]$env:LEARNING_STARTS } else { 2000 }),
    [int]$ActorHiddenDim = $(if ($env:ACTOR_HIDDEN_DIM) { [int]$env:ACTOR_HIDDEN_DIM } else { 256 }),
    [int]$CriticHiddenDim = $(if ($env:CRITIC_HIDDEN_DIM) { [int]$env:CRITIC_HIDDEN_DIM } else { 512 }),
    [int]$NumCritics = $(if ($env:NUM_CRITICS) { [int]$env:NUM_CRITICS } else { 2 }),
    [double]$Gamma = $(if ($env:GAMMA) { [double]$env:GAMMA } else { 0.97 }),
    [double]$FixedAlpha = $(if ($env:FIXED_ALPHA) { [double]$env:FIXED_ALPHA } else { 0.001 }),
    [double]$AlphaInit = $(if ($env:ALPHA_INIT) { [double]$env:ALPHA_INIT } else { 0.001 }),
    [double]$AlphaMin = $(if ($env:ALPHA_MIN) { [double]$env:ALPHA_MIN } else { 0.001 }),
    [double]$AlphaMax = $(if ($env:ALPHA_MAX) { [double]$env:ALPHA_MAX } else { 0.001 }),
    [double]$InterventionEpisodeProb = $(if ($env:INTERVENTION_EPISODE_PROB) { [double]$env:INTERVENTION_EPISODE_PROB } else { 1.0 }),
    [double]$DemoSampleRatio = $(if ($env:DEMO_SAMPLE_RATIO) { [double]$env:DEMO_SAMPLE_RATIO } else { 0.5 }),
    [int]$DemoPrefillEpisodes = $(if ($env:DEMO_PREFILL_EPISODES) { [int]$env:DEMO_PREFILL_EPISODES } else { 0 }),
    [int]$DemoPrefillNumEnvs = $(if ($env:DEMO_PREFILL_NUM_ENVS) { [int]$env:DEMO_PREFILL_NUM_ENVS } else { 0 }),
    [double]$PrefRankWeight = $(if ($env:PREF_RANK_WEIGHT) { [double]$env:PREF_RANK_WEIGHT } else { 1.0 }),
    [double]$PrefRankMargin = $(if ($env:PREF_RANK_MARGIN) { [double]$env:PREF_RANK_MARGIN } else { 0.01 }),
    [string]$PrefLossType = $(if ($env:PREF_LOSS_TYPE) { $env:PREF_LOSS_TYPE } else { "lagrangian" }),
    [string]$PrefCriticScope = $(if ($env:PREF_CRITIC_SCOPE) { $env:PREF_CRITIC_SCOPE } else { "all" }),
    [int]$EvalInterval = $(if ($env:EVAL_INTERVAL) { [int]$env:EVAL_INTERVAL } else { 10000 }),
    [int]$SaveInterval = $(if ($env:SAVE_INTERVAL) { [int]$env:SAVE_INTERVAL } else { 5000 }),
    [int]$LogInterval = $(if ($env:LOG_INTERVAL) { [int]$env:LOG_INTERVAL } else { 64 }),
    [int]$ExportReplayDatasetInterval = $(if ($env:EXPORT_REPLAY_DATASET_INTERVAL) { [int]$env:EXPORT_REPLAY_DATASET_INTERVAL } else { 1000 }),
    [string]$ExportReplayDatasetPath = $(if ($env:EXPORT_REPLAY_DATASET_PATH) { $env:EXPORT_REPLAY_DATASET_PATH } else { "" }),
    [switch]$VisualizeInterventionColors
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $RepoRoot "train_fast_sac_ogbench_manip.py"))) {
    throw "Could not find train_fast_sac_ogbench_manip.py under repo root: $RepoRoot"
}

$pythonPrefixArgs = @()
$resolvedPythonExe = $null
if ($PythonExe -and (Test-Path $PythonExe)) {
    $resolvedPythonExe = (Resolve-Path $PythonExe).Path
}
if (-not $resolvedPythonExe) {
    $pythonCmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $resolvedPythonExe = $pythonCmd.Source
    }
}
if (-not $resolvedPythonExe -and $PythonExe -eq "python") {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $resolvedPythonExe = $pyLauncher.Source
        $pythonPrefixArgs = @("-3")
    }
}
if (-not $resolvedPythonExe) {
    throw "Could not resolve Python executable '$PythonExe'. Pass -PythonExe with a full path or ensure python/py is on PATH."
}

$runName = if ($env:RUN_NAME) { $env:RUN_NAME } else { "${ExpPrefix}_${Timestamp}" }
$datasetDir = if ($env:EXPORT_REPLAY_DATASET_DIR) { $env:EXPORT_REPLAY_DATASET_DIR } else { Join-Path $RepoRoot "local\ogbench_manip_datasets" }
$datasetPath = if ($ExportReplayDatasetPath) { $ExportReplayDatasetPath } else { Join-Path $datasetDir "cube_single_task1_human_vr_replay_latest.npz" }
$logDir = Join-Path $RepoRoot "logs"
$logPath = Join-Path $logDir "${runName}.log"
$pythonPathEntries = @(
    $RepoRoot,
    (Join-Path $RepoRoot "fasttd3"),
    (Join-Path $RepoRoot "ogbench")
)
$pythonPathValue = (($pythonPathEntries + @($env:PYTHONPATH)) | Where-Object { $_ -and $_.Trim().Length -gt 0 }) -join ';'

New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$cmd = @(
    ".\train_fast_sac_ogbench_manip.py",
    "--env_name", $EnvName,
    "--num_envs", "1",
    "--max_episode_steps", "$MaxEpisodeSteps",
    "--total_timesteps", "$TotalTimesteps",
    "--device", "auto",
    "--train_render_mode", "human",
    "--obs_mode", "state",
    "--no_include_goal",
    "--include_relative_cube_features",
    "--relative_only_obs",
    "--reward_type", "sparse",
    "--cube_reward_mode", "dense",
    "--disable_rotation",
    "--use_intervention",
    "--intervention_mode", "human",
    "--human_input_device", "vr",
    "--vr_mode", "connect",
    "--vr_host", $VrHost,
    "--vr_port", "$VrPort",
    "--teacher_type", "cube_markov",
    "--tolerance_type", "angle",
    "--tolerance_value", "30.0",
    "--hard_block_lethal",
    "--num_critics", "$NumCritics",
    "--actor_hidden_dim", "$ActorHiddenDim",
    "--critic_hidden_dim", "$CriticHiddenDim",
    "--batch_size", "$BatchSize",
    "--num_updates", "$NumUpdates",
    "--cta_ratio", "$CtaRatio",
    "--learning_starts", "$LearningStarts",
    "--gamma", "$Gamma",
    "--alpha_init", "$AlphaInit",
    "--fixed_alpha", "$FixedAlpha",
    "--alpha_min", "$AlphaMin",
    "--alpha_max", "$AlphaMax",
    "--use_layer_norm",
    "--demo_buffer_enable",
    "--demo_prefill_episodes", "$DemoPrefillEpisodes",
    "--demo_prefill_num_envs", "$DemoPrefillNumEnvs",
    "--demo_prefill_target", "demo",
    "--demo_prefill_intervention_mode", "agent_always",
    "--demo_sample_ratio", "$DemoSampleRatio",
    "--pref_buffer_enable",
    "--pref_sampling_mode", "linked",
    "--pref_sample_ratio", "0.0",
    "--pref_rank_weight", "$PrefRankWeight",
    "--pref_rank_margin", "$PrefRankMargin",
    "--pref_critic_scope", $PrefCriticScope,
    "--pref_loss_type", $PrefLossType,
    "--pref_lambda_init", "1.0",
    "--pref_lambda_lr", "1e-3",
    "--pref_lambda_max", "10.0",
    "--pref_lambda_ema", "0.9",
    "--pref_violation_clip", "10.0",
    "--pref_violation_target", "0.0",
    "--pref_lagrangian_violation_type", "hinge",
    "--pref_stopgrad_positive",
    "--intervention_episode_prob", "$InterventionEpisodeProb",
    "--intervention_episode_prob_min", "$InterventionEpisodeProb",
    "--intervention_episode_prob_decay_steps", "0",
    "--eval_interval", "$EvalInterval",
    "--num_eval_episodes", "5",
    "--eval_num_envs", "5",
    "--eval_render_mode", "none",
    "--save_interval", "$SaveInterval",
    "--log_interval", "$LogInterval",
    "--use_wandb",
    "--project", $Project,
    "--exp_name", $runName,
    "--export_replay_dataset_interval", "$ExportReplayDatasetInterval",
    "--export_replay_dataset_path", $datasetPath,
    "--export_replay_dataset_label", "human_vr_online_replay",
    "--tolerance_adaptive_near_distance", "0.08",
    "--tolerance_adaptive_far_distance", "0.30",
    "--tolerance_adaptive_near_scale", "0.35"
)

if (-not $VisualizeInterventionColors.IsPresent) {
    $cmd += "--no_visualize_intervention_colors"
}

Write-Host "Starting manipulation human-VR collection run" -ForegroundColor Cyan
Write-Host "repo:            $RepoRoot"
Write-Host "python:          $resolvedPythonExe $($pythonPrefixArgs -join ' ')"
Write-Host "run:             $runName"
Write-Host "project:         $Project"
Write-Host "vr endpoint:     $VrHost`:$VrPort"
Write-Host "dataset path:    $datasetPath"
Write-Host "log path:        $logPath"
Write-Host "PYTHONPATH:      $pythonPathValue"
Write-Host "timesteps:       $TotalTimesteps"
Write-Host "updates / CTA:   $NumUpdates / $CtaRatio"
Write-Host "fixed alpha:     $FixedAlpha"

Push-Location $RepoRoot
try {
    $env:WANDB_MODE = $WandbMode
    $env:WANDB_CONSOLE = "off"
    $env:WANDB_SILENT = "true"
    $env:PYTHONPATH = $pythonPathValue
    & $resolvedPythonExe @pythonPrefixArgs @cmd 2>&1 | Tee-Object -FilePath $logPath
}
finally {
    Pop-Location
}
