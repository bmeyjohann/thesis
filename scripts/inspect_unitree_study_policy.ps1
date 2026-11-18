param(
    [ValidateSet('fixed1','alpha_student','alpha_all','linked','bc','lambda_tolerant')]
    [string]$Policy = 'alpha_student',
    [ValidateRange(1024,65535)][int]$Port = 8081,
    [int]$Seed = 941,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$localRoot = '/home/benjamin/thesis/artifacts/unitree_budget25_20261009/local_core_20261010_r7/results/runs'
$desktopRoot = '/home/benjamin/thesis/artifacts/unitree_budget25_20261009/desktop_snapshot_20261010T0717/results/runs'
$models = @{
    fixed1 = "$localRoot/fixed1_seed73_4500/final.pt"
    alpha_student = "$localRoot/alpha_student_seed73_4500/final.pt"
    alpha_all = "$localRoot/alpha_all_seed73_4500/final.pt"
    linked = "$localRoot/linked_fixed1_seed73_4500/final.pt"
    bc = "$desktopRoot/bc_rl_seed73_4500/final.pt"
    lambda_tolerant = "$desktopRoot/lambda_tolerant_seed73_4500/final.pt"
}
$entry = '/mnt/c/home/benjamin/thesis/scripts/inspect_unitree_study_policy.py'
$python = '/home/benjamin/miniconda3/envs/fasttd3/bin/python'
foreach ($path in @($python, $entry, $models[$Policy])) {
    & wsl.exe -d Ubuntu --exec /usr/bin/test -f $path
    if ($LASTEXITCODE -ne 0) { throw "Required WSL file is unavailable: $path" }
}
Write-Host "Policy-only inspection: $Policy, seed $Seed, navigation 5 Hz; no training or teacher gate."
Write-Host "Checkpoint: $($models[$Policy])"
Write-Host "Open http://localhost:$Port after the Viser server is ready."
$extra = @()
if ($CheckOnly) { $extra += '--check-only' }
& wsl.exe -d Ubuntu --exec $python $entry --controller policy --model-path $models[$Policy] --checkpoint-env-config --viewer-backend viser --viser-port $Port --seed $Seed --navigation-policy-hz 5 --sim-fps 20 --human-input-device none --autopilot @extra
exit $LASTEXITCODE
