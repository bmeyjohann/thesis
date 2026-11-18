param(
    [ValidateRange(1, 3)][int]$Repetitions = 1,
    [ValidateRange(1, 1000000)][int]$Steps = 4500,
    [int]$FirstSeed = 7,
    [string]$SessionPath = '',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'run_unitree_target_human_training_windows.ps1'
if (-not (Test-Path -LiteralPath $launcher)) { throw "Missing training launcher: $launcher" }
$root = Split-Path -Parent $PSScriptRoot
$conditions = @(
    @{ Sensors = 'rgb_height'; Mode = 'pref_rl' },
    @{ Sensors = 'php_depth'; Mode = 'pref_rl' },
    @{ Sensors = 'rgb_height'; Mode = 'pref_bc_rl' },
    @{ Sensors = 'php_depth'; Mode = 'pref_bc_rl' }
)
if (-not $SessionPath) {
    $SessionPath = Join-Path $root ('local\human_curricula\' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8))
}
$SessionPath = [IO.Path]::GetFullPath($SessionPath)
[void][IO.Directory]::CreateDirectory($SessionPath)
$statePath = Join-Path $SessionPath 'curriculum.json'
$lockPath = Join-Path $SessionPath 'session.lock'
$lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
try {
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.Profile -notin @('desktop_history_candidate_v1', 'stabilized_visual_goal_v1', 'stabilized_visual_goal_v2', 'stabilized_visual_goal_v3')) { throw 'Unknown saved curriculum profile.' }
        if (($PSBoundParameters.ContainsKey('Steps') -and $Steps -ne $state.Steps) -or
            ($PSBoundParameters.ContainsKey('FirstSeed') -and $FirstSeed -ne $state.FirstSeed)) {
            throw 'A resumed curriculum keeps its original steps and seed.'
        }
        $Steps = $state.Steps
        $FirstSeed = $state.FirstSeed
        $state.Runs = @($state.Runs)
    } else {
        $state = [pscustomobject]@{ Profile='stabilized_visual_goal_v3'; Steps=$Steps; FirstSeed=$FirstSeed; Runs=@() }
    }
    $imageProfile = switch ($state.Profile) {
        'stabilized_visual_goal_v3' { 'stabilized_goal_v3' }
        'stabilized_visual_goal_v2' { 'stabilized_goal_v2' }
        'stabilized_visual_goal_v1' { 'stabilized_goal_v1' }
        default { 'legacy' }
    }
    function Test-ImageProfile($receipt) {
        if (-not $receipt.image_profile) { return $imageProfile -eq 'legacy' }
        return $receipt.image_profile -eq $imageProfile
    }
    for ($repeat=0; $repeat -lt $Repetitions; $repeat++) {
        # Rotate order between repeats to reduce a fixed ordering effect on the operator.
        for ($slot=0; $slot -lt 4; $slot++) {
            $condition = $conditions[($slot + $repeat) % 4]
            $id = "r$($repeat+1)_$($condition.Sensors)_$($condition.Mode)"
            if (@($state.Runs | Where-Object { $_.Id -eq $id }).Count) { continue }
            $state.Runs += [pscustomobject]@{
                Id=$id; Sensors=$condition.Sensors; Mode=$condition.Mode; Seed=($FirstSeed+$repeat)
                RunName=('human_' + (Split-Path $SessionPath -Leaf) + '_' + $id)
                Status='pending'; Receipt=(Join-Path $SessionPath ($id + '.json'))
            }
        }
    }
    function Save-State {
        $state | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath ($statePath + '.tmp') -Encoding UTF8
        Move-Item -LiteralPath ($statePath + '.tmp') -Destination $statePath -Force
    }
    Save-State
    Write-Host "Curriculum: $statePath"
    Write-Host 'Profile: feature ages 0/0.2/5s; actor/critic LR 0.0001/0.0003; n1; TD/pref256; entropy -3; lambda1; BC arms 0.1.'
    Write-Host "Visual input: $imageProfile (saved curriculum profile is preserved)."
    $state.Runs | Select-Object Id,Seed,Status | Format-Table | Out-Host
    if ($PlanOnly) { return }
    foreach ($run in $state.Runs) {
        if ($run.Status -eq 'complete') { continue }
        if ($run.Status -ne 'pending') {
            # A launcher interruption may happen after the learner saved successfully.
            if (Test-Path -LiteralPath $run.Receipt) {
                $receipt = Get-Content -LiteralPath $run.Receipt -Raw | ConvertFrom-Json
                if ($receipt.limit_reached -and $receipt.navigation_decisions -eq $Steps -and
                    $receipt.run_name -eq $run.RunName -and $receipt.mode -eq $run.Mode -and
                    $receipt.sensors -eq $run.Sensors -and $receipt.seed -eq $run.Seed -and (Test-ImageProfile $receipt)) {
                    $run.Status = 'complete'; Save-State; continue
                }
            }
            throw "Run $($run.Id) is $($run.Status). Its saved work is preserved; it will not be repeated automatically."
        }
        Write-Host "Starting $($run.Id), seed $($run.Seed), $Steps decisions. Open http://localhost:8080. Each run starts paused."
        $run.Status = 'running'; Save-State
        try {
            & $launcher -Mode $run.Mode -Sensors $run.Sensors -Steps $Steps -Seed $run.Seed -RunName $run.RunName -ReceiptPath $run.Receipt -ImageProfile $imageProfile -StudyProfile
            if (-not (Test-Path -LiteralPath $run.Receipt)) { throw 'Training returned without a saved completion receipt.' }
            $receipt = Get-Content -LiteralPath $run.Receipt -Raw | ConvertFrom-Json
            if ($receipt.run_name -ne $run.RunName -or $receipt.mode -ne $run.Mode -or
                $receipt.sensors -ne $run.Sensors -or $receipt.seed -ne $run.Seed -or -not (Test-ImageProfile $receipt)) { throw 'Receipt does not match the requested run.' }
            if (-not $receipt.limit_reached -or $receipt.navigation_decisions -ne $Steps) {
                $run.Status = 'partial'; Save-State
                Write-Host "Stopped early; checkpoint saved at $($receipt.checkpoint). Curriculum stopped."
                return
            }
            $run.Status = 'complete'; Save-State
        } catch {
            if ($run.Status -eq 'running') { $run.Status = 'failed'; Save-State }
            throw
        }
    }
    Write-Host "All $($state.Runs.Count) human runs completed and saved. Receipts: $SessionPath"
} finally {
    $lock.Dispose()
}
