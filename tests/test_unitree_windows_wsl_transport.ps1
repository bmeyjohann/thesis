$ErrorActionPreference = 'Stop'
$launcher = Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\run_unitree_target_human_training_windows.ps1'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($launcher, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$calls = @($ast.FindAll({ param($node)
    $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'wsl.exe'
}, $true))
foreach ($call in $calls) {
    if ($call.Extent.Text -notmatch '--exec\s') { throw 'WSL commands must bypass shell parsing.' }
}
$pathCall = @($calls | Where-Object { $_.Extent.Text -match 'wslpath -u' })
if ($pathCall.Count -ne 1) { throw 'Expected one receipt conversion.' }
$sender = $ast.Find({ param($node)
    $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -eq 'Start-Process'
}, $true)
if ($pathCall[0].Extent.StartOffset -gt $sender.Extent.StartOffset) { throw 'Validate receipt before starting sender.' }
foreach ($ReceiptPath in @(
    'C:\home\benjamin\thesis\local\human_curricula\20261103_112159_4e64be9c\r1_rgb_height_pref_rl.json',
    "C:\home\benjamin\thesis\local\receipt space's & literal\receipt.json"
)) {
    # Execute the actual launcher's command AST, not a separately maintained copy.
    $linuxReceipt = & ([scriptblock]::Create($pathCall[0].Extent.Text))
    if ($LASTEXITCODE -ne 0) { throw 'Receipt conversion failed.' }
    $expected = '/mnt/c/' + $ReceiptPath.Substring(3).Replace('\', '/')
    if ($linuxReceipt -cne $expected) { throw "Receipt changed: $linuxReceipt" }
    $adapter = @{IPAddress='172.28.48.1'}
    $Mode='pref_rl'; $Steps=4500; $Sensors='rgb_height'; $Seed=7
    $EncoderCheckpoint="/tmp/encoder space's & literal.pt"
    $extraEnvironment = @("ONLINE_SESSION_RECEIPT=$linuxReceipt", 'NAVIGATION_FEATURE_LAGS=0 1 25')
    $launchCall = @($calls | Where-Object { $_.Extent.Text -match '--exec env' })
    if ($launchCall.Count -ne 1) { throw 'Expected one training invocation.' }
    # Replace only the expensive training program with an environment readback.
    $probe = $launchCall[0].Extent.Text.Replace('bash scripts/run_unitree_target_human_training_local.sh', '/usr/bin/printenv')
    $output = & ([scriptblock]::Create($probe))
    if ($LASTEXITCODE -ne 0) { throw 'Environment transport failed.' }
    foreach ($value in @("ONLINE_SESSION_RECEIPT=$expected", 'NAVIGATION_FEATURE_LAGS=0 1 25',
        "NAVIGATION_ENCODER_CHECKPOINT=$EncoderCheckpoint", 'SEED=7', 'ONLINE_TOTAL_STEPS=4500')) {
        if ($output -cnotcontains $value) { throw "Environment value changed: $value" }
    }
}
Write-Host 'PASS: actual launcher path and environment transport, including spaces and shell metacharacters; no training started.'
