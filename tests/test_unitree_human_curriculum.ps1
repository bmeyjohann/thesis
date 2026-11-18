$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repo ('local\curriculum_test_' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($root)
Copy-Item (Join-Path $repo 'scripts\run_unitree_human_curriculum_windows.ps1') $root
$fake = @'
param($Mode,$Sensors,$Steps,$Seed,$RunName,$ReceiptPath,$ImageProfile,[switch]$StudyProfile)
if (-not $StudyProfile) { throw 'StudyProfile not passed' }
Add-Content -LiteralPath (Join-Path $PSScriptRoot 'calls.txt') -Value $RunName
if (Test-Path (Join-Path $PSScriptRoot 'fail')) { throw 'Simulated child failure' }
$complete = -not (Test-Path (Join-Path $PSScriptRoot 'partial'))
@{ limit_reached=$complete; navigation_decisions=$(if($complete){$Steps}else{1});
   run_name=$RunName; mode=$Mode; sensors=$Sensors; seed=[int]$Seed; image_profile=$ImageProfile; checkpoint='mock-final.pt' } |
   ConvertTo-Json | Set-Content -LiteralPath $ReceiptPath
'@
Set-Content (Join-Path $root 'run_unitree_target_human_training_windows.ps1') $fake
$launcher = Join-Path $root 'run_unitree_human_curriculum_windows.ps1'
$session = Join-Path $root 'success'
& $launcher -SessionPath $session -Steps 2
$saved = Get-Content (Join-Path $session 'curriculum.json') -Raw | ConvertFrom-Json
if ($saved.Runs.Count -ne 4 -or @($saved.Runs | Where-Object Status -ne 'complete').Count) { throw 'Four-run completion failed' }
& $launcher -SessionPath $session -Repetitions 3
$saved = Get-Content (Join-Path $session 'curriculum.json') -Raw | ConvertFrom-Json
if ($saved.Runs.Count -ne 12 -or @(Get-Content (Join-Path $root 'calls.txt')).Count -ne 12) { throw 'Expansion repeated a run' }
& $launcher -SessionPath $session
if (@(Get-Content (Join-Path $root 'calls.txt')).Count -ne 12) { throw 'Completed runs repeated' }
if ($saved.Profile -ne 'stabilized_visual_goal_v3') { throw 'New visual profile not selected' }
$legacyPath = Join-Path $root 'legacy'
& $launcher -SessionPath $legacyPath -Steps 2 -PlanOnly
$legacy = Get-Content (Join-Path $legacyPath 'curriculum.json') -Raw | ConvertFrom-Json
$legacy.Profile = 'desktop_history_candidate_v1'
$legacy | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $legacyPath 'curriculum.json')
& $launcher -SessionPath $legacyPath
$receipt = Get-Content $legacy.Runs[0].Receipt -Raw | ConvertFrom-Json
if ($receipt.image_profile -ne 'legacy') { throw 'Old session silently changed image profile' }
$v1Path = Join-Path $root 'v1'
& $launcher -SessionPath $v1Path -Steps 2 -PlanOnly
$v1 = Get-Content (Join-Path $v1Path 'curriculum.json') -Raw | ConvertFrom-Json
$v1.Profile = 'stabilized_visual_goal_v1'
$v1 | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $v1Path 'curriculum.json')
& $launcher -SessionPath $v1Path
$receipt = Get-Content $v1.Runs[0].Receipt -Raw | ConvertFrom-Json
if ($receipt.image_profile -ne 'stabilized_goal_v1') { throw 'V1 session silently changed image profile' }
[void](New-Item (Join-Path $root 'partial') -ItemType File)
& $launcher -SessionPath (Join-Path $root 'early') -Steps 2
$early = Get-Content (Join-Path $root 'early\curriculum.json') -Raw | ConvertFrom-Json
if ($early.Runs[0].Status -ne 'partial' -or $early.Runs[1].Status -ne 'pending') { throw 'Early stop advanced curriculum' }
$blocked = $false
try { & $launcher -SessionPath (Join-Path $root 'early') } catch { $blocked = $true }
if (-not $blocked) { throw 'Partial run incorrectly retried' }
[void](New-Item (Join-Path $root 'fail') -ItemType File)
$blocked = $false
try { & $launcher -SessionPath (Join-Path $root 'failure') -Steps 2 } catch { $blocked = $true }
$failed = Get-Content (Join-Path $root 'failure\curriculum.json') -Raw | ConvertFrom-Json
if (-not $blocked -or $failed.Runs[0].Status -ne 'failed' -or $failed.Runs[1].Status -ne 'pending') { throw 'Failed child advanced curriculum' }
Write-Host 'PASS: four conditions, twelve-run expansion, no duplicates, early stop, failed child.'
