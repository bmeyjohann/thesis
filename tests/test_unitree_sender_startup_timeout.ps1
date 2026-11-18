$ErrorActionPreference = 'Stop'
$launcher = Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\run_unitree_target_human_training_windows.ps1'
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($launcher,[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Receive-UnitreeStartupPacket'},$true)
Invoke-Expression $function.Extent.Text

# An unresolved connect must be cancellable even if the OS never responds.
$fake=[pscustomobject]@{ Pending=[Threading.Tasks.TaskCompletionSource[bool]]::new(); Disposed=$false }
$fake | Add-Member ScriptMethod ConnectAsync { param($address,$port) return $this.Pending.Task }
$fake | Add-Member ScriptMethod Dispose { $this.Disposed=$true }
$timer=[Diagnostics.Stopwatch]::StartNew()
$result=Receive-UnitreeStartupPacket -TimeoutMs 150 -ClientFactory { $fake }
if ($null -ne $result -or -not $fake.Disposed -or $timer.ElapsedMilliseconds -gt 1500) { throw 'Blocked connect exceeded bound or was not disposed.' }

# A successful handshake without a packet must also time out.
$listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
$listener.Start()
try {
    $port=$listener.LocalEndpoint.Port
    $timer.Restart()
    $result=Receive-UnitreeStartupPacket -Port $port -TimeoutMs 200
    if ($null -ne $result -or $timer.ElapsedMilliseconds -gt 1500) { throw 'Silent listener exceeded packet timeout.' }
} finally { $listener.Stop() }

Write-Host 'PASS: stalled connect and silent publisher are bounded and disposed; launcher syntax valid.'
