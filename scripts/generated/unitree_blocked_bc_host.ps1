$ErrorActionPreference = 'Stop'
$folder = 'C:\home\benjamin\thesis\artifacts\unitree_bc_blocked_20261012_r4\host'
[void][IO.Directory]::CreateDirectory($folder)
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class UnitreeBlockedPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$trace = Join-Path $folder 'host.trace'
$process = $null
try {
    if ([UnitreeBlockedPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Temporary sleep inhibitor failed'
    }
    [IO.File]::WriteAllText($trace, "started=$([DateTime]::UtcNow.ToString('o')) host_pid=$PID`n")
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = 'C:\Windows\System32\wsl.exe'
    $info.Arguments = '-d Ubuntu --exec /bin/bash /mnt/c/home/benjamin/thesis/scripts/generated/unitree_blocked_bc_foreground.sh'
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    [void]$process.Start()
    [IO.File]::AppendAllText($trace, "foreground_wsl_pid=$($process.Id)`n")
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit(259200000)) {
        [IO.File]::AppendAllText($trace, "host_deadline_exceeded`n")
        # Linux has an earlier timeout and cleans up its owned child sessions.
        & 'C:\Windows\System32\taskkill.exe' /PID $process.Id /T /F | Out-Null
        throw '72-hour host deadline exceeded; inspect preserved artifacts'
    }
    [IO.File]::WriteAllText((Join-Path $folder 'wsl.stdout'), $stdout.GetAwaiter().GetResult())
    [IO.File]::WriteAllText((Join-Path $folder 'wsl.stderr'), $stderr.GetAwaiter().GetResult())
    [IO.File]::AppendAllText($trace, "exit=$($process.ExitCode) finished=$([DateTime]::UtcNow.ToString('o'))`n")
    exit $process.ExitCode
} finally {
    [void][UnitreeBlockedPower]::SetThreadExecutionState([uint32]2147483648)
}
