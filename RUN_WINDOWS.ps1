param([switch]$Base, [ValidateRange(0,5)][int]$Scene = 1,
    [ValidateSet('original','toi','compensated','gated')][string]$Termination = 'compensated')
$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "build/gipc.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "Build first with BUILD_WINDOWS.cmd" }
$env:PATH = (Join-Path $PSScriptRoot "build") + ";C:\Users\hj0320\AppData\Local\Temp\vcpkg_installed\x64-windows\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:PATH
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $PSScriptRoot ("run-" + $stamp + ".stdout.log")
$stderr = Join-Path $PSScriptRoot ("run-" + $stamp + ".stderr.log")
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $env:ComSpec
$startInfo.WorkingDirectory = $PSScriptRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$solverArgs = ' --scene ' + $Scene
if (-not $Base) { $solverArgs += ' --pcg-dual-graph --termination ' + $Termination }
$startInfo.Arguments = '/d /s /c ""' + $exe + '"' + $solverArgs + ' 1>"' + $stdout + '" 2>"' + $stderr + '""'
$launcher = [System.Diagnostics.Process]::Start($startInfo)
if ($launcher.WaitForExit(2000)) { throw ("StiffGIPC exited during startup: " + $launcher.ExitCode) }
$child = Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $launcher.Id) | Where-Object Name -eq "gipc.exe" | Select-Object -First 1
if (-not $child) { throw "StiffGIPC process not found" }
[PSCustomObject]@{pid=$child.ProcessId;launcher_pid=$launcher.Id;exe=$exe;stdout=$stdout;stderr=$stderr;started=(Get-Date).ToString("o");launch_mode="console hidden; GLUT viewer visible"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot "last-run.json") -Encoding utf8
Write-Output ("Started StiffGIPC PID " + $child.ProcessId)
Write-Output ("Log: " + $stdout)
