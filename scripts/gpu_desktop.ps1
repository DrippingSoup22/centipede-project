# What scripts/gpu_desktop.py runs on the GPU desktop. The laptop copies this
# file into the desktop's home folder before every command, then calls one of
# the functions below over SSH.

$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Project = "$HOME\Projects\Centipede"
$Python = "$HOME\.venvs\Centipede\Scripts\python.exe"
$Launched = "$Project\runs\launched"
# The last line of every run's output, written after its process ends,
# followed by the exit code.
$EndMarker = 'run ended with exit code'
Set-Location $Project

# Pulls the pushed code, but never new commits while a run or queue runs: each
# run of a queue starts with the code on disk, so its next runs would change.
function Update-Code {
    git fetch -q
    if ($LASTEXITCODE) { Write-Output 'git fetch failed on the desktop'; exit 1 }
    $behind = [int](git rev-list --count 'HEAD..@{upstream}')
    $running = @(Get-Runs)
    if ($behind -and $running.Count) {
        $names = ($running | ForEach-Object { Get-RunName $_ } | Sort-Object -Unique) -join ', '
        Write-Output "not pulling $behind new commits while $names runs: the runs a queue"
        Write-Output 'starts next would use them. Try again when it has ended.'
        exit 1
    }
    git pull --ff-only -q
    if ($LASTEXITCODE) { Write-Output 'git pull failed on the desktop'; exit 1 }
    Write-Output ('code ' + (git log --oneline -1))
}

function Invoke-Tests {
    Update-Code
    & $Python -m pytest -q
    exit $LASTEXITCODE
}

# The running training, evaluation, and queue processes: two per run,
# because a virtual environment's python.exe starts the real interpreter as a
# second process with the same arguments.
function Get-Runs {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like '*-m centipede*' -or $_.CommandLine -like '*run_queue.py*' }
}

# A run's name is its configuration file's name in runs\launched, the last
# argument of its command; a queue's is its folder's name there.
function Get-RunName($process) {
    $last = ($process.CommandLine.Trim() -split '\s+')[-1].Trim('"')
    [IO.Path]::GetFileNameWithoutExtension($last)
}

function Test-Running($name) {
    [bool](Get-Runs | Where-Object { (Get-RunName $_) -eq $name })
}

# A console file's lines as a terminal shows them: the progress bar redraws
# its line with carriage returns, and only the last drawing stays. Lines end
# in CR LF, as Python writes them to a file on Windows.
function Get-ShownLines($console, $count) {
    # Opened for sharing, since a running run is still writing it.
    $stream = [IO.File]::Open($console, 'Open', 'Read', 'ReadWrite')
    $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
    $text = $reader.ReadToEnd().Replace("`r`n", "`n")
    $reader.Close()
    $lines = $text.TrimEnd("`n").Split("`n") | ForEach-Object { $_.Split("`r")[-1] }
    $lines | Select-Object -Last $count
}

function Get-RunState($console) {
    if (Test-Running $console.BaseName) { return 'running' }
    $last = Get-ShownLines $console.FullName 1
    if ($last -eq "$EndMarker 0") { return 'finished' }
    if ($last -like "$EndMarker *") {
        return 'failed, exit code ' + $last.Split(' ')[-1]
    }
    if ($last -eq 'stopped from the laptop') { return 'stopped' }
    'ended without its last line'
}

# The latest launched run whose name contains $pattern, after 'queue' or
# 'run'. A queue keeps its folder in runs\launched, and its runs are named
# after it: where a queue matches, its runs are left out, since the queue is
# the one meant.
function Find-Run($pattern) {
    $consoles = @(Get-ChildItem "$Launched\*$pattern*.txt" -ErrorAction SilentlyContinue)
    $queues = @($consoles | Where-Object { Test-Path "$Launched\$($_.BaseName)" -PathType Container } |
        ForEach-Object { $_.BaseName })
    $consoles = @($consoles | Where-Object {
        $name = $_.BaseName
        -not ($queues | Where-Object { $name.StartsWith("${_}_") })
    } | Sort-Object LastWriteTime)
    if ($consoles.Count) {
        $name = $consoles[-1].BaseName
        $kind = if ($queues -contains $name) { 'queue' } else { 'run' }
        Write-Output "$kind $name"
    }
}

# The run folder a launched run wrote into, from the paths it printed:
# "Run folder: runs\<folder>" in training, "Results: runs\<folder>\..." in
# evaluation.
function Get-RunFolder($name) {
    $console = "$Launched\$name.txt"
    if (-not (Test-Path $console)) { return }
    $found = Select-String -Path $console -Pattern '(Run folder|Results): (.*\\)?runs\\([^\\]+)' |
        Select-Object -First 1
    if ($found) { Write-Output $found.Matches[0].Groups[3].Value.Trim() }
}

# Refuses to start while another run or queue is running, unless $alongside.
function Assert-Idle($alongside) {
    $running = @(Get-Runs)
    if ($running.Count -and -not $alongside) {
        $names = ($running | ForEach-Object { Get-RunName $_ } | Sort-Object -Unique) -join ', '
        Write-Output "already running: $names"
        Write-Output ('Two runs share the GPU and slow each other;' +
            ' add --alongside to start anyway.')
        exit 1
    }
}

# Starts Python with $arguments, its output in $console, detached from the
# SSH session: Windows stops what an SSH session started when it disconnects,
# but not what its process service (WMI) starts. cmd's !errorlevel!, with
# /v:on, is read after the process has ended.
function Start-Detached($arguments, $console) {
    $command = "cmd /v:on /c `"set PYTHONUTF8=1&& `"$Python`" -u $arguments " +
        "> `"$console`" 2>&1 & echo $EndMarker !errorlevel!>> `"$console`"`""
    $started = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $command; CurrentDirectory = $Project }
    if ($started.ReturnValue) { Write-Output 'the run could not be started'; exit 1 }
}

# Starts a run from the configuration file the laptop copied.
function Start-Run($name, $configurationCopy, $alongside) {
    Assert-Idle $alongside
    Update-Code
    New-Item -ItemType Directory -Force $Launched | Out-Null
    $file = "$Launched\$name.toml"
    Move-Item -Force $configurationCopy $file
    Start-Detached "-m centipede `"$file`"" "$Launched\$name.txt"
}

# Starts a queue from the folder of configuration files the laptop copied:
# scripts\run_queue.py runs them one after another.
function Start-Queue($name, $folderCopy, $alongside) {
    Assert-Idle $alongside
    Update-Code
    New-Item -ItemType Directory -Force $Launched | Out-Null
    $folder = "$Launched\$name"
    Move-Item -Force $folderCopy $folder
    Start-Detached "scripts\run_queue.py `"$folder`"" "$Launched\$name.txt"
}

# The names of a queue's runs and evaluations, in order.
function Get-QueueItems($name) {
    Get-ChildItem "$Launched\${name}_*.txt" -ErrorAction SilentlyContinue |
        Sort-Object Name | ForEach-Object { Write-Output $_.BaseName }
}

# Passes a run's output through as it is written, from its start, carriage
# returns included, until its end line, which it leaves out; exits with the
# run's exit code.
function Watch-Run($name) {
    $console = "$Launched\$name.txt"
    for ($wait = 0; -not (Test-Path $console) -and $wait -lt 30; $wait++) {
        Start-Sleep 1
    }
    if (-not (Test-Path $console)) { Write-Output "no output for $name"; exit 1 }
    $stream = [IO.File]::Open($console, 'Open', 'Read', 'ReadWrite')
    $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
    $buffer = New-Object char[] 8192
    $recent = ''
    $stillRunning = $true
    while ($true) {
        $count = $reader.Read($buffer, 0, $buffer.Length)
        if ($count -gt 0) {
            $text = [string]::new($buffer, 0, $count)
            [Console]::Out.Write(($text -replace "$EndMarker -?\d+\r?\n?", ''))
            [Console]::Out.Flush()
            $recent = $recent + $text
            if ($recent.Length -gt 400) { $recent = $recent.Substring($recent.Length - 400) }
            if ($recent -match "$EndMarker (-?\d+)") { exit [int]$Matches[1] }
            continue
        }
        if (-not $stillRunning) {
            Write-Output 'the run is no longer running'
            exit 1
        }
        # One more pass after the process ends, for its last lines.
        if (-not (Test-Running $name)) { $stillRunning = $false; Start-Sleep 2 }
        else { Start-Sleep -Milliseconds 500 }
    }
}

function Show-Status($recentShown, $linesShown) {
    $running = @(Get-Runs | Group-Object { Get-RunName $_ })
    foreach ($run in $running) {
        $since = ($run.Group | Sort-Object CreationDate)[0].CreationDate.ToString('HH:mm')
        Write-Output "running since ${since}: $($run.Name)"
    }
    if (-not $running.Count) { Write-Output 'no run is running' }
    $consoles = @(Get-ChildItem "$Launched\*.txt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last $recentShown)
    if ($consoles.Count) {
        Write-Output ''
        Write-Output 'recent runs:'
        foreach ($console in $consoles) {
            Write-Output "  $($console.BaseName): $(Get-RunState $console)"
        }
        Write-Output ''
        Write-Output "latest output, $($consoles[-1].BaseName):"
        Get-ShownLines $consoles[-1].FullName $linesShown
    }
    exit 0
}

function Stop-Runs {
    # A queue first, so that it starts nothing after its run is stopped.
    $running = @(Get-Runs | Group-Object { Get-RunName $_ } |
        Sort-Object { -not ($_.Group[0].CommandLine -like '*run_queue.py*') })
    foreach ($run in $running) {
        $run.Group | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep 1
        Add-Content "$Launched\$($run.Name).txt" 'stopped from the laptop'
        Write-Output "stopped $($run.Name); it keeps its last checkpoint"
    }
    if (-not $running.Count) { Write-Output 'no run is running' }
    exit 0
}
