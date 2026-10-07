"""Run the project on the GPU desktop from the laptop, over SSH.

The desktop holds its own copy of this repository, RL_lib, and MujocoReplay,
and its own Python environment (README, Setup). This script, run on the
laptop, sends it commands:

    python scripts/gpu_desktop.py update      # git pull on the desktop
    python scripts/gpu_desktop.py tests       # pull, then run the tests there
    python scripts/gpu_desktop.py run configs/baseline.toml
    python scripts/gpu_desktop.py status      # running runs, latest output
    python scripts/gpu_desktop.py stop        # stop every running run

Run it from the repository root, with the project's Python environment.

``run`` pulls the pushed code, then sends the laptop's configuration file as it
is, so a changed setting needs no commit. The file and the run's console
output are kept in the desktop's ``runs/launched/``. The run is started by
Windows' process service rather than by the SSH session, because Windows
stops everything an SSH session started when it disconnects; the run keeps
going after this script returns. Its folder in ``runs/`` reaches the laptop
through the folder synchronisation (README, Training machine).

``CENTIPEDE_GPU_HOST`` names the desktop; by default ``gpu``, an alias in
``~/.ssh/config``. Every command goes to the desktop's Windows PowerShell,
encoded, so that nothing in it needs quoting twice.
"""

import argparse
import base64
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from centipede.experiment.configuration import read_configuration
from centipede.settings_section import SettingsError

HOST = os.environ.get("CENTIPEDE_GPU_HOST", "gpu")
REPOSITORY = Path(__file__).resolve().parents[1]
# PowerShell expands $HOME on the desktop.
REMOTE_PROJECT = r"$HOME\Projects\Centipede"
REMOTE_PYTHON = r"$HOME\.venvs\Centipede\Scripts\python.exe"
# Code that must be pushed before the desktop can run it; configuration files
# are sent as they are.
CODE_PATHS = ("src", "models", "tests", "benchmarks", "pyproject.toml")
CONSOLE_LINES_SHOWN = 8

PRELUDE = """
$ProgressPreference = 'SilentlyContinue'
Set-Location "{project}"
function Update-Code {{
    git pull --ff-only -q
    if ($LASTEXITCODE) {{ Write-Output 'git pull failed on the desktop'; exit 1 }}
    Write-Output ("desktop at " + (git log --oneline -1))
}}
function Get-Runs {{
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {{ $_.CommandLine -like '*-m centipede*' }}
}}
"""


def remote(script: str) -> int:
    """Run ``script`` in the desktop's PowerShell, in the project; its exit code."""
    full_script = PRELUDE.format(project=REMOTE_PROJECT) + script
    encoded = base64.b64encode(full_script.encode("utf-16-le")).decode("ascii")
    command = f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command]).returncode


def git(*arguments: str) -> str:
    """A git command's output in the laptop's repository."""
    return subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, capture_output=True, text=True, check=True
    ).stdout.strip()


def require_pushed_code() -> None:
    """Stop unless the desktop can pull exactly the code the laptop has."""
    git("fetch", "-q")
    unpushed = int(git("rev-list", "--count", "@{upstream}..HEAD"))
    uncommitted = git(
        "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS
    )
    if unpushed or uncommitted:
        sys.exit(
            "The desktop runs the code on GitHub: commit and push first"
            f" ({unpushed} unpushed commits; uncommitted: {uncommitted or 'none'})."
        )


def update(_: argparse.Namespace) -> int:
    require_pushed_code()
    return remote("Update-Code")


def tests(_: argparse.Namespace) -> int:
    require_pushed_code()
    return remote(f'Update-Code\n& "{REMOTE_PYTHON}" -m pytest -q\nexit $LASTEXITCODE')


def run(arguments: argparse.Namespace) -> int:
    configuration_path: Path = arguments.configuration
    try:
        read_configuration(configuration_path)
    except SettingsError as error:
        sys.exit(f"Configuration error: {error}")
    require_pushed_code()
    name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{configuration_path.stem}"
    content = base64.b64encode(configuration_path.read_bytes()).decode("ascii")
    return remote(f"""
Update-Code
$launched = Join-Path (Get-Location) 'runs\\launched'
New-Item -ItemType Directory -Force $launched | Out-Null
$file = Join-Path $launched '{name}.toml'
$console = Join-Path $launched '{name}.txt'
[IO.File]::WriteAllBytes($file, [Convert]::FromBase64String('{content}'))
$python = "{REMOTE_PYTHON}"
$command = 'cmd /c "set PYTHONUTF8=1&& "' + $python + '" -u -m centipede "' +
    $file + '" > "' + $console + '" 2>&1"'
$started = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{
    CommandLine = $command; CurrentDirectory = (Get-Location).Path }}
if ($started.ReturnValue) {{ Write-Output 'the run could not be started'; exit 1 }}
Write-Output "started {name}; its output: runs\\launched\\{name}.txt"
""")


def status(_: argparse.Namespace) -> int:
    return remote(f"""
$runs = @(Get-Runs)
if ($runs.Count) {{
    foreach ($run in $runs) {{
        Write-Output ("running since " + $run.CreationDate.ToString('HH:mm') +
            ": " + ($run.CommandLine -replace '^.*-m centipede ', ''))
    }}
}} else {{ Write-Output 'no run is running' }}
$latest = Get-ChildItem runs\\launched\\*.txt -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($latest) {{
    Write-Output ""
    Write-Output ("latest output, " + $latest.Name + ":")
    Get-Content $latest.FullName -Tail {CONSOLE_LINES_SHOWN}
}}
exit 0
""")


def stop(_: argparse.Namespace) -> int:
    return remote("""
$runs = @(Get-Runs)
foreach ($run in $runs) { Stop-Process -Id $run.ProcessId -Force }
Write-Output ("stopped " + $runs.Count + " run(s); each keeps its last checkpoint")
exit 0
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    commands.add_parser("update", help="git pull on the desktop").set_defaults(
        action=update
    )
    commands.add_parser("tests", help="pull, then run the tests").set_defaults(
        action=tests
    )
    run_parser = commands.add_parser("run", help="pull, then start a run")
    run_parser.add_argument("configuration", type=Path, help="a TOML file")
    run_parser.set_defaults(action=run)
    commands.add_parser("status", help="running runs and latest output").set_defaults(
        action=status
    )
    commands.add_parser("stop", help="stop every running run").set_defaults(action=stop)
    arguments = parser.parse_args()
    sys.exit(arguments.action(arguments))


if __name__ == "__main__":
    main()
