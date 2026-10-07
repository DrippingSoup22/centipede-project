"""Run the project on the GPU desktop from the laptop, over SSH.

The desktop holds its own copy of this repository, RL_lib, and MujocoReplay,
and its own Python environment (README, Setup). This script, run on the
laptop from the repository root with the project's Python, sends it commands:

    python scripts/gpu_desktop.py run configs/baseline.toml   # start, then watch
    python scripts/gpu_desktop.py watch          # follow the latest run's output
    python scripts/gpu_desktop.py status         # running and recent runs
    python scripts/gpu_desktop.py stop           # stop every running run
    python scripts/gpu_desktop.py tests          # pull, then run the tests there
    python scripts/gpu_desktop.py update         # git pull on the desktop

``run`` pulls the pushed code, sends the laptop's configuration file as it is,
so a changed setting needs no commit, and starts the run. It then shows the
run's output as it is written, until the run ends; Ctrl+C stops watching, not
the run, and ``watch`` picks it up again. Each run's file and console output
are kept in the desktop's ``runs/launched/``; the output ends with the run's
exit code. Its run folder in ``runs/`` reaches the laptop through the folder
synchronisation (README, Training machine).

What runs on the desktop is ``gpu_desktop.ps1``: every command first copies
it, and ``run`` its configuration file, into the desktop's home folder with
scp, then sends one short call to its functions, encoded so that nothing
needs quoting twice. The whole file does not fit in one command: Windows cuts
commands that long. ``CENTIPEDE_GPU_HOST`` names the desktop; by default
``gpu``, an alias in ``~/.ssh/config``.
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
DESKTOP_FUNCTIONS = Path(__file__).with_suffix(".ps1")
# Where the copied files land, in the desktop user's home folder.
DESKTOP_FUNCTIONS_COPY = ".centipede_gpu_desktop.ps1"
CONFIGURATION_COPY = ".centipede_configuration.toml"
# Code that must be pushed before the desktop can run it; configuration files
# are sent as they are.
CODE_PATHS = ("src", "models", "tests", "benchmarks", "pyproject.toml")
RECENT_RUNS_SHOWN = 5
OUTPUT_LINES_SHOWN = 6
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
SSH_UNREACHABLE = 255
INTERRUPTED = 130


def unreachable() -> None:
    sys.exit(
        f"Could not reach the desktop ({HOST}): is it on and awake, and is its"
        " SSH server running?"
    )


def copy_to_desktop(source: Path, destination: str) -> None:
    """Copy one file into the desktop user's home folder."""
    command = ["scp", "-q", *SSH_OPTIONS, str(source), f"{HOST}:{destination}"]
    if subprocess.run(command).returncode:
        unreachable()


def on_desktop(call: str, watching: bool = False) -> int:
    """Run one call to the desktop's functions, printing its output as it comes."""
    copy_to_desktop(DESKTOP_FUNCTIONS, DESKTOP_FUNCTIONS_COPY)
    # Quiet before anything loads, or PowerShell reports loading its modules.
    script = (
        "$ProgressPreference = 'SilentlyContinue'\n"
        f". (Join-Path $HOME '{DESKTOP_FUNCTIONS_COPY}')\n{call}"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    command = (
        "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass"
        f" -EncodedCommand {encoded}"
    )
    ssh = subprocess.Popen(["ssh", *SSH_OPTIONS, HOST, command], stdout=subprocess.PIPE)
    assert ssh.stdout is not None
    try:
        for line in ssh.stdout:
            print(line.decode("utf-8", errors="replace"), end="", flush=True)
    except KeyboardInterrupt:
        ssh.terminate()
        if watching:
            print(
                "\nStopped watching; the run goes on. Follow it again with"
                " 'watch', or stop it with 'stop'."
            )
        return INTERRUPTED
    exit_code = ssh.wait()
    if exit_code == SSH_UNREACHABLE:
        # The copy just reached the desktop, so the connection broke on the way.
        print("\nThe connection to the desktop was lost.", file=sys.stderr)
        if watching:
            print("The run goes on; follow it again with 'watch'.", file=sys.stderr)
    return exit_code


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


def run(arguments: argparse.Namespace) -> int:
    configuration_path: Path = arguments.configuration
    try:
        read_configuration(configuration_path)
    except SettingsError as error:
        sys.exit(f"Configuration error: {error}")
    require_pushed_code()
    name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{configuration_path.stem}"
    copy_to_desktop(configuration_path, CONFIGURATION_COPY)
    alongside = "$true" if arguments.alongside else "$false"
    exit_code = on_desktop(
        f"Start-Run '{name}' (Join-Path $HOME '{CONFIGURATION_COPY}') {alongside}"
    )
    if exit_code or arguments.detach:
        return exit_code
    print("Its output follows; Ctrl+C stops watching, not the run.\n")
    return on_desktop(f"Watch-Run '{name}' $false", watching=True)


def watch(arguments: argparse.Namespace) -> int:
    pattern = (arguments.name or "").replace("'", "")
    return on_desktop(f"Watch-Latest '{pattern}' {OUTPUT_LINES_SHOWN}", watching=True)


def status(_: argparse.Namespace) -> int:
    return on_desktop(f"Show-Status {RECENT_RUNS_SHOWN} {OUTPUT_LINES_SHOWN}")


def stop(_: argparse.Namespace) -> int:
    return on_desktop("Stop-Runs")


def tests(_: argparse.Namespace) -> int:
    require_pushed_code()
    return on_desktop("Invoke-Tests")


def update(_: argparse.Namespace) -> int:
    require_pushed_code()
    return on_desktop("Update-Code")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    run_parser = commands.add_parser("run", help="pull, start a run, and watch it")
    run_parser.add_argument("configuration", type=Path, help="a TOML file")
    run_parser.add_argument(
        "--detach", action="store_true", help="start the run without watching it"
    )
    run_parser.add_argument(
        "--alongside",
        action="store_true",
        help="start even if another run is running; they share the GPU",
    )
    run_parser.set_defaults(action=run)
    watch_parser = commands.add_parser("watch", help="follow a run's output")
    watch_parser.add_argument(
        "name", nargs="?", help="part of the run's name; by default the latest run"
    )
    watch_parser.set_defaults(action=watch)
    for name, action, description in (
        ("status", status, "running and recent runs"),
        ("stop", stop, "stop every running run"),
        ("tests", tests, "pull, then run the tests"),
        ("update", update, "git pull on the desktop"),
    ):
        commands.add_parser(name, help=description).set_defaults(action=action)
    arguments = parser.parse_args()
    sys.exit(arguments.action(arguments))


if __name__ == "__main__":
    main()
