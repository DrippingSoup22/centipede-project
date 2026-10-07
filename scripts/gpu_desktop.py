"""Run the project on the GPU desktop from the laptop, over SSH.

The desktop holds its own copy of this repository, RL_lib, and MujocoReplay,
and its own Python environment (README, Setup). This script, run on the
laptop from the repository root with the project's Python, sends it commands:

    python scripts/gpu_desktop.py run configs/baseline.toml   # start, watch, fetch
    python scripts/gpu_desktop.py watch          # follow the latest run
    python scripts/gpu_desktop.py fetch          # copy the latest run's folder here
    python scripts/gpu_desktop.py status         # running and recent runs
    python scripts/gpu_desktop.py stop           # stop every running run
    python scripts/gpu_desktop.py tests          # pull, then run the tests there
    python scripts/gpu_desktop.py update         # git pull on the desktop

``run`` pulls the pushed code, sends the laptop's configuration file as it is,
so a changed setting needs no commit, and starts the run. It then shows the
run's output as it is written, one line per window with a bar that fills as
the window is collected, and when the run ends copies its run folder into
the laptop's ``runs/``, with the configuration file sent and the console
output in its ``launched/`` folder, named after the launch. Ctrl+C
stops watching, not the run; ``watch`` picks it up again and also copies the
folder at the end, and ``fetch`` copies it at any time. ``watch`` and
``fetch`` take part of a run's name to choose an earlier run. Each run's file
and console output are kept in the desktop's ``runs/launched/``; the output
ends with the run's exit code.

What runs on the desktop is ``gpu_desktop.ps1``: every command first copies
it, and ``run`` its configuration file, into the desktop's home folder with
scp, then sends one short call to its functions, encoded so that nothing
needs quoting twice. The whole file does not fit in one command: Windows cuts
commands that long. ``CENTIPEDE_GPU_HOST`` names the desktop; by default
``gpu``, an alias in ``~/.ssh/config``.
"""

import argparse
import base64
import codecs
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from centipede.experiment.configuration import read_configuration
from centipede.settings_section import SettingsError

HOST = os.environ.get("CENTIPEDE_GPU_HOST", "gpu")
REPOSITORY = Path(__file__).resolve().parents[1]
LAPTOP_RUNS = REPOSITORY / "runs"
DESKTOP_FUNCTIONS = Path(__file__).with_suffix(".ps1")
# Paths on the desktop, relative to its user's home folder, where scp starts.
DESKTOP_RUNS = "Projects/Centipede/runs"
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


def scp(*paths: str) -> int:
    return subprocess.run(["scp", "-q", *SSH_OPTIONS, *paths]).returncode


def ssh_command(call: str) -> list[str]:
    """Copy the desktop's functions over, and the command that makes one call."""
    if scp(str(DESKTOP_FUNCTIONS), f"{HOST}:{DESKTOP_FUNCTIONS_COPY}"):
        unreachable()
    # Quiet before anything loads, or PowerShell reports loading its modules.
    script = (
        "$ProgressPreference = 'SilentlyContinue'\n"
        f". (Join-Path $HOME '{DESKTOP_FUNCTIONS_COPY}')\n{call}"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [
        "ssh",
        *SSH_OPTIONS,
        HOST,
        "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass"
        f" -EncodedCommand {encoded}",
    ]


def on_desktop(call: str, watching: bool = False) -> int:
    """Make one call, passing its output through as it comes; its exit code.

    The output goes through unchanged, carriage returns included, so the
    terminal redraws each window's progress bar in place.
    """
    ssh = subprocess.Popen(ssh_command(call), stdout=subprocess.PIPE)
    assert ssh.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := os.read(ssh.stdout.fileno(), 4096):
            sys.stdout.write(decoder.decode(chunk))
            sys.stdout.flush()
    except KeyboardInterrupt:
        ssh.terminate()
        if watching:
            print(
                "\n\nStopped watching; the run goes on. Follow it again with"
                " 'watch', or stop it with 'stop'."
            )
        return INTERRUPTED
    exit_code = ssh.wait()
    if exit_code == SSH_UNREACHABLE:
        # The copy just reached the desktop, so the connection broke on the way.
        print("\n\nThe connection to the desktop was lost.", file=sys.stderr)
        if watching:
            print("The run goes on; follow it again with 'watch'.", file=sys.stderr)
    return exit_code


def desktop_answer(call: str) -> tuple[int, str]:
    """Make one call; its exit code and output."""
    answer = subprocess.run(ssh_command(call), capture_output=True)
    if answer.returncode == SSH_UNREACHABLE:
        unreachable()
    return answer.returncode, answer.stdout.decode("utf-8", errors="replace").strip()


def find_run(pattern: str | None) -> str:
    """The name of the latest launched run whose name contains ``pattern``."""
    _, name = desktop_answer(f"Find-Run '{(pattern or '').replace(chr(39), '')}'")
    if not name:
        sys.exit("No launched run matches." if pattern else "No run was launched yet.")
    return name


def fetch_run(name: str) -> int:
    """Copy a launched run's folder and console output into the laptop's runs."""
    _, folder = desktop_answer(f"Get-RunFolder '{name}'")
    if not folder:
        print(f"{name} wrote no run folder; its output is in runs/launched there.")
        return 1
    LAPTOP_RUNS.mkdir(exist_ok=True)
    copied = scp("-r", f"{HOST}:{DESKTOP_RUNS}/{folder}", str(LAPTOP_RUNS))
    # Every launch into the folder, training or evaluation, keeps its own
    # configuration file and console output, named after the launch.
    launched = LAPTOP_RUNS / folder / "launched"
    launched.mkdir(exist_ok=True)
    for suffix in (".toml", ".txt"):
        source = f"{HOST}:{DESKTOP_RUNS}/launched/{name}{suffix}"
        copied = copied or scp(source, str(launched / f"{name}{suffix}"))
    if copied:
        print(f"Copying runs/{folder} to the laptop failed; try 'fetch' again.")
        return 1
    print(f"Copied to the laptop: runs/{folder}")
    if (LAPTOP_RUNS / folder / "report.html").exists():
        print(f"  report: runs/{folder}/report.html")
    return 0


def watch_and_fetch(name: str) -> int:
    """Follow a run's output to its end, then copy its folder here."""
    exit_code = on_desktop(f"Watch-Run '{name}'", watching=True)
    if exit_code in (INTERRUPTED, SSH_UNREACHABLE):
        return exit_code
    if exit_code:
        print(f"\nThe run failed (exit code {exit_code}); its last lines are above.")
    print()
    fetch_run(name)
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
    if scp(str(configuration_path), f"{HOST}:{CONFIGURATION_COPY}"):
        unreachable()
    alongside = "$true" if arguments.alongside else "$false"
    exit_code, answer = desktop_answer(
        f"Start-Run '{name}' (Join-Path $HOME '{CONFIGURATION_COPY}') {alongside}"
    )
    if exit_code:
        print(answer)
        return exit_code
    code = answer.removeprefix("code ")
    print(f"Started {name} on the GPU desktop")
    print(f"  configuration  {configuration_path.as_posix()}")
    print(f"  code           {code}")
    if arguments.detach:
        print("Follow it with 'watch'; its results are copied here at the end.")
        return 0
    print("Ctrl+C stops watching; the run goes on.\n")
    return watch_and_fetch(name)


def watch(arguments: argparse.Namespace) -> int:
    name = find_run(arguments.name)
    print(f"Watching {name}; Ctrl+C stops watching, not the run.\n")
    return watch_and_fetch(name)


def fetch(arguments: argparse.Namespace) -> int:
    return fetch_run(find_run(arguments.name))


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
    run_parser = commands.add_parser(
        "run", help="pull, start a run, watch it, and copy its results here"
    )
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
    for name, action, description in (
        ("watch", watch, "follow a run's output, then copy its results here"),
        ("fetch", fetch, "copy a run's results here"),
    ):
        chosen = commands.add_parser(name, help=description)
        chosen.add_argument(
            "name", nargs="?", help="part of the run's name; by default the latest run"
        )
        chosen.set_defaults(action=action)
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
