"""Run the project on the GPU desktop from the laptop, over SSH.

The desktop holds its own copy of this repository, RL_lib, and MujocoReplay,
and its own Python environment (README, Setup). This script, run on the
laptop from the repository root with the project's Python, sends it commands:

    python scripts/gpu_desktop.py run configs/baseline.toml   # start, watch, copy
    python scripts/gpu_desktop.py queue configs/overnight     # a folder, one by one
    python scripts/gpu_desktop.py watch          # follow the latest launch
    python scripts/gpu_desktop.py fetch          # copy the latest launch here now
    python scripts/gpu_desktop.py status         # running and recent runs
    python scripts/gpu_desktop.py stop           # stop every running run
    python scripts/gpu_desktop.py tests          # pull, then run the tests there
    python scripts/gpu_desktop.py update         # git pull on the desktop

``run`` pulls the pushed code, sends the laptop's configuration file as it
is, so a changed setting needs no commit, and starts the run. ``queue`` does
the same for a folder of training files, run one after another by
``run_queue.py`` on the desktop, each evaluated after it with the folder's
``evaluation.toml`` if there is one.

Every training that ``run`` or ``queue`` launches is copied into the laptop's
``runs/`` as soon as it ends, with its evaluation when it has one. A copier
does it: a process of its own on the laptop, started with the launch, that
goes on when the terminal stops watching or is closed, and tries again every
minute while the desktop cannot be reached (the laptop asleep, the network
down). It writes what it copied to ``runs/copies/<launch>.txt``. A copier
stopped before the end, for example by restarting the laptop, is started
again by the next command. Each run folder holds, in ``launched/``, the
configuration file and console output of every launch into it.

``run`` and ``queue`` then show the output as it is written, one line per
window with a bar that fills as the window is collected, with the copier's
lines among it, and wait at the end for the last copy. Ctrl+C stops watching,
not the run or its copying. ``watch`` follows the latest launch again, and
``fetch`` copies one at any time; both take part of a launch's name to choose
an earlier one, and where a queue matches, they choose the queue rather than
its runs. Each launch's file and console output are kept in the desktop's
``runs/launched/``; the output ends with the exit code.

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
import re
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, NoReturn

from centipede.experiment.configuration import read_configuration
from centipede.settings_section import SettingsError

if os.name == "nt":
    import msvcrt
else:
    import fcntl

HOST = os.environ.get("CENTIPEDE_GPU_HOST", "gpu")
REPOSITORY = Path(__file__).resolve().parents[1]
LAPTOP_RUNS = REPOSITORY / "runs"
COPIES = LAPTOP_RUNS / "copies"
DESKTOP_FUNCTIONS = Path(__file__).with_suffix(".ps1")
# Paths on the desktop, relative to its user's home folder, where scp starts.
DESKTOP_RUNS = "Projects/Centipede/runs"
DESKTOP_FUNCTIONS_COPY = ".centipede_gpu_desktop.ps1"
CONFIGURATION_COPY = ".centipede_configuration.toml"
QUEUE_COPY = ".centipede_queue_"
# Part of every queue's name; the queue's runs are named after it.
QUEUE_MARK = "_queue_"
# The line a queue prints as each of its runs starts (scripts/run_queue.py):
# the run before it, evaluation included, is then done.
QUEUE_RUN_START = re.compile(r"=== \d+/\d+  (\S+)  \(started")
EVALUATION_FILE = "evaluation.toml"
# Code that must be pushed before the desktop can run it; configuration files
# are sent as they are.
CODE_PATHS = ("src", "models", "tests", "benchmarks", "pyproject.toml")
RECENT_RUNS_SHOWN = 5
OUTPUT_LINES_SHOWN = 6
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    # A connection that dies, as when the laptop sleeps, ends ssh within a
    # minute instead of leaving it waiting for ever.
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
)
# No ssh or scp gets the console's keyboard input, which none needs: an ssh
# started while another one in the same console holds it never ends, as
# happened when a watched queue's runs were copied.
NO_INPUT = subprocess.DEVNULL
SSH_UNREACHABLE = 255
INTERRUPTED = 130
# A copier waits this long before trying the desktop again, and tries the
# last copies of a launch that has ended this many times.
RETRY_SECONDS = 60
FINAL_ATTEMPTS = 3
# A copier's log opens with what it copies, records each run copied on a
# line of its own, and ends with a line holding DONE.
COPYING_LINE = re.compile(r"^\d\d:\d\d copying (queue|run) ", re.MULTILINE)
COPIED_LINE = re.compile(r"^\d\d:\d\d copied (\S+)$", re.MULTILINE)
DONE = " done: "


class DesktopUnreachable(Exception):
    """The desktop did not answer."""


def unreachable() -> NoReturn:
    raise DesktopUnreachable(
        f"Could not reach the desktop ({HOST}): is it on and awake, and is its"
        " SSH server running?"
    )


def scp(*paths: str) -> int:
    return subprocess.run(
        ["scp", "-q", *SSH_OPTIONS, *paths], stdin=NO_INPUT
    ).returncode


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


def on_desktop(
    call: str,
    watching: bool = False,
    follow: Callable[[str], None] | None = None,
    show: bool = True,
) -> int:
    """Make one call, passing its output through as it comes; its exit code.

    The output goes through unchanged, carriage returns included, so the
    terminal redraws each window's progress bar in place. ``follow``, if
    given, receives the output too, after it is shown; ``show`` False keeps
    it off the terminal.
    """
    ssh = subprocess.Popen(ssh_command(call), stdin=NO_INPUT, stdout=subprocess.PIPE)
    assert ssh.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := os.read(ssh.stdout.fileno(), 4096):
            text = decoder.decode(chunk)
            if show:
                sys.stdout.write(text)
                sys.stdout.flush()
            if follow:
                follow(text)
    except KeyboardInterrupt:
        ssh.terminate()
        if watching:
            print(
                "\n\nStopped watching; the run and its copying go on. Follow it"
                " again with 'watch', or stop it with 'stop'."
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
    answer = subprocess.run(ssh_command(call), stdin=NO_INPUT, capture_output=True)
    if answer.returncode == SSH_UNREACHABLE:
        unreachable()
    return answer.returncode, answer.stdout.decode("utf-8", errors="replace").strip()


def find_run(pattern: str | None) -> tuple[str, bool]:
    """The latest launched run whose name contains ``pattern``; whether a queue.

    Where a queue matches, the runs named after it are left out.
    """
    _, answer = desktop_answer(f"Find-Run '{(pattern or '').replace(chr(39), '')}'")
    if not answer:
        sys.exit("No launched run matches." if pattern else "No run was launched yet.")
    kind, name = answer.split(" ", 1)
    return name, kind == "queue"


def copy_launched(name: str, folder: str) -> int:
    """Copy a launch's configuration file and console output into its run folder.

    Every launch into the folder, training or evaluation, keeps its own,
    named after the launch.
    """
    launched = LAPTOP_RUNS / folder / "launched"
    launched.mkdir(parents=True, exist_ok=True)
    failed = 0
    for suffix in (".toml", ".txt"):
        source = f"{HOST}:{DESKTOP_RUNS}/launched/{name}{suffix}"
        failed = failed or scp(source, str(launched / f"{name}{suffix}"))
    return failed


def fetch_run(name: str) -> int:
    """Copy a launched run's folder and console output into the laptop's runs."""
    _, folder = desktop_answer(f"Get-RunFolder '{name}'")
    if not folder:
        print(f"{name} wrote no run folder; its output is in runs/launched there.")
        return 1
    LAPTOP_RUNS.mkdir(exist_ok=True)
    copied = scp("-r", f"{HOST}:{DESKTOP_RUNS}/{folder}", str(LAPTOP_RUNS))
    copied = copied or copy_launched(name, folder)
    if copied:
        print(f"Copying runs/{folder} to the laptop failed; try 'fetch' again.")
        return 1
    print(f"Copied to the laptop: runs/{folder}")
    if (LAPTOP_RUNS / folder / "report.html").exists():
        print(f"  report: runs/{folder}/report.html")
    for report in sorted((LAPTOP_RUNS / folder / "evaluations").glob("*.html")):
        print(f"  evaluation: runs/{folder}/evaluations/{report.name}")
    return 0


def fetch_queue_run(name: str) -> int:
    """Copy one run of a queue here, with its evaluation's file and output."""
    failed = fetch_run(name)
    # The evaluation's results came with its run's folder.
    evaluation = f"{name}_evaluation"
    _, folder = desktop_answer(f"Get-RunFolder '{evaluation}'")
    return int(bool(failed or (folder and copy_launched(evaluation, folder))))


def queue_runs(name: str) -> list[str]:
    """The runs a queue has started so far, in order, without their evaluations."""
    _, answer = desktop_answer(f"Get-QueueItems '{name}'")
    return [item for item in answer.split() if not item.endswith("_evaluation")]


def fetch_queue_output(name: str) -> int:
    """Copy a queue's own output, which ends with how each run ended."""
    queues = LAPTOP_RUNS / "queues"
    queues.mkdir(parents=True, exist_ok=True)
    source = f"{HOST}:{DESKTOP_RUNS}/launched/{name}.txt"
    failed = scp(source, str(queues / f"{name}.txt"))
    if not failed:
        print(f"Queue output: runs/queues/{name}.txt")
    return failed


def fetch_queue(name: str) -> int:
    """Copy every run of a queue, with its evaluation, into the laptop's runs."""
    failed = 0
    for run_name in queue_runs(name):
        failed = fetch_queue_run(run_name) or failed
    return int(bool(fetch_queue_output(name) or failed))


def now() -> str:
    return f"{datetime.now():%H:%M}"


def copier_log(name: str) -> Path:
    return COPIES / f"{name}.txt"


def take_copy_lock(name: str) -> IO[str] | None:
    """The lock a launch's copier holds while it runs, or None if it is held.

    The system lets it go when the process ends, however it ends.
    """
    COPIES.mkdir(parents=True, exist_ok=True)
    lock = (COPIES / f"{name}.lock").open("a")
    lock.seek(0)
    try:
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


def copier_running(name: str) -> bool:
    lock = take_copy_lock(name)
    if lock is None:
        return True
    lock.close()
    return False


def copier_done(name: str) -> bool:
    log = copier_log(name)
    return log.exists() and DONE in log.read_text(encoding="utf-8", errors="replace")


def start_copier(name: str, is_queue: bool) -> None:
    """Start a launch's copier as a process of its own, unless it runs or is done."""
    if copier_running(name) or copier_done(name):
        return
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "copy",
        name,
        "--kind",
        "queue" if is_queue else "run",
    ]
    with copier_log(name).open("a", encoding="utf-8") as log:
        options = {
            "cwd": REPOSITORY,
            "stdin": NO_INPUT,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "env": {**os.environ, "PYTHONUTF8": "1"},
        }
        if os.name != "nt":
            subprocess.Popen(command, start_new_session=True, **options)
            return
        # Its own hidden console and process group, so that neither Ctrl+C
        # nor closing the terminal reaches it; and outside the terminal's job
        # where the terminal allows it.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(
                command,
                creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                **options,
            )
        except OSError:
            subprocess.Popen(command, creationflags=flags, **options)


def resume_copiers() -> None:
    """Start again every copier that stopped before its launch was all copied."""
    for log in COPIES.glob("*.txt"):
        text = log.read_text(encoding="utf-8", errors="replace")
        kind = COPYING_LINE.search(text)
        if kind and DONE not in text:
            start_copier(log.stem, kind.group(1) == "queue")


def copy_and_record(name: str, copied: set[str], queue_run: bool) -> bool:
    """Copy one run here; record it in ``copied`` and the log if that worked."""
    print(f"\n{now()} {name} is done; copying it here")
    if fetch_queue_run(name) if queue_run else fetch_run(name):
        return False
    copied.add(name)
    print(f"{now()} copied {name}")
    return True


class QueueCopier:
    """Copies each run of a queue here once it and its evaluation are done.

    It receives the queue's output from its start. A run is done when the
    next one starts; the last one is left for the end of the queue. Runs in
    ``copied`` are not copied again, and each run copied is added to it.
    """

    def __init__(self, queue_name: str, copied: set[str]) -> None:
        self.queue_name = queue_name
        self.copied = copied
        self.running: str | None = None
        self.unfinished_line = ""

    def __call__(self, text: str) -> None:
        *lines, self.unfinished_line = (self.unfinished_line + text).split("\n")
        for line in lines:
            started = QUEUE_RUN_START.match(line)
            if not started:
                continue
            if self.running and self.running not in self.copied:
                copy_and_record(self.running, self.copied, queue_run=True)
            self.running = f"{self.queue_name}_{started.group(1)}"


def copy_rest(name: str, is_queue: bool, copied: set[str]) -> list[str]:
    """Copy what a launch that has ended left uncopied; what still is."""
    runs = queue_runs(name) if is_queue else [name]
    left = [
        run_name
        for run_name in runs
        if run_name not in copied
        and not copy_and_record(run_name, copied, queue_run=is_queue)
    ]
    if is_queue and fetch_queue_output(name):
        left.append("the queue's output")
    return left


def copy_launch(name: str, is_queue: bool) -> int:
    """Follow a launch, copying each of its runs here as it ends, until all are.

    This is the copier; its output is its log. It holds the launch's lock
    while it runs, so that only one copies each launch.
    """
    lock = take_copy_lock(name)
    if lock is None:
        print(f"{now()} {name} is already being copied.")
        return 0
    log = copier_log(name)
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    copied = set(COPIED_LINE.findall(text))
    kind = "queue" if is_queue else "run"
    print(f"{now()} copying {kind} {name} to the laptop as it ends")
    while True:
        try:
            follower = QueueCopier(name, copied) if is_queue else None
            exit_code = on_desktop(f"Watch-Run '{name}'", follow=follower, show=False)
            if exit_code != SSH_UNREACHABLE:
                break
        except DesktopUnreachable:
            pass
        print(f"{now()} the desktop cannot be reached; trying again in a minute")
        time.sleep(RETRY_SECONDS)
    for attempt in range(FINAL_ATTEMPTS):
        try:
            left = copy_rest(name, is_queue, copied)
        except DesktopUnreachable:
            left = ["the rest (the desktop cannot be reached)"]
        if not left:
            break
        if attempt + 1 < FINAL_ATTEMPTS:
            time.sleep(RETRY_SECONDS)
    if left:
        print(f"{now()}{DONE}not copied: {', '.join(left)}; try 'fetch'")
    else:
        print(f"{now()}{DONE}everything is on the laptop")
    lock.close()
    return int(bool(left))


class CopierLog:
    """Shows the new lines of a launch's copier log among the watched output."""

    def __init__(self, name: str, from_start: bool) -> None:
        self.path = copier_log(name)
        exists = self.path.exists()
        self.position = self.path.stat().st_size if exists and not from_start else 0
        self.at_line_start = True

    def __call__(self, text: str = "") -> None:
        if text:
            self.at_line_start = text.endswith("\n")
        if not self.path.exists():
            return
        with self.path.open("rb") as log:
            log.seek(self.position)
            lines, newline, _ = log.read().rpartition(b"\n")
        if not newline:
            return
        self.position += len(lines) + 1
        if not self.at_line_start:
            print()
        # The copier writes its log with Windows line ends.
        shown = lines.replace(b"\r\n", b"\n").removesuffix(b"\r")
        print(shown.decode("utf-8", errors="replace"), flush=True)
        self.at_line_start = True


def wait_for_copier(name: str, is_queue: bool, log: CopierLog) -> None:
    """Show the copier's lines until it is done, starting it again if it stopped."""
    try:
        while not copier_done(name):
            log()
            if not copier_running(name):
                start_copier(name, is_queue)
                time.sleep(5)
            time.sleep(0.5)
        log()
    except KeyboardInterrupt:
        print("\nStopped waiting; the copying goes on.")


def watch_launch(name: str, is_queue: bool, log: CopierLog) -> int:
    """Follow a launch's output to its end with its copier's lines, then wait.

    The wait is for the copier's last copies.
    """
    exit_code = on_desktop(f"Watch-Run '{name}'", watching=True, follow=log)
    if exit_code in (INTERRUPTED, SSH_UNREACHABLE):
        print(f"The copier's log: runs/copies/{name}.txt")
        return exit_code
    if exit_code:
        print(f"\nThe run failed (exit code {exit_code}); its last lines are above.")
    print()
    wait_for_copier(name, is_queue, log)
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
    log = CopierLog(name, from_start=True)
    start_copier(name, is_queue=False)
    code = answer.removeprefix("code ")
    print(f"Started {name} on the GPU desktop")
    print(f"  configuration  {configuration_path.as_posix()}")
    print(f"  code           {code}")
    print(f"  copied here    as soon as it ends (log: runs/copies/{name}.txt)")
    if arguments.detach:
        print("Follow it with 'watch'.")
        return 0
    print("Ctrl+C stops watching; the run and its copying go on.\n")
    return watch_launch(name, False, log)


def queue(arguments: argparse.Namespace) -> int:
    folder: Path = arguments.folder
    files = sorted(
        path for path in folder.glob("*.toml") if path.name != EVALUATION_FILE
    )
    if not files:
        sys.exit(f"No training files in {folder}.")
    for path in files:
        try:
            configuration = read_configuration(path)
        except SettingsError as error:
            sys.exit(f"Configuration error in {path.name}: {error}")
        if configuration.evaluation is not None:
            sys.exit(f"{path.name} is an evaluation file; name it {EVALUATION_FILE}.")
    evaluation = folder / EVALUATION_FILE
    if evaluation.exists():
        # Its source is replaced by each run's folder on the desktop.
        run_values = tomllib.loads(evaluation.read_text(encoding="utf-8")).get(
            "run", {}
        )
        if run_values.get("mode") != "evaluate" or "source" not in run_values:
            sys.exit(f'{evaluation} needs mode = "evaluate" and a source in [run].')
    require_pushed_code()
    name = f"{datetime.now():%Y-%m-%d_%H%M%S}{QUEUE_MARK}{folder.name}"
    folder_copy = f"{QUEUE_COPY}{name}"
    if scp("-r", str(folder), f"{HOST}:{folder_copy}"):
        unreachable()
    alongside = "$true" if arguments.alongside else "$false"
    exit_code, answer = desktop_answer(
        f"Start-Queue '{name}' (Join-Path $HOME '{folder_copy}') {alongside}"
    )
    if exit_code:
        print(answer)
        return exit_code
    log = CopierLog(name, from_start=True)
    start_copier(name, is_queue=True)
    evaluated = f"each run, with {EVALUATION_FILE}" if evaluation.exists() else "none"
    print(f"Started {name} on the GPU desktop")
    print(f"  runs           {', '.join(path.stem for path in files)}")
    print(f"  evaluation     {evaluated}")
    print(f"  code           {answer.removeprefix('code ')}")
    print(
        "  copied here    each run as soon as it and its evaluation are done"
        f" (log: runs/copies/{name}.txt)"
    )
    if arguments.detach:
        print("Follow it with 'watch queue'.")
        return 0
    print("Ctrl+C stops watching; the queue and its copying go on.\n")
    return watch_launch(name, True, log)


def watch(arguments: argparse.Namespace) -> int:
    name, is_queue = find_run(arguments.name)
    log = CopierLog(name, from_start=False)
    start_copier(name, is_queue)
    print(f"Watching {name}; Ctrl+C stops watching, not the run or its copying.\n")
    return watch_launch(name, is_queue, log)


def fetch(arguments: argparse.Namespace) -> int:
    name, is_queue = find_run(arguments.name)
    return fetch_queue(name) if is_queue else fetch_run(name)


def copy(arguments: argparse.Namespace) -> int:
    return copy_launch(arguments.name, arguments.kind == "queue")


def status(_: argparse.Namespace) -> int:
    exit_code = on_desktop(f"Show-Status {RECENT_RUNS_SHOWN} {OUTPUT_LINES_SHOWN}")
    copying = [
        log.stem for log in sorted(COPIES.glob("*.txt")) if copier_running(log.stem)
    ]
    print(f"\ncopying to the laptop: {', '.join(copying) or 'nothing'}")
    return exit_code


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
    queue_parser = commands.add_parser(
        "queue", help="pull, then run a folder of training files one after another"
    )
    queue_parser.add_argument(
        "folder", type=Path, help="training files, and optionally evaluation.toml"
    )
    queue_parser.add_argument(
        "--detach", action="store_true", help="start the queue without watching it"
    )
    queue_parser.add_argument(
        "--alongside",
        action="store_true",
        help="start even if another run is running; they share the GPU",
    )
    queue_parser.set_defaults(action=queue)
    for name, action, description in (
        ("watch", watch, "follow a launch's output while it is copied here"),
        ("fetch", fetch, "copy a launch's results here now"),
    ):
        chosen = commands.add_parser(name, help=description)
        chosen.add_argument(
            "name", nargs="?", help="part of the launch's name; by default the latest"
        )
        chosen.set_defaults(action=action)
    copy_parser = commands.add_parser(
        "copy",
        help="the copier, which run, queue, and watch start: copy each run of a"
        " launch here as it ends",
    )
    copy_parser.add_argument("name", help="the launch's whole name")
    copy_parser.add_argument("--kind", choices=("queue", "run"), required=True)
    copy_parser.set_defaults(action=copy)
    for name, action, description in (
        ("status", status, "running and recent runs, and what is being copied"),
        ("stop", stop, "stop every running run"),
        ("tests", tests, "pull, then run the tests"),
        ("update", update, "git pull on the desktop"),
    ):
        commands.add_parser(name, help=description).set_defaults(action=action)
    arguments = parser.parse_args()
    try:
        if arguments.action is not copy:
            resume_copiers()
        sys.exit(arguments.action(arguments))
    except DesktopUnreachable as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
