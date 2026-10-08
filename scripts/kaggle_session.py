"""One training run and its evaluation on a Kaggle T4, started by kaggle_gpu.py.

``kaggle_gpu.py run`` pushes this file to Kaggle as a script, with ``LAUNCH``
filled in, and Kaggle runs it in the background. The session clones
Centipede at the laptop's commit, and RL_lib and MujocoReplay as they are on
GitHub, installs them, runs the tests, places the run that training starts
from, trains, and evaluates the new run.

Kaggle keeps whatever is in ``/kaggle/working`` as the session's output, and
that folder mirrors the laptop's ``runs/``: the run folder, whose
``launched/`` holds the configuration file and console output of the
training and of its evaluation, as on the desktop, and ``kaggle/<launch>/``
with the console output of every step, so that a failure before the run
folder exists is kept too. The code goes to ``/tmp``, outside the output.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Filled in by kaggle_gpu.py when it pushes this file.
LAUNCH: dict = {}

GITHUB = "https://github.com/DrippingSoup22"
CODE = Path("/tmp/centipede")
SIBLINGS = Path("/tmp")
OUTPUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RUN_FOLDER = re.compile(r"Run folder: (\S+)")
SOURCE_LINE = re.compile(r"^source\s*=.*$", re.MULTILINE)


def shell(command: str) -> None:
    """Run a setup command in the code's folder; a failure ends the session."""
    print("$", command, flush=True)
    subprocess.run(command.split(), cwd=CODE if CODE.exists() else None, check=True)


def run_logged(command: list[str], console_path: Path) -> int:
    """Run a command, keeping its whole output in ``console_path``; its exit code.

    The session's own output, which ``kaggle_gpu.py watch`` follows, gets each
    line once it is finished: the progress line redraws itself with carriage
    returns, and only its last drawing is passed on.
    """
    environment = {**os.environ, "PYTHONUTF8": "1"}
    with console_path.open("wb") as console:
        process = subprocess.Popen(
            command,
            cwd=CODE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        assert process.stdout is not None
        unfinished = b""
        while chunk := os.read(process.stdout.fileno(), 4096):
            console.write(chunk)
            console.flush()
            *lines, unfinished = (unfinished + chunk).split(b"\n")
            for line in lines:
                drawn = line.rstrip(b"\r").rsplit(b"\r", 1)[-1]
                print(drawn.decode(errors="replace"), flush=True)
        if unfinished:
            print(unfinished.rsplit(b"\r", 1)[-1].decode(errors="replace"), flush=True)
        return process.wait()


def launch(name: str, text: str, console: Path) -> tuple[int, Path | None]:
    """Run one configuration file with ``python -m centipede``.

    Returns its exit code and the run folder it named, if any.
    """
    file = console / f"{name}.toml"
    file.write_text(text, encoding="utf-8")
    output = console / f"{name}.txt"
    exit_code = run_logged([sys.executable, "-u", "-m", "centipede", str(file)], output)
    found = RUN_FOLDER.search(output.read_text(encoding="utf-8", errors="replace"))
    return exit_code, Path(found.group(1)) if found else None


def keep_launched(name: str, console: Path, folder: Path) -> None:
    """Copy a launch's configuration file and console output into its run folder."""
    launched = folder / "launched"
    launched.mkdir(exist_ok=True)
    for suffix in (".toml", ".txt"):
        shutil.copy(console / f"{name}{suffix}", launched)


def fetch_code() -> None:
    shell(f"git clone --quiet {GITHUB}/centipede-project.git {CODE}")
    shell(f"git checkout --quiet {LAUNCH['commit']}")
    for sibling in ("RL_lib", "MujocoReplay"):
        shell(
            f"git clone --quiet --depth 1 {GITHUB}/{sibling}.git {SIBLINGS / sibling}"
        )
    for folder in (CODE, SIBLINGS / "RL_lib", SIBLINGS / "MujocoReplay"):
        shell(f"git -C {folder} log -1 --oneline")


def install() -> None:
    """Install the siblings, then the package with the GPU backend and the test tools.

    Kaggle provides PyTorch with CUDA, so it is not downloaded again.
    """
    shell(f"{sys.executable} -m pip install --quiet -e {SIBLINGS / 'RL_lib'}")
    shell(f"{sys.executable} -m pip install --quiet -e {SIBLINGS / 'MujocoReplay'}")
    shell(f"{sys.executable} -m pip install --quiet -e .[gpu,dev]")


def check_gpu() -> None:
    """Stop unless the GPU can compile MuJoCo Warp's Newton solver (Volta or newer)."""
    answer = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(answer, flush=True)
    if float(answer.split(",")[1]) < 7.0:
        sys.exit("This GPU is older than Volta; the session needs a T4.")


def place_start_runs() -> None:
    """Rebuild each run that training starts from out of its dataset.

    A start run needs only its saved configuration and the checkpoint used;
    it goes where the configuration file's ``start_from`` points, relative to
    the code's folder, where the command runs.
    """
    for start in LAUNCH["start_runs"]:
        found = [
            path
            for path in INPUT.rglob(start["checkpoint"])
            if start["dataset"] in path.parts
        ]
        if len(found) != 1:
            sys.exit(
                f"Dataset {start['dataset']}: {start['checkpoint']} found"
                f" {len(found)} times under {INPUT}."
            )
        checkpoints = CODE / start["folder"] / "checkpoints"
        checkpoints.mkdir(parents=True)
        shutil.copy(found[0], checkpoints)
        shutil.copy(found[0].parent / "configuration.toml", checkpoints.parent)
        print(f"Start run {start['folder']}, from {start['dataset']}", flush=True)


def main() -> None:
    name = LAUNCH["name"]
    console = OUTPUT / "kaggle" / name
    console.mkdir(parents=True)
    check_gpu()
    fetch_code()
    install()
    if LAUNCH["tests"]:
        print("\n=== tests ===\n", flush=True)
        failed = run_logged(
            [sys.executable, "-m", "pytest", "-q", "--durations=5"],
            console / "tests.txt",
        )
        if failed:
            print(
                f"\nSession finished: the tests failed, nothing was trained ({name})."
            )
            return
    place_start_runs()

    print(f"\n=== training {name} ===\n", flush=True)
    exit_code, folder = launch(name, LAUNCH["training"], console)
    trained = "finished" if exit_code == 0 else f"failed ({exit_code})"
    evaluated = "none"
    if folder is not None:
        keep_launched(name, console, folder)
    if exit_code == 0 and folder is not None and LAUNCH["evaluation"]:
        print(f"\n=== evaluating {folder.name} ===\n", flush=True)
        evaluation = f"{name}_evaluation"
        text = SOURCE_LINE.sub(f'source = "{folder.as_posix()}"', LAUNCH["evaluation"])
        evaluation_code, _ = launch(evaluation, text, console)
        keep_launched(evaluation, console, folder)
        evaluated = f"failed ({evaluation_code})" if evaluation_code else "finished"
    print(
        f"\nSession finished: training {trained}, evaluation {evaluated},"
        f" run folder {folder.name if folder else '-'} ({name}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
