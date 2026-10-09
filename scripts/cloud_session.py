"""Training runs and their evaluations on a cloud machine's GPUs.

A launcher on the laptop sends this file with ``LAUNCH`` filled in:
``kaggle_gpu.py`` pushes it to Kaggle as a script, which Kaggle runs in the
background, and ``runpod_gpu.py`` copies it to a rented Runpod pod, which
runs it with ``runpod_pod.sh``. The session clones Centipede at the laptop's
commit, and RL_lib and MujocoReplay as they are on GitHub, installs them,
runs the tests, and places the runs that training starts from. Then it
trains its runs, one per GPU at the same time (the same file with different
seeds, or two files), and evaluates each run when its training ends.

The launch names the session's output folder, which mirrors the laptop's
``runs/``: the run folders, whose ``launched/`` holds the configuration file
and console output of the training and of its evaluation, as on the desktop,
and ``<console>/<launch>/`` with the console output of every step, so that a
failure before a run folder exists is kept too. On Kaggle that is
``/kaggle/working``, which Kaggle keeps as the session's output. The code
goes to ``/tmp``, outside the output.
"""

import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Filled in by the launcher when it sends this file. Besides the runs, it
# names the folders: "output" for the results, "console" for the console
# output's subfolder in it, and "input" where the start runs' files are.
LAUNCH: dict = {}

GITHUB = "https://github.com/DrippingSoup22"
CODE = Path("/tmp/centipede")
SIBLINGS = Path("/tmp")
RUN_FOLDER = re.compile(r"Run folder: (\S+)")
SOURCE_LINE = re.compile(r"^source\s*=.*$", re.MULTILINE)


def shell(command: str) -> None:
    """Run a setup command in the code's folder; a failure ends the session."""
    print("$", command, flush=True)
    subprocess.run(command.split(), cwd=CODE if CODE.exists() else None, check=True)


def run_logged(
    command: list[str], console_path: Path, gpu: int = 0, label: str = ""
) -> int:
    """Run a command on one GPU, keeping its whole output in ``console_path``.

    Returns its exit code. The session's own output, which ``kaggle_gpu.py
    watch`` follows, gets each line once it is finished, after ``label``: the
    progress line redraws itself with carriage returns, and only its last
    drawing is passed on.
    """
    environment = {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": str(gpu)}
    prefix = f"[{label}] " if label else ""
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
                print(prefix + drawn.decode(errors="replace"), flush=True)
        if unfinished:
            drawn = unfinished.rsplit(b"\r", 1)[-1]
            print(prefix + drawn.decode(errors="replace"), flush=True)
        return process.wait()


def launch(
    name: str, text: str, console: Path, gpu: int, label: str
) -> tuple[int, Path | None]:
    """Run one configuration file with ``python -m centipede`` on one GPU.

    Returns its exit code and the run folder it named, if any.
    """
    file = console / f"{name}.toml"
    file.write_text(text, encoding="utf-8")
    output = console / f"{name}.txt"
    command = [sys.executable, "-u", "-m", "centipede", str(file)]
    exit_code = run_logged(command, output, gpu, label)
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

    Kaggle's Python and the Runpod pod's environment already have PyTorch with
    CUDA, so it is not downloaded again.
    """
    shell(f"{sys.executable} -m pip install --quiet -e {SIBLINGS / 'RL_lib'}")
    shell(f"{sys.executable} -m pip install --quiet -e {SIBLINGS / 'MujocoReplay'}")
    shell(f"{sys.executable} -m pip install --quiet -e .[gpu,dev]")


def check_gpus() -> None:
    """Stop unless there is a GPU per run, each able to compile MuJoCo Warp's
    Newton solver (Volta or newer)."""
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
    gpus = answer.strip().splitlines()
    if len(gpus) < len(LAUNCH["runs"]):
        sys.exit(f"{len(LAUNCH['runs'])} runs need as many GPUs; found {len(gpus)}.")
    if min(float(gpu.split(",")[1]) for gpu in gpus) < 7.0:
        sys.exit("A GPU is older than Volta, too old for MuJoCo Warp's Newton solver.")


def place_start_runs() -> None:
    """Rebuild each run that training starts from out of its uploaded files.

    The launcher uploads a start run's saved configuration and the checkpoint
    used into a folder of the input, named as ``dataset`` says: a Kaggle
    dataset, or a folder copied to the pod. The run goes where the
    configuration file's ``start_from`` points, relative to the code's
    folder, where the command runs.
    """
    uploads = Path(LAUNCH["input"])
    for start in LAUNCH["start_runs"]:
        found = [
            path
            for path in uploads.rglob(start["checkpoint"])
            if start["dataset"] in path.parts
        ]
        if len(found) != 1:
            sys.exit(
                f"Dataset {start['dataset']}: {start['checkpoint']} found"
                f" {len(found)} times under {uploads}."
            )
        checkpoints = CODE / start["folder"] / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        shutil.copy(found[0], checkpoints)
        shutil.copy(found[0].parent / "configuration.toml", checkpoints.parent)
        print(f"Start run {start['folder']}, from {start['dataset']}", flush=True)


def train_and_evaluate(run: dict, gpu: int, console: Path) -> str:
    """Train one run on one GPU, then evaluate it; how both ended."""
    name, label = run["name"], run["label"]
    exit_code, folder = launch(name, run["training"], console, gpu, label)
    trained = "finished" if exit_code == 0 else f"failed ({exit_code})"
    evaluated = "none"
    if folder is not None:
        keep_launched(name, console, folder)
    if exit_code == 0 and folder is not None and LAUNCH["evaluation"]:
        evaluation = f"{name}_evaluation"
        text = SOURCE_LINE.sub(f'source = "{folder.as_posix()}"', LAUNCH["evaluation"])
        evaluation_code, _ = launch(evaluation, text, console, gpu, label)
        keep_launched(evaluation, console, folder)
        evaluated = f"failed ({evaluation_code})" if evaluation_code else "finished"
    return (
        f"{label or name}: training {trained}, evaluation {evaluated},"
        f" run folder {folder.name if folder else '-'}"
    )


def main() -> None:
    name = LAUNCH["name"]
    console = Path(LAUNCH["output"]) / LAUNCH["console"] / name
    console.mkdir(parents=True, exist_ok=True)
    check_gpus()
    fetch_code()
    install()
    if LAUNCH["tests"]:
        print("\n=== tests ===\n", flush=True)
        failed = run_logged(
            [sys.executable, "-m", "pytest", "-q", "--durations=5"],
            console / "tests.txt",
        )
        if failed:
            print(f"\nSession finished: tests failed, nothing trained ({name}).")
            return
    place_start_runs()

    runs = LAUNCH["runs"]
    print(f"\n=== training {', '.join(run['name'] for run in runs)} ===\n", flush=True)
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        endings = list(
            pool.map(train_and_evaluate, runs, range(len(runs)), [console] * len(runs))
        )
    print(f"\nSession finished: {'; '.join(endings)} ({name}).", flush=True)


if __name__ == "__main__":
    main()
