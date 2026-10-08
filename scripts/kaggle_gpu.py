"""Train on Kaggle's GPUs from the laptop, while the GPU desktop cannot be reached.

Each launch is one training run and its evaluation, in one session of the
private Kaggle notebook ``<user>/centipede-training``: a script that runs on
a T4 in the background, for up to 12 hours, whether or not the laptop is on.
This script, run on the laptop from the repository root with the project's
Python, controls it through Kaggle's command-line tool, installed in the
project's environment and signed in once with ``kaggle auth login``:

    python scripts/kaggle_gpu.py run configs/spine_movement/02_movement_1.toml
    python scripts/kaggle_gpu.py watch    # follow the session, then fetch
    python scripts/kaggle_gpu.py status   # queued, running, complete, or error
    python scripts/kaggle_gpu.py fetch    # copy its results here once it ends

``run`` checks the configuration file, and that GitHub has the laptop's code,
since Kaggle clones it from there at the laptop's commit. If the file starts
from another run, that run (its configuration and checkpoints) is uploaded
once as a private Kaggle dataset. ``run`` then pushes ``kaggle_session.py``,
filled in with the configuration file as it is on the laptop and with the
``evaluation.toml`` beside it, if there is one, which evaluates the new run.
The session first runs the tests, and trains only if they pass;
``--skip-tests`` leaves them out.

``run`` then watches, like ``watch``: it shows the session's output as
Kaggle streams it, one line per window once the window is done, and when the
session ends copies its output into the laptop's ``runs/``: the run folder,
as on the desktop, and ``runs/kaggle/<launch>/`` with the console output of
every step and the session's whole output. Ctrl+C stops watching, not the
session; its output stays on Kaggle, and ``fetch`` copies it at any time,
waiting first while it runs. Only the latest session is followed or fetched,
so a new one is refused until it has ended.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime
from pathlib import Path

from gpu_desktop import EVALUATION_FILE, LAPTOP_RUNS, git, require_pushed_code

from centipede.experiment.configuration import read_configuration, run_folder_of
from centipede.experiment.run_folder import RunFolder
from centipede.settings_section import SettingsError

# Kaggle's tool, installed next to the project's Python.
KAGGLE = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
# It writes the session's output, which is UTF-8, to the terminal and to files.
KAGGLE_ENVIRONMENT = {**os.environ, "PYTHONUTF8": "1"}
SESSION_FILE = Path(__file__).with_name("kaggle_session.py")
LAUNCH_LINE = "LAUNCH: dict = {}"
NOTEBOOK_SLUG = "centipede-training"
NOTEBOOK_TITLE = "Centipede training"
ACCELERATOR = "NvidiaTeslaT4"
KAGGLE_RUNS = LAPTOP_RUNS / "kaggle"
# Kaggle's names for a session that has not ended yet.
WAITING = ("queued", "new_script")
UNFINISHED = (*WAITING, "running", "cancel_requested")
STATUS = re.compile(r'has status "(?:\w+\.)?(\w+)"')
NOT_FOUND = re.compile(r"404|403|was denied")
RUN_LINE = re.compile(r"^\[run\][ \t]*$", re.MULTILINE)
WAIT_SECONDS = 60
DATASET_WAIT_SECONDS = 10
DATASET_ATTEMPTS = 60
LAST_LINES_SHOWN = 20


def kaggle(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Kaggle's tool, capturing its output."""
    return subprocess.run(
        [str(KAGGLE), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=KAGGLE_ENVIRONMENT,
    )


def kaggle_or_stop(*arguments: str) -> str:
    """Kaggle's answer; a failure ends the script with it."""
    result = kaggle(*arguments)
    if result.returncode:
        sys.exit(
            f"kaggle {' '.join(arguments)} failed:\n{result.stdout}{result.stderr}"
        )
    return result.stdout


def missing(result: subprocess.CompletedProcess[str]) -> bool:
    """Whether Kaggle answered that the thing asked about does not exist.

    For a notebook or dataset the user does not have, Kaggle answers that it
    is not found or that reading it is denied. Any other failure ends the
    script.
    """
    if not result.returncode:
        return False
    if NOT_FOUND.search(result.stdout + result.stderr):
        return True
    sys.exit(f"Kaggle failed:\n{result.stdout}{result.stderr}")


def username() -> str:
    found = re.search(r"- username: (\S+)", kaggle_or_stop("config", "view"))
    if not found or found.group(1) == "None":
        sys.exit("Sign in to Kaggle first: kaggle auth login")
    return found.group(1)


def notebook_id() -> str:
    return f"{username()}/{NOTEBOOK_SLUG}"


def session_status(notebook: str) -> str | None:
    """The latest session's status, such as running or complete; None before any."""
    result = kaggle("kernels", "status", notebook)
    if missing(result):
        return None
    found = STATUS.search(result.stdout)
    if not found:
        sys.exit(f"Unexpected answer from Kaggle: {result.stdout}")
    return found.group(1).lower()


# -- Starting a session ----------------------------------------------------------------


def start_run_dataset(start_from: Path, user: str) -> dict[str, str]:
    """Make sure Kaggle has the run a training starts from; where the session finds it.

    The run's saved configuration and every checkpoint become a private
    dataset named after the run folder, uploaded only once.
    """
    if start_from.is_absolute():
        sys.exit("start_from must be relative to the repository, such as runs/<run>.")
    folder = run_folder_of(start_from)
    checkpoint = RunFolder.open(start_from).checkpoint_named_by(start_from)
    slug = ("centipede-" + re.sub(r"[^a-z0-9]+", "-", folder.name.lower()))[:50]
    slug = slug.strip("-")
    dataset = f"{user}/{slug}"
    if missing(kaggle("datasets", "status", dataset)):
        print(f"Uploading {folder.as_posix()} to the private dataset {dataset}")
        with tempfile.TemporaryDirectory() as temporary:
            upload = Path(temporary)
            shutil.copy(folder / "configuration.toml", upload)
            for path in (folder / "checkpoints").glob("cycle_*.pt"):
                shutil.copy(path, upload)
            metadata = {
                "title": f"Centipede {folder.name}"[:50],
                "id": dataset,
                "licenses": [{"name": "CC0-1.0"}],
            }
            (upload / "dataset-metadata.json").write_text(json.dumps(metadata))
            kaggle_or_stop("datasets", "create", "-p", temporary, "-q")
    for _ in range(DATASET_ATTEMPTS):
        status = kaggle("datasets", "status", dataset).stdout.strip()
        if status == "ready":
            break
        print(f"  dataset {status or 'not ready'}, waiting", flush=True)
        time.sleep(DATASET_WAIT_SECONDS)
    else:
        sys.exit(f"The dataset {dataset} is not ready; try again later.")
    return {"folder": folder.as_posix(), "dataset": slug, "checkpoint": checkpoint.name}


def with_session_runs_folder(text: str) -> str:
    """The training file, its run written to the session's output folder."""
    if "runs_folder" in tomllib.loads(text)["run"]:
        sys.exit("A Kaggle session sets [run] runs_folder itself; leave it out.")
    text, count = RUN_LINE.subn(
        '[run]\nruns_folder = "/kaggle/working"  # set by kaggle_gpu.py', text, count=1
    )
    if not count:
        sys.exit("The file's [run] header must stand alone on its line.")
    return text


def push(notebook: str, launch: dict, datasets: list[str]) -> None:
    """Push the session file with the launch filled in; Kaggle starts running it."""
    session = SESSION_FILE.read_text(encoding="utf-8")
    metadata = {
        "id": notebook,
        "title": NOTEBOOK_TITLE,
        "code_file": SESSION_FILE.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": ACCELERATOR,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / SESSION_FILE.name).write_text(
            session.replace(LAUNCH_LINE, f"LAUNCH = {launch!r}"), encoding="utf-8"
        )
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata))
        answer = kaggle_or_stop(
            "kernels", "push", "-p", temporary, "--accelerator", ACCELERATOR
        )
    if "successfully pushed" not in answer:
        sys.exit(f"Kaggle did not start the session:\n{answer}")


# -- Following and copying a session ---------------------------------------------------


def session_text(log: str) -> str:
    """The session's output out of Kaggle's log, a JSON list of pieces of output."""
    try:
        return "".join(piece["data"] for piece in json.loads(log))
    except (ValueError, TypeError, KeyError):
        return log


def wait_until_ended(notebook: str) -> str:
    """Wait while the latest session has not ended; how it ended."""
    status = session_status(notebook)
    if status is None:
        sys.exit("The notebook has no session yet; start one with 'run'.")
    while status in UNFINISHED:
        print(f"{datetime.now():%H:%M} session {status}, waiting for it to end")
        time.sleep(WAIT_SECONDS)
        status = session_status(notebook)
    return status


def fetch_session(notebook: str) -> int:
    """Copy the latest session's output into the laptop's runs once it has ended."""
    status = wait_until_ended(notebook)
    download = KAGGLE_RUNS / "download"
    shutil.rmtree(download, ignore_errors=True)
    kaggle_or_stop("kernels", "output", notebook, "-p", str(download), "-o", "-q")
    launches = sorted((download / "kaggle").glob("*"))
    name = launches[-1].name if launches else f"{datetime.now():%Y-%m-%d_%H%M%S}"
    console = download / "kaggle" / name
    console.mkdir(parents=True, exist_ok=True)
    # Loose files are Kaggle's: its log of the session's output, and pages
    # made from it.
    output = ""
    for path in [path for path in download.iterdir() if path.is_file()]:
        if path.suffix == ".log":
            output = session_text(path.read_text(encoding="utf-8"))
            (console / "session.txt").write_text(output, encoding="utf-8")
            path.unlink()
        else:
            shutil.move(path, console / path.name)
    run_folders = [
        path.name
        for path in download.iterdir()
        if path.is_dir() and path.name != "kaggle"
    ]
    shutil.copytree(download, LAPTOP_RUNS, dirs_exist_ok=True)
    shutil.rmtree(download)

    lines = output.splitlines()
    finished = [line for line in lines if line.startswith("Session finished")]
    print("\n".join(finished[-1:] or lines[-LAST_LINES_SHOWN:]))
    print(f"\nSession {status}; copied to the laptop:")
    for folder in run_folders:
        print(f"  runs/{folder}")
        if (LAPTOP_RUNS / folder / "report.html").exists():
            print(f"    report: runs/{folder}/report.html")
        for report in sorted((LAPTOP_RUNS / folder / "evaluations").glob("*.html")):
            print(f"    evaluation: runs/{folder}/evaluations/{report.name}")
    print(f"  runs/kaggle/{name}/ (console output of every step)")
    return 0 if status == "complete" and finished else 1


def watch_session(notebook: str) -> int:
    """Show the latest session's output as it comes, then copy its results here."""
    try:
        status = session_status(notebook)
        if status in (None, *WAITING):
            print("Waiting for Kaggle to start the session ...", flush=True)
        while status in (None, *WAITING):
            time.sleep(DATASET_WAIT_SECONDS)
            status = session_status(notebook)
        if status == "running":
            subprocess.run(
                [str(KAGGLE), "kernels", "logs", notebook, "--follow"],
                env=KAGGLE_ENVIRONMENT,
            )
        return fetch_session(notebook)
    except KeyboardInterrupt:
        print(
            "\nStopped watching; the session goes on. 'watch' follows it again,"
            " and 'fetch' copies its results once it ends."
        )
        return 130


# -- Commands --------------------------------------------------------------------------


def run(arguments: argparse.Namespace) -> int:
    configuration_path: Path = arguments.configuration
    try:
        configuration = read_configuration(configuration_path)
    except SettingsError as error:
        sys.exit(f"Configuration error: {error}")
    if configuration.evaluation is not None or configuration.continue_from is not None:
        sys.exit("Kaggle sessions train new runs only: give a training file.")
    evaluation = configuration_path.with_name(EVALUATION_FILE)
    evaluation_text = (
        evaluation.read_text(encoding="utf-8") if evaluation.exists() else ""
    )
    if evaluation_text:
        # Its source is replaced by the new run's folder in the session.
        run_values = tomllib.loads(evaluation_text).get("run", {})
        if run_values.get("mode") != "evaluate" or "source" not in run_values:
            sys.exit(f'{evaluation} needs mode = "evaluate" and a source in [run].')
    require_pushed_code()
    user = username()
    notebook = f"{user}/{NOTEBOOK_SLUG}"
    if session_status(notebook) in UNFINISHED:
        sys.exit("The latest session has not ended; 'watch' it or stop it on Kaggle.")
    start_from = configuration.run.start_from
    start_runs = [start_run_dataset(start_from, user)] if start_from else []
    name = f"{datetime.now():%Y-%m-%d_%H%M%S}_{configuration_path.stem}"
    commit = git("rev-parse", "HEAD")
    launch = {
        "name": name,
        "commit": commit,
        "tests": not arguments.skip_tests,
        "training": with_session_runs_folder(
            configuration_path.read_text(encoding="utf-8")
        ),
        "evaluation": evaluation_text,
        "start_runs": start_runs,
    }
    push(notebook, launch, [f"{user}/{start['dataset']}" for start in start_runs])
    print(f"Started {name} on Kaggle")
    print(f"  configuration  {configuration_path.as_posix()}")
    print(f"  evaluation     {evaluation.as_posix() if evaluation_text else 'none'}")
    print(f"  tests          {'first' if launch['tests'] else 'skipped'}")
    print(f"  code           {commit[:7]}")
    print(f"  notebook       https://www.kaggle.com/code/{notebook}")
    if arguments.detach:
        print("Follow it with 'watch', or copy its results later with 'fetch'.")
        return 0
    print("Ctrl+C stops watching; the session goes on.\n")
    return watch_session(notebook)


def watch(_: argparse.Namespace) -> int:
    notebook = notebook_id()
    if session_status(notebook) is None:
        sys.exit("The notebook has no session yet; start one with 'run'.")
    return watch_session(notebook)


def fetch(_: argparse.Namespace) -> int:
    try:
        return fetch_session(notebook_id())
    except KeyboardInterrupt:
        print("\nStopped waiting; the session goes on.")
        return 130


def status(_: argparse.Namespace) -> int:
    notebook = notebook_id()
    print(f"{notebook}: {session_status(notebook) or 'no session yet'}")
    print(f"  https://www.kaggle.com/code/{notebook}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    run_parser = commands.add_parser(
        "run", help="start a training and its evaluation on Kaggle, watch, then fetch"
    )
    run_parser.add_argument("configuration", type=Path, help="a training TOML file")
    run_parser.add_argument(
        "--skip-tests", action="store_true", help="train without running the tests"
    )
    run_parser.add_argument(
        "--detach", action="store_true", help="start the session without watching it"
    )
    run_parser.set_defaults(action=run)
    for name, action, description in (
        ("watch", watch, "follow the latest session's output, then copy it here"),
        ("fetch", fetch, "copy the latest session's output here once it ends"),
        ("status", status, "whether the latest session is queued, running, or ended"),
    ):
        commands.add_parser(name, help=description).set_defaults(action=action)
    arguments = parser.parse_args()
    sys.exit(arguments.action(arguments))


if __name__ == "__main__":
    main()
