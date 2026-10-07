"""Run a folder of training files one after another, on the GPU desktop.

``gpu_desktop.py queue FOLDER`` copies the folder into the desktop's
``runs/launched/`` and starts this script there, detached from SSH:

    python -u scripts/run_queue.py runs/launched/<queue name>

Every training file in the folder, in the order of their names, becomes one
launched run, named ``<queue name>_<file name>``, with its file and
console output in ``runs/launched/`` like a run started with ``run``, so
``status``, ``watch``, and ``fetch`` know each one. If the folder also holds
``evaluation.toml``, each training run that finished is evaluated right after
it with that file, its ``source`` replaced by the run's folder. A run that
fails does not stop the queue. The queue's own output, which ``watch`` follows,
holds every run's output in turn and ends with a table of how each one ended.
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# The last line of every run's output, as the launcher writes it.
END_MARKER = "run ended with exit code"
EVALUATION_FILE = "evaluation.toml"
RUN_FOLDER = re.compile(r"Run folder: (?:.*[\\/])?runs[\\/]([^\\/\r\n]+)")
SOURCE_LINE = re.compile(r"^source\s*=.*$", re.MULTILINE)


def run_one(name: str, text: str, launched: Path) -> tuple[int, str | None]:
    """Run one configuration as a launched run; its exit code and run folder.

    The output goes to the run's own console file and, as it comes, to this
    queue's output.
    """
    file = launched / f"{name}.toml"
    file.write_text(text, encoding="utf-8")
    console_path = launched / f"{name}.txt"
    environment = {**os.environ, "PYTHONUTF8": "1"}
    with console_path.open("wb") as console:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "centipede", str(file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        assert process.stdout is not None
        while chunk := os.read(process.stdout.fileno(), 4096):
            console.write(chunk)
            console.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        exit_code = process.wait()
        console.write(f"{END_MARKER} {exit_code}\r\n".encode())
    found = RUN_FOLDER.search(
        console_path.read_text(encoding="utf-8", errors="replace")
    )
    return exit_code, found.group(1).strip() if found else None


def duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def main() -> int:
    queue_folder = Path(sys.argv[1]).resolve()
    launched = queue_folder.parent
    evaluation = queue_folder / EVALUATION_FILE
    files = sorted(
        path for path in queue_folder.glob("*.toml") if path.name != EVALUATION_FILE
    )
    queue_start = time.monotonic()
    print(
        f"Queue {queue_folder.name}: {len(files)} training runs"
        + (", each evaluated after it" if evaluation.exists() else ""),
        flush=True,
    )
    outcomes = []
    for number, file in enumerate(files, 1):
        name = f"{queue_folder.name}_{file.stem}"
        print(
            f"\n=== {number}/{len(files)}  {file.stem}  (started"
            f" {datetime.now():%H:%M}, queue running for"
            f" {duration(time.monotonic() - queue_start)}) ===\n",
            flush=True,
        )
        run_start = time.monotonic()
        exit_code, folder = run_one(name, file.read_text(encoding="utf-8"), launched)
        trained = "finished" if exit_code == 0 else f"failed ({exit_code})"
        evaluated = "-"
        if exit_code == 0 and folder and evaluation.exists():
            print(f"\n--- evaluating {folder} ---\n", flush=True)
            text = SOURCE_LINE.sub(
                f'source = "runs/{folder}"', evaluation.read_text(encoding="utf-8")
            )
            evaluation_code, _ = run_one(f"{name}_evaluation", text, launched)
            evaluated = (
                "finished" if evaluation_code == 0 else f"failed ({evaluation_code})"
            )
        outcomes.append(
            (
                file.stem,
                trained,
                evaluated,
                duration(time.monotonic() - run_start),
                folder,
            )
        )

    print(f"\n=== Queue finished in {duration(time.monotonic() - queue_start)} ===")
    width = max(len(stem) for stem, *_ in outcomes) if outcomes else 0
    for stem, trained, evaluated, took, folder in outcomes:
        print(
            f"  {stem:<{width}}  training {trained:<12}  evaluation {evaluated:<12}"
            f"  {took:>6}  runs/{folder or '-'}"
        )
    return 0 if all(trained == "finished" for _, trained, *_ in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
