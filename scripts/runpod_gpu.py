"""Train on Runpod's GPUs from the laptop, one rented pod per run.

A launch trains up to four runs at the same time, each on a pod of its own
with one GPU, rented through Runpod's REST API: by default an RTX 5090 in the
community cloud. Each pod runs ``cloud_session.py``, as a Kaggle session
does: it clones Centipede at the laptop's commit, and RL_lib and MujocoReplay
as they are on GitHub, installs them, runs the tests, places the run that
training starts from, trains, and evaluates. This script, run on the laptop
from the repository root with the project's Python, controls the pods:

    python scripts/runpod_gpu.py run configs/pink_study/03_worlds1024.toml
    python scripts/runpod_gpu.py run <file 1> <file 2>  # two runs, two pods
    python scripts/runpod_gpu.py run configs/reward_study/01_baseline.toml --seeds 1 2
    python scripts/runpod_gpu.py watch    # follow the latest launch's runs
    python scripts/runpod_gpu.py status   # recent launches, and the pods rented now
    python scripts/runpod_gpu.py fetch    # copy the latest launch's runs once they end
    python scripts/runpod_gpu.py stop     # copy what there is, then delete the pods

``run`` checks the configuration files, that GitHub has the laptop's code,
and that at most four pods would be rented at once, counting those rented
already. It rents the pods one after another, waits until each answers over
SSH, and checks that its GPU is idle: a pod whose GPU is already busy, so
shared or faulty, is deleted and another rented. Then it copies to each its
session, filled in with its configuration file as it is on the
laptop and the ``evaluation.toml`` beside it, if there is one, and the run
that training starts from, if any. With ``--seeds``, each run gets one of the
seeds and the file's name with ``_seed<N>`` added. ``runpod_pod.sh`` then
runs the session on the pod in the background. If anything fails before
every session has started, the launch's pods are deleted.

Each pod has a copier: a process of its own on the laptop, as for the GPU
desktop, that checks the pod every minute and, once its session has ended,
copies its output into the laptop's ``runs/``: the run folder, and
``runs/runpod/<launch>/`` with the console output of every step. Then it
deletes the pod. A copier stopped before the end, for example by restarting
the laptop, is started again by the next command; its log is
``runs/runpod/<launch>/copier.txt``. As a backstop, a pod deletes itself
three hours after its session ends, and in any case at the end of its
lifetime (``--max-hours``, 12 by default).

``run`` then shows the sessions' output, one line per window, each line
marked with its run's file or seed when there are several, and waits for the
copies. Ctrl+C stops watching, not the runs or their copying. ``watch``,
``fetch``, and ``stop`` act on the latest launch, or on the launches whose
names contain the text given.

Runpod needs, once: an API key, in the environment variable
``RUNPOD_API_KEY`` or in ``~/.runpod/config.toml`` as ``apikey = "..."``,
where Runpod's own tools read it; and the laptop's SSH public key, registered
in the Runpod account, with which the laptop reaches the pods. Run folders
made on a pod are named by its clock, in UTC.
"""

import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from gpu_desktop import (
    LAPTOP_RUNS,
    NO_INPUT,
    git,
    require_pushed_code,
    start_detached,
    take_lock,
)
from kaggle_gpu import LAUNCH_LINE, SESSION_FILE, evaluation_beside, with_run_values

from centipede.experiment.configuration import (
    Configuration,
    read_configuration,
    run_folder_of,
)
from centipede.experiment.run_folder import RunFolder
from centipede.settings_section import SettingsError

API = "https://api.runpod.io/v2"
API_KEY_FILE = Path.home() / ".runpod" / "config.toml"
# Runpod's official PyTorch 2.8 image, CUDA 12.8, checked on an RTX 5090
# (plan.md, Stage 8.4); a host needs CUDA 12.8 or newer to run it.
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
MIN_CUDA_VERSION = "12.8"
# The container's command: the image's own start script (which starts the SSH
# server), after making the home folder writable by root alone. On some hosts
# it is writable by everyone, and the SSH server then refuses every key
# (found 2026-10-09).
POD_COMMAND = ["bash", "-c", "chmod 755 /root && exec /start.sh"]
DISK_GB = 30
DEFAULT_GPU = "NVIDIA GeForce RTX 5090"
DEFAULT_CLOUD = "COMMUNITY"
# The most pods rented at once, counting those rented already (decided with
# the user, 2026-10-09).
MAX_PODS = 4
# A pod deletes itself this long after its session ends, unless its copier
# has deleted it already.
GRACE_HOURS = 3
DEFAULT_MAX_HOURS = 12
RUNPOD_RUNS = LAPTOP_RUNS / "runpod"
POD_JOB_FILE = Path(__file__).with_name("runpod_pod.sh")
# Paths on the pod.
POD_OUTPUT = "/root/output"
POD_INPUT = "/root/input"
POD_SESSION_EXIT = "/root/session_exit"
POD_SESSION_PID = "/root/session.pid"
SSH_WAIT_SECONDS = 600
# A pod whose GPU is busier than this before any of our work is shared or
# faulty, and is replaced, this many times at most (a community host's GPU
# was 99% busy before any work on 2026-10-09, and trained at half speed).
IDLE_GPU_PERCENT = 10
GPU_REPLACEMENTS = 2
POLL_SECONDS = 5
CHECK_SECONDS = 60
COPY_ATTEMPTS = 3
SSH_UNREACHABLE = 255
INTERRUPTED = 130
STAMP_FORMAT = "%Y-%m-%d_%H%M%S"
STAMP_LENGTH = len(datetime.now().strftime(STAMP_FORMAT))
RECENT_LAUNCHES_SHOWN = 8
# A copier's log records the copy on a line holding COPIED, and its last line
# holds DONE once the copier has ended; a clean end, CLEAN_END too. 'fetch'
# adds a line to try a copy again.
COPIED = " copied to the laptop: "
DONE = " done: "
CLEAN_END = "pod deleted (session exit code 0)"


class RunpodError(Exception):
    """Runpod refused a request, could not be reached, or a pod failed to start."""


class PodGone(Exception):
    """The pod no longer exists."""


def now() -> str:
    return f"{datetime.now():%H:%M}"


# -- Runpod's REST API -----------------------------------------------------------------


@functools.cache
def api_key() -> str:
    """Runpod's API key, found where Runpod's own tools look for it."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key and API_KEY_FILE.exists():
        try:
            key = tomllib.loads(API_KEY_FILE.read_text(encoding="utf-8")).get("apikey")
        except tomllib.TOMLDecodeError as error:
            sys.exit(f"{API_KEY_FILE} is not valid TOML: {error}")
    if not key:
        sys.exit(
            "No Runpod API key: create one in the Runpod console (Settings, API"
            ' Keys), then set RUNPOD_API_KEY or write apikey = "..." in'
            f" {API_KEY_FILE}."
        )
    return key


def runpod(method: str, path: str, body: dict | None = None) -> dict | None:
    """Runpod's answer to one request: its JSON, or None for no body or a 404."""
    request = urllib.request.Request(
        API + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "centipede-runpod-launcher",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        detail = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise RunpodError(f"Runpod answered {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise RunpodError(f"Runpod could not be reached: {error}") from None
    return json.loads(text) if text else None


def rented_pods() -> list[dict]:
    answer = runpod("GET", "/pods")
    return answer["pods"] if answer else []


def delete_pod(pod_id: str) -> None:
    runpod("DELETE", f"/pods/{pod_id}")


# -- One launch: a run on its own pod --------------------------------------------------


@dataclass
class Launch:
    """A run on a pod of its own, recorded in ``runs/runpod/<name>/launch.json``."""

    name: str
    label: str
    pod_id: str
    gpu: str
    cloud: str
    cost_per_hour: float
    rented_at: str
    host: str = ""
    port: int = 0

    @property
    def folder(self) -> Path:
        return RUNPOD_RUNS / self.name

    @property
    def copier_log(self) -> Path:
        return self.folder / "copier.txt"

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "launch.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, name: str) -> "Launch":
        return cls(**json.loads((RUNPOD_RUNS / name / "launch.json").read_text()))

    def ssh_options(self) -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            # Each pod has host keys of its own, first seen when it starts.
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={(self.folder / 'known_hosts').as_posix()}",
        ]

    def ssh_command(self, command: str) -> list[str]:
        return [
            "ssh",
            *self.ssh_options(),
            "-p",
            str(self.port),
            f"root@{self.host}",
            command,
        ]

    def ssh(self, command: str) -> subprocess.CompletedProcess[str]:
        """Run a command on the pod; its exit code and output."""
        return subprocess.run(
            self.ssh_command(command),
            stdin=NO_INPUT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def scp(self, *paths: str) -> int:
        """Copy with scp; a path on the pod is written ``:<path>``."""
        pod = f"root@{self.host}"
        paths = tuple(pod + path if path.startswith(":") else path for path in paths)
        return subprocess.run(
            ["scp", "-q", "-r", *self.ssh_options(), "-P", str(self.port), *paths],
            stdin=NO_INPUT,
        ).returncode


def launch_names() -> list[str]:
    return sorted(path.parent.name for path in RUNPOD_RUNS.glob("*/launch.json"))


def chosen_launches(pattern: str | None) -> list[Launch]:
    """The launches whose names contain ``pattern``; by default the latest's."""
    names = launch_names()
    if not names:
        sys.exit("No launch yet; start one with 'run'.")
    if pattern:
        chosen = [name for name in names if pattern in name]
    else:
        latest = names[-1][:STAMP_LENGTH]
        chosen = [name for name in names if name.startswith(latest)]
    if not chosen:
        sys.exit(f"No launch matches {pattern!r}.")
    return [Launch.load(name) for name in chosen]


# -- Starting a launch -----------------------------------------------------------------


@dataclass
class Job:
    """One run of a launch, ready to be sent to its pod."""

    name: str
    label: str
    session: dict
    # The start run's files, and the input folder they go to on the pod.
    start_files: list[Path]
    start_folder: str


def training_configuration(path: Path) -> Configuration:
    try:
        configuration = read_configuration(path)
    except SettingsError as error:
        sys.exit(f"Configuration error in {path}: {error}")
    if configuration.evaluation is not None or configuration.continue_from is not None:
        sys.exit("Runpod launches train new runs only: give training files.")
    return configuration


def prepared_jobs(arguments: argparse.Namespace, stamp: str, commit: str) -> list[Job]:
    """Each run of the launch: its file, filled in, and its session."""
    paths: list[Path] = arguments.configurations
    configurations = [training_configuration(path) for path in paths]
    if arguments.seeds and len(paths) > 1:
        sys.exit("--seeds trains one file with each seed: give a single file.")
    if len({configuration.run.name for configuration in configurations}) < len(paths):
        sys.exit("Two files need different run names, or their folders would clash.")
    runs = (
        [(paths[0], configurations[0], seed) for seed in arguments.seeds]
        if arguments.seeds
        else [
            (path, configuration, None)
            for path, configuration in zip(paths, configurations, strict=True)
        ]
    )
    if len(runs) > MAX_PODS:
        sys.exit(f"At most {MAX_PODS} runs per launch, one pod each.")
    jobs = []
    for path, configuration, seed in runs:
        values: dict[str, str | int] = {"runs_folder": POD_OUTPUT}
        name = f"{stamp}_{path.stem}"
        label = path.stem if len(runs) > 1 else ""
        if seed is not None:
            label = f"seed{seed}"
            name += f"_{label}"
            values |= {"seed": seed, "name": f"{configuration.run.name}_{label}"}
        start_files, start_runs, start_folder = [], [], ""
        start_from = configuration.run.start_from
        if start_from is not None:
            if start_from.is_absolute():
                sys.exit(
                    "start_from must be relative to the repository, such as runs/<run>."
                )
            folder = run_folder_of(start_from)
            checkpoint = RunFolder.open(start_from).checkpoint_named_by(start_from)
            start_files = [folder / "configuration.toml", checkpoint]
            start_folder = f"{POD_INPUT}/{folder.name}"
            start_runs = [
                {
                    "folder": folder.as_posix(),
                    "dataset": folder.name,
                    "checkpoint": checkpoint.name,
                }
            ]
        session = {
            "name": name,
            "commit": commit,
            "tests": not arguments.skip_tests,
            "runs": [
                {
                    "name": name,
                    "label": "",
                    "training": with_run_values(
                        path.read_text(encoding="utf-8"), values
                    ),
                }
            ],
            "evaluation": evaluation_beside(path),
            "start_runs": start_runs,
            "output": POD_OUTPUT,
            "console": "runpod",
            "input": POD_INPUT,
        }
        jobs.append(Job(name, label, session, start_files, start_folder))
    return jobs


def wait_for_ssh(launch: Launch) -> None:
    """Wait until the pod answers over SSH, and record where it answers."""
    deadline = time.monotonic() + SSH_WAIT_SECONDS
    while time.monotonic() < deadline:
        pod = runpod("GET", f"/pods/{launch.pod_id}")
        if pod is None:
            raise RunpodError(f"Pod {launch.pod_id} disappeared while starting.")
        direct = (pod.get("ssh") or {}).get("direct")
        if direct:
            launch.host, launch.port = direct["host"], direct["port"]
            if launch.ssh("true").returncode == 0:
                launch.save()
                return
        time.sleep(POLL_SECONDS)
    raise RunpodError(
        f"Pod {launch.pod_id} did not answer over SSH within"
        f" {SSH_WAIT_SECONDS // 60} minutes; its machine may be faulty."
    )


def start_session(launch: Launch, job: Job, max_hours: int) -> None:
    """Copy the session and the start run to the pod, and start the session."""
    if job.start_files:
        made = launch.ssh(f"mkdir -p {job.start_folder}")
        if made.returncode or launch.scp(
            *map(str, job.start_files), f":{job.start_folder}/"
        ):
            raise RunpodError(f"Copying the start run to pod {launch.pod_id} failed.")
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        session = SESSION_FILE.read_text(encoding="utf-8").replace(
            LAUNCH_LINE, f"LAUNCH = {job.session!r}"
        )
        # Linux line ends, whatever the laptop's checkout has.
        files = {
            folder / SESSION_FILE.name: session,
            folder / POD_JOB_FILE.name: POD_JOB_FILE.read_text(encoding="utf-8"),
        }
        for path, text in files.items():
            path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        if launch.scp(*map(str, files), ":/root/"):
            raise RunpodError(f"Copying the session to pod {launch.pod_id} failed.")
    job_command = (
        f"setsid nohup bash /root/{POD_JOB_FILE.name} {launch.name}"
        f" {GRACE_HOURS * 3600} {max_hours * 3600} > /dev/null 2>&1 < /dev/null &"
    )
    if launch.ssh(job_command).returncode:
        raise RunpodError(f"Starting the session on pod {launch.pod_id} failed.")


def rent_pod(job: Job, arguments: argparse.Namespace) -> Launch:
    """Rent one pod for a job, and record it."""
    pod = runpod(
        "POST",
        "/pods",
        {
            "name": job.name,
            "image": IMAGE,
            "cmd": POD_COMMAND,
            "cloud": arguments.cloud,
            "gpu": {
                "id": arguments.gpu,
                "count": 1,
                "minCudaVersion": MIN_CUDA_VERSION,
            },
            "disk": DISK_GB,
            "ports": ["22/tcp"],
            "startSsh": True,
        },
    )
    assert pod is not None
    launch = Launch(
        name=job.name,
        label=job.label,
        pod_id=pod["id"],
        gpu=arguments.gpu,
        cloud=arguments.cloud,
        cost_per_hour=pod["cost"],
        rented_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    launch.save()
    print(f"{now()} rented pod {launch.pod_id} for {job.name}", flush=True)
    return launch


def gpu_busy_percent(launch: Launch) -> int:
    """How busy the pod's GPU is before any of our work: the median of five
    samples a second apart."""
    answer = launch.ssh(
        "for sample in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu"
        " --format=csv,noheader,nounits; sleep 1; done"
    )
    samples = sorted(int(value) for value in answer.stdout.split() if value.isdigit())
    if answer.returncode or len(samples) < 5:
        raise RunpodError(f"nvidia-smi failed on pod {launch.pod_id}.")
    return samples[2]


def rent_and_start(jobs: list[Job], arguments: argparse.Namespace) -> list[Launch]:
    """Rent a pod per job, with an idle GPU, and start its session; on any
    failure, delete them all."""
    launches: list[Launch] = []
    try:
        for job in jobs:
            for _ in range(1 + GPU_REPLACEMENTS):
                launch = rent_pod(job, arguments)
                launches.append(launch)
                wait_for_ssh(launch)
                busy = gpu_busy_percent(launch)
                if busy <= IDLE_GPU_PERCENT:
                    break
                print(
                    f"{now()} pod {launch.pod_id}'s GPU is {busy}% busy before our"
                    " work, so shared or faulty; renting another",
                    flush=True,
                )
                delete_pod(launch.pod_id)
                launches.remove(launch)
                shutil.rmtree(launch.folder, ignore_errors=True)
            else:
                raise RunpodError(f"Every pod rented for {job.name} had a busy GPU.")
            start_session(launch, job, arguments.max_hours)
            print(
                f"{now()} started {launch.name} on {launch.host}:{launch.port}",
                flush=True,
            )
    except (RunpodError, KeyboardInterrupt) as error:
        for launch in launches:
            try:
                delete_pod(launch.pod_id)
            except RunpodError:
                print(
                    f"Delete pod {launch.pod_id} in the Runpod console.",
                    file=sys.stderr,
                )
            shutil.rmtree(launch.folder, ignore_errors=True)
        reason = "Interrupted" if isinstance(error, KeyboardInterrupt) else str(error)
        sys.exit(f"{reason}\nThe launch's pods were deleted; nothing runs.")
    return launches


# -- Copying a launch here -------------------------------------------------------------


def session_exit_code(launch: Launch) -> str | None:
    """The session's exit code once it has ended; None while it runs, or while
    the pod cannot be checked. Raises PodGone when the pod no longer exists."""
    try:
        if runpod("GET", f"/pods/{launch.pod_id}") is None:
            raise PodGone
    except RunpodError as error:
        print(f"{now()} {error}; checking again in a minute", flush=True)
        return None
    answer = launch.ssh(f"cat {POD_SESSION_EXIT} 2>/dev/null")
    if answer.returncode == SSH_UNREACHABLE:
        print(
            f"{now()} the pod cannot be reached; checking again in a minute", flush=True
        )
        return None
    return answer.stdout.strip() or None


def copy_output(launch: Launch) -> list[str] | None:
    """Copy the pod's output into the laptop's runs; the run folders, or None
    if the copy failed."""
    download = launch.folder / "download"
    shutil.rmtree(download, ignore_errors=True)
    if launch.scp(f":{POD_OUTPUT}", str(download)):
        return None
    run_folders = [
        path.name
        for path in download.iterdir()
        if path.is_dir() and path.name != "runpod"
    ]
    shutil.copytree(download, LAPTOP_RUNS, dirs_exist_ok=True)
    shutil.rmtree(download)
    return run_folders


def copy_launch(name: str) -> int:
    """Wait for a pod's session to end, copy its output here, delete the pod.

    This is the copier; its output is its log. It holds the launch's lock
    while it runs, so that only one copies each launch.
    """
    launch = Launch.load(name)
    lock = take_lock(launch.folder / "copier.lock")
    if lock is None:
        # Started twice at once; the other copier writes the log.
        return 0
    print(f"{now()} copying {name} here once its session ends", flush=True)
    try:
        while (exit_code := session_exit_code(launch)) is None:
            time.sleep(CHECK_SECONDS)
    except PodGone:
        print(
            f"{now()}{DONE}the pod no longer exists (deleted by 'stop' or its backstop)"
        )
        return 1
    print(f"{now()} the session ended with exit code {exit_code}", flush=True)
    for attempt in range(COPY_ATTEMPTS):
        run_folders = copy_output(launch)
        if run_folders is not None:
            break
        if attempt + 1 < COPY_ATTEMPTS:
            time.sleep(CHECK_SECONDS)
    else:
        print(
            f"{now()}{DONE}not copied; the pod deletes itself {GRACE_HOURS} hours"
            " after its session ended: try 'fetch' before then"
        )
        return 1
    copied = ", ".join(f"runs/{folder}" for folder in run_folders) or "no run folder"
    print(f"{now()}{COPIED}{copied}, runs/runpod/{name}/", flush=True)
    try:
        delete_pod(launch.pod_id)
    except RunpodError as error:
        print(f"{now()}{DONE}copied, but deleting pod {launch.pod_id} failed: {error}")
        return 1
    print(f"{now()}{DONE}copied, pod deleted (session exit code {exit_code})")
    return 0


def copier_lines(launch: Launch) -> list[str]:
    if not launch.copier_log.exists():
        return []
    text = launch.copier_log.read_text(encoding="utf-8", errors="replace")
    return text.strip().splitlines()


def copier_done(launch: Launch) -> bool:
    lines = copier_lines(launch)
    return bool(lines) and DONE in lines[-1]


def copier_running(launch: Launch) -> bool:
    lock = take_lock(launch.folder / "copier.lock")
    if lock is None:
        return True
    lock.close()
    return False


def start_copier(launch: Launch) -> None:
    """Start a launch's copier as a process of its own, unless it runs or is done."""
    if copier_done(launch) or copier_running(launch):
        return
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "copy", launch.name]
    start_detached(command, launch.copier_log)


def resume_copiers() -> None:
    """Start again every copier that stopped before its launch was done.

    Only launches whose copier ``run`` started, once their session had: a
    launch still starting has no copier log yet.
    """
    for name in launch_names():
        launch = Launch.load(name)
        if launch.copier_log.exists():
            start_copier(launch)


def wait_for_copiers(launches: list[Launch]) -> int:
    """Wait until every launch's copier is done, show how each ended, and
    return 1 unless each ended cleanly."""
    marked = len(launches) > 1
    unclean = 0
    waiting = list(launches)
    while waiting:
        for launch in list(waiting):
            if not copier_done(launch):
                if launch.copier_log.exists():
                    start_copier(launch)
                continue
            lines = copier_lines(launch)
            ending = (
                lines[-2:] if len(lines) > 1 and COPIED in lines[-2] else lines[-1:]
            )
            prefix = f"[{launch.label}] " if marked else ""
            print("\n".join(prefix + line for line in ending), flush=True)
            unclean += CLEAN_END not in lines[-1]
            waiting.remove(launch)
        time.sleep(2)
    return int(bool(unclean))


# -- Following a launch ----------------------------------------------------------------


def follow_session(launch: Launch, marked: bool, print_lock: threading.Lock) -> None:
    """Show a pod's session output, from its start to its end."""
    session_text = f"{POD_OUTPUT}/runpod/{launch.name}/session.txt"
    command = (
        f"while [ ! -s {POD_SESSION_PID} ]; do sleep 1; done;"
        f" tail -n +1 -f --pid=$(cat {POD_SESSION_PID}) {session_text}"
    )
    prefix = f"[{launch.label}] " if marked else ""
    ssh = subprocess.Popen(
        launch.ssh_command(command), stdin=NO_INPUT, stdout=subprocess.PIPE
    )
    assert ssh.stdout is not None
    for line in iter(ssh.stdout.readline, b""):
        with print_lock:
            print(prefix + line.decode("utf-8", errors="replace").rstrip(), flush=True)
    if ssh.wait() == SSH_UNREACHABLE:
        with print_lock:
            print(
                f"{prefix}The connection to the pod was lost; 'watch' follows it again."
            )


def watch_launches(launches: list[Launch]) -> int:
    """Follow the launches' sessions to their ends, then wait for the copies."""
    marked = len(launches) > 1
    print_lock = threading.Lock()
    followers = [
        threading.Thread(
            target=follow_session, args=(launch, marked, print_lock), daemon=True
        )
        for launch in launches
        if not copier_done(launch)
    ]
    try:
        for follower in followers:
            follower.start()
        while any(follower.is_alive() for follower in followers):
            time.sleep(0.5)
        print()
        return wait_for_copiers(launches)
    except KeyboardInterrupt:
        print("\nStopped watching; the runs and their copying go on.")
        return INTERRUPTED


# -- Commands --------------------------------------------------------------------------


def run(arguments: argparse.Namespace) -> int:
    stamp = datetime.now().strftime(STAMP_FORMAT)
    commit = git("rev-parse", "HEAD")
    jobs = prepared_jobs(arguments, stamp, commit)
    require_pushed_code()
    resume_copiers()
    try:
        rented = len(rented_pods())
    except RunpodError as error:
        sys.exit(str(error))
    if rented + len(jobs) > MAX_PODS:
        sys.exit(
            f"{rented} pods are rented already, and at most {MAX_PODS} at once are"
            f" allowed; {len(jobs)} more would be too many ('status' lists them)."
        )
    launches = rent_and_start(jobs, arguments)
    for launch in launches:
        start_copier(launch)
    cost = sum(launch.cost_per_hour for launch in launches)
    print(f"\nStarted {len(launches)} run(s) on Runpod, ${cost:.2f}/h in all:")
    for launch in launches:
        print(
            f"  {launch.name}  pod {launch.pod_id}, {launch.gpu}"
            f" ({launch.cloud.lower()}), ${launch.cost_per_hour:.2f}/h"
        )
    evaluated = sum(bool(job.session["evaluation"]) for job in jobs)
    print(f"  tests        {'skipped' if arguments.skip_tests else 'first'}")
    print(f"  code         {commit[:7]}")
    print(
        f"  evaluation   {evaluated} of {len(jobs)} runs, by the file's evaluation.toml"
    )
    print(
        f"Each pod is copied here and deleted when its session ends; as a backstop"
        f" it deletes itself {GRACE_HOURS} h later, or after {arguments.max_hours} h"
        " in all."
    )
    if arguments.detach:
        print("Follow them with 'watch'; 'status' shows the pods and their copies.")
        return 0
    print("Ctrl+C stops watching; the runs and their copying go on.\n")
    return watch_launches(launches)


def watch(arguments: argparse.Namespace) -> int:
    resume_copiers()
    return watch_launches(chosen_launches(arguments.pattern))


def fetch(arguments: argparse.Namespace) -> int:
    """Copy the chosen launches here, waiting while their sessions run; a copy
    that failed is tried again."""
    resume_copiers()
    launches = chosen_launches(arguments.pattern)
    for launch in launches:
        if copier_done(launch) and not any(
            COPIED in line for line in copier_lines(launch)
        ):
            with launch.copier_log.open("a", encoding="utf-8") as log:
                log.write(f"{now()} trying again ('fetch')\n")
            start_copier(launch)
    try:
        return wait_for_copiers(launches)
    except KeyboardInterrupt:
        print("\nStopped waiting; the runs and their copying go on.")
        return INTERRUPTED


def stop(arguments: argparse.Namespace) -> int:
    """Copy what the chosen launches' pods hold so far, then delete the pods."""
    failed = 0
    for launch in chosen_launches(arguments.pattern):
        if copier_done(launch):
            continue
        if runpod("GET", f"/pods/{launch.pod_id}") is None:
            print(f"{launch.name}: the pod no longer exists.")
            continue
        run_folders = copy_output(launch)
        if run_folders is None:
            print(f"{launch.name}: copying failed; the output so far is lost.")
            failed = 1
        else:
            print(f"{launch.name}: copied {', '.join(run_folders) or 'no run folder'}")
        delete_pod(launch.pod_id)
        print(f"{launch.name}: pod {launch.pod_id} deleted")
    return failed


def status(_: argparse.Namespace) -> int:
    resume_copiers()
    names = launch_names()[-RECENT_LAUNCHES_SHOWN:]
    print("Recent launches:" if names else "No launch yet.")
    for name in names:
        lines = copier_lines(Launch.load(name))
        print(f"  {name}\n      {lines[-1] if lines else 'no copier yet'}")
    pods = rented_pods()
    print(f"\nPods rented now: {len(pods) or 'none'}")
    for pod in pods:
        rented_at = datetime.fromisoformat(pod["createdAt"].replace("Z", "+00:00"))
        hours = (datetime.now(UTC) - rented_at).total_seconds() / 3600
        print(
            f"  {pod['id']}  {pod['name']}  {(pod.get('gpu') or {}).get('id', 'CPU')},"
            f" ${pod['cost']:.2f}/h, {pod['status'].lower()} for {hours:.1f} h"
            f" (about ${pod['cost'] * hours:.2f})"
        )
    if pods:
        print(f"  ${sum(pod['cost'] for pod in pods):.2f}/h in all")
    return 0


def copy(arguments: argparse.Namespace) -> int:
    return copy_launch(arguments.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    run_parser = commands.add_parser(
        "run", help="rent a pod per run, train and evaluate, watch, then copy here"
    )
    run_parser.add_argument(
        "configurations",
        type=Path,
        nargs="+",
        metavar="configuration",
        help=f"training TOML files, at most {MAX_PODS}, one pod each",
    )
    run_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help=f"train the one file with each seed, one pod each (at most {MAX_PODS})",
    )
    run_parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help=f"Runpod's GPU type id (default: {DEFAULT_GPU})",
    )
    run_parser.add_argument(
        "--cloud",
        choices=("COMMUNITY", "SECURE"),
        default=DEFAULT_CLOUD,
        help=f"Runpod's cloud tier (default: {DEFAULT_CLOUD})",
    )
    run_parser.add_argument(
        "--max-hours",
        type=int,
        default=DEFAULT_MAX_HOURS,
        help=f"hours until a pod deletes itself (default: {DEFAULT_MAX_HOURS})",
    )
    run_parser.add_argument(
        "--skip-tests", action="store_true", help="train without running the tests"
    )
    run_parser.add_argument(
        "--detach", action="store_true", help="start the runs without watching them"
    )
    run_parser.set_defaults(action=run)
    for name, action, description in (
        ("watch", watch, "follow a launch's sessions, then wait for their copies"),
        ("fetch", fetch, "wait until a launch's runs are copied here"),
        ("stop", stop, "copy what a launch's pods hold so far, then delete them"),
    ):
        chosen = commands.add_parser(name, help=description)
        chosen.add_argument(
            "pattern",
            nargs="?",
            help="part of the launches' names; default: the latest",
        )
        chosen.set_defaults(action=action)
    commands.add_parser(
        "status", help="recent launches and their copies, and the pods rented now"
    ).set_defaults(action=status)
    copy_parser = commands.add_parser(
        "copy", help="the copier of one launch, started by the other commands"
    )
    copy_parser.add_argument("name", help="the launch's full name")
    copy_parser.set_defaults(action=copy)
    arguments = parser.parse_args()
    try:
        sys.exit(arguments.action(arguments))
    except RunpodError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
