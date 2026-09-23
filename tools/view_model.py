"""Open a Centipede MJCF model in MuJoCo's interactive viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as error:
    if error.name != "mujoco":
        raise
    raise SystemExit(
        "MuJoCo is not installed in this Python environment.\n"
        "Run the viewer with the Centipede environment:\n"
        '  PowerShell: & "$HOME\\.venvs\\Centipede\\Scripts\\python.exe" '
        ".\\tools\\view_model.py\n"
        "  WSL: ~/.venvs/Centipede/bin/python tools/view_model.py"
    ) from None

# Resolve the default model relative to the project rather than the caller's shell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "assembly.xml"
# The viewer is synchronized at a modest fixed rate because physics is paused.
VIEWER_FPS = 60


def parse_args() -> argparse.Namespace:
    """Read the optional model path used for interactive inspection."""
    parser = argparse.ArgumentParser(
        description="Open a Centipede XML model in MuJoCo's interactive viewer."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help="XML model to open (default: models/assembly.xml)",
    )
    return parser.parse_args()


def main() -> None:
    """Load the selected XML and keep a centered passive viewer responsive."""
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"Opening {model_path}")
    print(f"Model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(
        "Static inspection mode: drag the camera with the mouse; "
        "close the window to exit."
    )

    # Passive mode gives this loop explicit ownership of camera setup and refresh.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 1.6 * model.stat.extent
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45

        frame_period = 1 / VIEWER_FPS
        while viewer.is_running():
            viewer.sync()
            time.sleep(frame_period)


if __name__ == "__main__":
    main()
