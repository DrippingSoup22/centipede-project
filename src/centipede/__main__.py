"""The command line: ``python -m centipede CONFIG.toml``.

Trains or evaluates as the configuration file's ``[run] mode`` says. A problem
in the file is reported in one line, without a traceback.
"""

import argparse
import sys
from pathlib import Path

from centipede.experiment.experiment import run
from centipede.settings_section import SettingsError


def main() -> None:
    """Read the command's one argument and run it."""
    parser = argparse.ArgumentParser(
        prog="python -m centipede",
        description="Train or evaluate centipede agents from one TOML file.",
    )
    parser.add_argument("configuration", type=Path, help="the run's TOML file")
    arguments = parser.parse_args()
    try:
        run(arguments.configuration)
    except SettingsError as error:
        sys.exit(f"Configuration error: {error}")


if __name__ == "__main__":
    main()
