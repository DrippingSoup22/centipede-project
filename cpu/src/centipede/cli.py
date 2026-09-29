"""Run one Centipede plan file.

Every setting lives in the TOML plan; its top-level kind selects training,
evaluation of an existing run, or a comparison of training variants.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from centipede.experiment import workflows
from centipede.experiment.presets import (
    ComparisonPlan,
    EvaluationPlan,
    TrainingPlan,
    load_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the plan file, then pass it to the workflow for its kind."""
    parser = argparse.ArgumentParser(prog="centipede-cpu", description=__doc__)
    parser.add_argument("plan", type=Path, help="experiment plan (TOML file)")
    plan = load_plan(parser.parse_args(argv).plan)

    match plan:
        case TrainingPlan():
            workflows.train(plan)
        case EvaluationPlan():
            workflows.evaluate(plan)
        case ComparisonPlan():
            workflows.compare(plan)
    return 0
