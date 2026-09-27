"""Automate the human selection using maximum ImageNet confidence.

Run from the project root:
    uv run --extra imagenet python experiments/imagenet_selection.py --steps 100
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.imagenet import ImageNetEvaluator
from automated_picbreeder.selection import MaximumClassConfidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50, help="Selection decisions, including the initial grid; 9 image evaluations each.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--mutation-strength", type=float, default=0.2)
    parser.add_argument("--no-topology", action="store_true", help="Freeze structure and activation functions, as in the human UI.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="A new output directory; existing directories are never overwritten.")
    args = parser.parse_args()
    try:
        settings = ExperimentSettings(seed=args.seed, steps=args.steps, size=args.size,
                                      mutation_strength=args.mutation_strength, topology=not args.no_topology,
                                      checkpoint_every=args.checkpoint_every)
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output = args.output or root / "runs" / f"imagenet-selection-{stamp}-seed{args.seed}"
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    evaluator = ImageNetEvaluator(device=args.device, batch_size=9, cache_dir=root / ".cache" / "imagenet")
    summary = run_experiment(evaluator, MaximumClassConfidence(), output, settings)
    print(f"Saved {summary['decisions']} decisions to {output.resolve()}")


if __name__ == "__main__":
    main()
