"""Run selection strategies in the shared breeding loop.

Examples from the project root:
    uv run python experiments/run_selection.py --selection-strategy random
    uv run python experiments/run_selection.py --selection-strategy novelty
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy novelty-predictability --observer-initialization random
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet --epsilon 0.1
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy, OffspringValueSelectionStrategy, NoveltySelectionStrategy, NoveltyPredictabilitySelectionStrategy, RandomSelectionStrategy,
)


def main(argv=None):
    defaults = ExperimentSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-strategy", choices=("random", "novelty", "novelty-predictability", "offspring-value", "imagenet"), required=True,
                        help="How to choose the next image; random and novelty need no classifier.")
    parser.add_argument("--epsilon", type=float,
                        help="ImageNet random-choice probability, in [0, 1] (default: 0, greedy).")
    parser.add_argument("--steps", type=int, default=defaults.steps,
                        help=f"Decisions including the initial grid (default: {defaults.steps}); 9 candidates per decision.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help=f"Breeding seed (default: {defaults.seed}).")
    parser.add_argument("--selection-seed", type=int, help="Selection seed (default: derived separately from --seed).")
    parser.add_argument("--size", type=int, default=defaults.size, help=f"Image side length (default: {defaults.size}).")
    parser.add_argument("--mutation-strength", type=float, default=defaults.mutation_strength,
                        help=f"Mutation strength (default: {defaults.mutation_strength}).")
    parser.add_argument("--no-topology", action="store_true", help="Freeze structure and activation functions.")
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every,
                        help=f"Decisions between audit snapshots (default: {defaults.checkpoint_every}).")
    parser.add_argument("--device", help="ImageNet or predictability observer device (default: cpu; observer supports cpu/mps).")
    parser.add_argument("--comprehension-weight", type=float, help="Novelty-predictability rank weight in [0,1] (default: 0.5).")
    parser.add_argument("--observer-initialization", choices=("random", "imagenet"), help="Observer backbone initialization (default: random).")
    parser.add_argument("--training-steps", type=int, help="Observer optimizer updates per decision (default: 20).")
    parser.add_argument("--observer-batch-size", type=int, help="Observer training batch size >= 2 (default: 16).")
    parser.add_argument("--learning-rate", type=float, help="Observer Adam learning rate (default: 0.001).")
    parser.add_argument("--gamma", type=float, help="Offspring forecast weight >= 0 (default: 1; 0 is the current-image control).")
    parser.add_argument("--warmup-targets", type=int, help="Completed targets before forecasts affect selection (default: 10).")
    parser.add_argument("--predictor-training-steps", type=int, help="Scalar predictor updates per target (default: 10).")
    parser.add_argument("--predictor-batch-size", type=int, help="Scalar predictor transition batch size (default: 16).")
    parser.add_argument("--predictor-learning-rate", type=float, help="Scalar predictor Adam learning rate (default: 0.001).")
    parser.add_argument("--output", type=Path, help="A new output directory; existing directories are never overwritten.")
    args = parser.parse_args(argv)
    if args.selection_strategy != "imagenet" and args.epsilon is not None:
        parser.error("--epsilon applies only to --selection-strategy imagenet.")
    if args.selection_strategy not in ("imagenet", "novelty-predictability", "offspring-value") and args.device is not None:
        parser.error("--device applies only to imagenet, novelty-predictability or offspring-value.")
    observer_options = {
        "comprehension_weight": args.comprehension_weight, "observer_initialization": args.observer_initialization,
        "training_steps": args.training_steps, "batch_size": args.observer_batch_size, "learning_rate": args.learning_rate,
    }
    observer_options = {key: value for key, value in observer_options.items() if value is not None}
    if observer_options and args.selection_strategy not in ("novelty-predictability", "offspring-value"):
        parser.error("Observer options apply only to novelty-predictability or offspring-value.")
    predictor_options = {key: getattr(args, key) for key in (
        "gamma", "warmup_targets", "predictor_training_steps", "predictor_batch_size", "predictor_learning_rate")
        if getattr(args, key) is not None}
    if predictor_options and args.selection_strategy != "offspring-value":
        parser.error("Predictor options apply only to --selection-strategy offspring-value.")
    epsilon = 0.0 if args.epsilon is None else args.epsilon
    if not 0 <= epsilon <= 1:
        parser.error("--epsilon must be a finite number in [0, 1].")
    try:
        settings = ExperimentSettings(seed=args.seed, selection_seed=args.selection_seed,
                                      steps=args.steps, size=args.size, mutation_strength=args.mutation_strength,
                                      topology=not args.no_topology, checkpoint_every=args.checkpoint_every)
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output = args.output or root / "runs" / f"{args.selection_strategy}-selection-{stamp}-seed{args.seed}"
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    if args.selection_strategy == "random":
        selection_strategy = RandomSelectionStrategy()
    elif args.selection_strategy == "novelty":
        selection_strategy = NoveltySelectionStrategy()
    elif args.selection_strategy in ("novelty-predictability", "offspring-value"):
        try:
            strategy_type = OffspringValueSelectionStrategy if args.selection_strategy == "offspring-value" else NoveltyPredictabilitySelectionStrategy
            selection_strategy = strategy_type(**observer_options, **predictor_options, device=args.device or "cpu", cache_dir=root / ".cache" / "imagenet")
        except ValueError as exc:
            parser.error(str(exc))
    else:
        from automated_picbreeder.imagenet import ImageNetEvaluator

        evaluator = ImageNetEvaluator(device=args.device or "cpu", cache_dir=root / ".cache" / "imagenet")
        selection_strategy = ImageNetSelectionStrategy(epsilon=epsilon, evaluator=evaluator)
    summary = run_experiment(selection_strategy=selection_strategy, output_dir=output, settings=settings)
    print(f"Saved {summary['decisions']} decisions to {output.resolve()}")


if __name__ == "__main__":
    main()
