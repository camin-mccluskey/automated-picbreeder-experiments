"""Run selection strategies in the shared breeding loop.

Examples from the project root:
    uv run python experiments/run_selection.py --selection-strategy random
    uv run python experiments/run_selection.py --selection-strategy novelty
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy novelty-imagenet --comprehension-weight 0.5
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy novelty-predictability --observer-initialization random
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy offspring-value --gamma 1
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy offspring-value-imagenet --gamma 1
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet
    uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet --epsilon 0.1
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.selection_strategies import (
    ImageNetSelectionStrategy,
    NoveltyImageNetSelectionStrategy,
    NoveltyPredictabilitySelectionStrategy,
    NoveltySelectionStrategy,
    OffspringValueSelectionStrategy,
    OffspringValueImageNetSelectionStrategy,
    RandomSelectionStrategy,
)


def main(argv=None):
    defaults = ExperimentSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-strategy", choices=("random", "novelty", "novelty-imagenet", "novelty-predictability", "offspring-value", "offspring-value-imagenet", "imagenet"), required=True,
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
    parser.add_argument("--device", help="Model device (default: cpu; observers/predictors support cpu/mps; ImageNet also supports cuda).")
    parser.add_argument("--cache-dir", type=Path,
                        help="Model checkpoint directory for ImageNet and observer strategies (default: .cache/imagenet in the repository).")
    parser.add_argument("--imagenet-model", help="Torchvision ImageNet-1K model name (default: resnet18).")
    parser.add_argument("--imagenet-weights", help="Explicit checkpoint version for the chosen model (default: IMAGENET1K_V1; DEFAULT is rejected).")
    parser.add_argument("--imagenet-batch-size", type=int, help="ImageNet inference batch size >= 1 (default: 16).")
    parser.add_argument("--comprehension-weight", type=float, help="Quality rank weight for combined novelty strategies, in [0,1] (default: 0.5).")
    parser.add_argument("--comprehension-warmup-steps", type=int,
                        help="Completed decisions before comprehension affects selection and offspring targets become eligible (default: 10; 0 disables).")
    parser.add_argument("--observer-initialization", choices=("random", "imagenet"), help="Observer backbone initialization (default: random; offspring-value requires random).")
    parser.add_argument("--training-steps", type=int, help="Observer optimizer updates per decision (default: 20).")
    parser.add_argument("--observer-batch-size", type=int, help="Observer training batch size >= 2 (default: 16).")
    parser.add_argument("--learning-rate", type=float, help="Observer Adam learning rate (default: 0.001).")
    parser.add_argument("--gamma", type=float, help="Offspring forecast weight >= 0 (default: 1; 0 is the current-image control).")
    parser.add_argument("--warmup-targets", type=int, help="Completed eligible targets before offspring forecasts affect selection (default: 10; minimum: 1; patch strategy also has comprehension warmup).")
    parser.add_argument("--predictor-training-steps", type=int, help="Scalar predictor updates per target (default: 10).")
    parser.add_argument("--predictor-batch-size", type=int, help="Scalar predictor transition batch size (default: 16).")
    parser.add_argument("--predictor-learning-rate", type=float, help="Scalar predictor Adam learning rate (default: 0.001).")
    parser.add_argument("--output", type=Path, help="A new output directory; existing directories are never overwritten.")
    args = parser.parse_args(argv)
    imagenet_strategies = ("imagenet", "novelty-imagenet", "offspring-value-imagenet")
    observer_strategies = ("novelty-predictability", "offspring-value")
    offspring_strategies = ("offspring-value", "offspring-value-imagenet")
    model_strategies = imagenet_strategies + observer_strategies
    if args.selection_strategy != "imagenet" and args.epsilon is not None:
        parser.error("--epsilon applies only to --selection-strategy imagenet.")
    if args.selection_strategy not in model_strategies and args.device is not None:
        parser.error("--device applies only to strategies using a model.")
    if args.selection_strategy not in model_strategies and args.cache_dir is not None:
        parser.error("--cache-dir applies only to strategies using a model.")
    if args.selection_strategy in offspring_strategies and args.device not in (None, "cpu", "mps"):
        parser.error("Offspring predictor device must be cpu or mps.")
    imagenet_options = {
        "model_name": args.imagenet_model,
        "weights": args.imagenet_weights,
        "batch_size": args.imagenet_batch_size,
    }
    imagenet_options = {key: value for key, value in imagenet_options.items() if value is not None}
    if imagenet_options and args.selection_strategy not in imagenet_strategies:
        parser.error("ImageNet model, weights and batch-size options apply only to imagenet, novelty-imagenet or offspring-value-imagenet.")
    if args.imagenet_batch_size is not None and args.imagenet_batch_size < 1:
        parser.error("--imagenet-batch-size must be >= 1.")
    if args.imagenet_weights == "DEFAULT":
        parser.error("--imagenet-weights requires an explicit checkpoint version, e.g. IMAGENET1K_V1.")
    comprehension_options = {}
    if args.comprehension_weight is not None:
        if args.selection_strategy not in ("novelty-imagenet", "offspring-value-imagenet") + observer_strategies:
            parser.error("--comprehension-weight applies only to combined novelty strategies.")
        if not 0 <= args.comprehension_weight <= 1:
            parser.error("--comprehension-weight must be a finite number in [0, 1].")
        comprehension_options["comprehension_weight"] = args.comprehension_weight
    observer_options = {
        "comprehension_warmup_steps": args.comprehension_warmup_steps,
        "observer_initialization": args.observer_initialization,
        "training_steps": args.training_steps, "batch_size": args.observer_batch_size, "learning_rate": args.learning_rate,
    }
    observer_options = {key: value for key, value in observer_options.items() if value is not None}
    if observer_options and args.selection_strategy not in observer_strategies:
        parser.error("Observer options apply only to novelty-predictability or offspring-value.")
    predictor_options = {key: getattr(args, key) for key in (
        "gamma", "warmup_targets", "predictor_training_steps", "predictor_batch_size", "predictor_learning_rate")
        if getattr(args, key) is not None}
    if predictor_options and args.selection_strategy not in offspring_strategies:
        parser.error("Predictor options apply only to offspring-value or offspring-value-imagenet.")
    for name, value in predictor_options.items():
        if name == "gamma":
            valid = math.isfinite(value) and value >= 0
        elif name == "predictor_learning_rate":
            valid = math.isfinite(value) and value > 0
        else:
            valid = value >= 1
        if not valid:
            parser.error(f"Invalid --{name.replace('_', '-')}: {value}.")
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
    cache_dir = args.cache_dir if args.cache_dir is not None else root / ".cache" / "imagenet"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    output = args.output or root / "runs" / f"{args.selection_strategy}-selection-{stamp}-seed{args.seed}"
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    if args.selection_strategy == "random":
        selection_strategy = RandomSelectionStrategy()
    elif args.selection_strategy == "novelty":
        selection_strategy = NoveltySelectionStrategy()
    elif args.selection_strategy in observer_strategies:
        try:
            strategy_type = OffspringValueSelectionStrategy if args.selection_strategy == "offspring-value" else NoveltyPredictabilitySelectionStrategy
            selection_strategy = strategy_type(**comprehension_options, **observer_options, **predictor_options, device=args.device or "cpu", cache_dir=cache_dir)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        from automated_picbreeder.imagenet import ImageNetEvaluator

        try:
            evaluator = ImageNetEvaluator(**imagenet_options, device=args.device or "cpu", cache_dir=cache_dir)
        except ValueError as exc:
            parser.error(str(exc))
        if args.selection_strategy == "novelty-imagenet":
            selection_strategy = NoveltyImageNetSelectionStrategy(**comprehension_options, evaluator=evaluator)
        elif args.selection_strategy == "offspring-value-imagenet":
            try:
                selection_strategy = OffspringValueImageNetSelectionStrategy(
                    **comprehension_options, **predictor_options, evaluator=evaluator, device=args.device or "cpu")
            except ValueError as exc:
                parser.error(str(exc))
        else:
            selection_strategy = ImageNetSelectionStrategy(epsilon=epsilon, evaluator=evaluator)
    summary = run_experiment(selection_strategy=selection_strategy, output_dir=output, settings=settings)
    print(f"Saved {summary['decisions']} decisions to {output.resolve()}")


if __name__ == "__main__":
    main()
