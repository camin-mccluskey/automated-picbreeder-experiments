"""Strategy subcommands for experiments/run_selection.py.

To add a strategy, write a configure function that adds its option groups and
sets build_strategy, then register it in COMMANDS. Keep model construction inside
the builder so parsing and help never load models.
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from .experiment import ExperimentSettings, run_experiment
from .experiment_batch import run_batch
from .selection_strategies import (
    ImageNetSelectionStrategy,
    NoveltyImageNetSelectionStrategy,
    NoveltyPredictabilitySelectionStrategy,
    NoveltySelectionStrategy,
    OffspringValueImageNetSelectionStrategy,
    OffspringValueSelectionStrategy,
    RandomSelectionStrategy,
)

ROOT = Path(__file__).resolve().parents[2]


def _integer_at_least(minimum):
    def parse(value):
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}")
        return number
    return parse


def _finite_float(minimum, maximum=None, *, exclusive_minimum=False):
    def parse(value):
        number = float(value)
        lower_valid = number > minimum if exclusive_minimum else number >= minimum
        if not math.isfinite(number) or not lower_valid or (maximum is not None and number > maximum):
            bounds = f"{'>' if exclusive_minimum else '>='} {minimum}"
            if maximum is not None:
                bounds += f" and <= {maximum}"
            raise argparse.ArgumentTypeError(f"must be finite and {bounds}")
        return number
    return parse


def _explicit_weights(value):
    if value == "DEFAULT":
        raise argparse.ArgumentTypeError("requires an explicit checkpoint version, e.g. IMAGENET1K_V1")
    return value


def _run_options(parser):
    defaults = ExperimentSettings()
    group = parser.add_argument_group("Run settings")
    group.add_argument("--steps", type=_integer_at_least(1), default=defaults.steps,
                       help=f"Decisions, with 9 candidates each (default: {defaults.steps}).")
    group.add_argument("--runs", type=_integer_at_least(1),
                       help="Run a batch of N independent runs using --seed + index (default: one standalone run).")
    group.add_argument("--seed", type=int, default=defaults.seed, help=f"Breeding seed (default: {defaults.seed}).")
    group.add_argument("--selection-seed", type=int,
                       help="Selection seed (default: derived from each run seed; explicit value increments in batches).")
    group.add_argument("--size", type=_integer_at_least(2), default=defaults.size,
                       help=f"Image side length (default: {defaults.size}).")
    group.add_argument("--mutation-strength", type=_finite_float(0), default=defaults.mutation_strength,
                       help=f"Mutation strength (default: {defaults.mutation_strength}).")
    group.add_argument("--no-topology", action="store_true", help="Freeze structure and activation functions.")
    group.add_argument("--checkpoint-every", type=_integer_at_least(1), default=defaults.checkpoint_every,
                       help=f"Decisions between audit snapshots (default: {defaults.checkpoint_every}).")
    group.add_argument("--output", type=Path, help="New output directory (default: timestamped directory in runs/).")


def _runtime_options(parser, *, devices=None):
    group = parser.add_argument_group("Runtime")
    group.add_argument("--device", choices=devices, default="cpu",
                       help="Model device (default: cpu)." if devices else
                            "Classifier device, e.g. cpu, mps or cuda (default: cpu).")
    group.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "imagenet",
                       help="Model checkpoint directory (default: repository .cache/imagenet).")


def _selection_options(parser, *, offspring=False):
    group = parser.add_argument_group("Selection")
    group.add_argument("--comprehension-weight", type=_finite_float(0, 1),
                       help="Quality rank weight in [0, 1] (default: 0.5).")
    if offspring:
        group.add_argument("--gamma", type=_finite_float(0),
                           help="Offspring forecast weight (default: 1; 0 is the current-image control).")


def _novelty_options(parser):
    group = parser.add_argument_group("Novelty reference")
    group.add_argument("--novelty-reference", choices=("previous-grid-mean", "previous-parent"),
                       help="Pixel distance reference (default: previous-grid-mean; previous-parent uses the preceding selected image).")


def _imagenet_options(parser):
    group = parser.add_argument_group("ImageNet classifier")
    group.add_argument("--imagenet-model", help="Torchvision ImageNet-1K model (default: resnet18).")
    group.add_argument("--imagenet-weights", type=_explicit_weights,
                       help="Explicit checkpoint version (default: IMAGENET1K_V1; DEFAULT is rejected).")
    group.add_argument("--imagenet-batch-size", type=_integer_at_least(1),
                       help="Inference batch size (default: 16).")


def _observer_options(parser, *, initializations=("random", "imagenet")):
    group = parser.add_argument_group("Comprehension observer")
    group.add_argument("--comprehension-warmup-steps", type=_integer_at_least(0),
                       help="Decisions before comprehension affects selection and targets (default: 10; 0 disables).")
    group.add_argument("--observer-initialization", choices=initializations,
                       help="Observer backbone initialization (default: random).")
    group.add_argument("--training-steps", type=_integer_at_least(1),
                       help="Observer optimizer updates per decision (default: 20).")
    group.add_argument("--observer-batch-size", type=_integer_at_least(2),
                       help="Observer training batch size, at least 2 (default: 16).")
    group.add_argument("--learning-rate", type=_finite_float(0, exclusive_minimum=True),
                       help="Observer Adam learning rate (default: 0.001).")


def _predictor_options(parser):
    group = parser.add_argument_group("Offspring predictor")
    group.add_argument("--offspring-aggregation", choices=("max", "mean"),
                       help="Predict the maximum or mean value of eight actual children (default: max).")
    group.add_argument("--warmup-targets", type=_integer_at_least(1),
                       help="Eligible targets before forecasts affect selection (default: 10).")
    group.add_argument("--predictor-training-steps", type=_integer_at_least(1),
                       help="Predictor optimizer updates per target (default: 10).")
    group.add_argument("--predictor-batch-size", type=_integer_at_least(1),
                       help="Predictor transition batch size (default: 16).")
    group.add_argument("--predictor-learning-rate", type=_finite_float(0, exclusive_minimum=True),
                       help="Predictor Adam learning rate (default: 0.001).")


def _provided(args, *names, **renamed):
    """Forward explicit values; leave strategy/evaluator defaults in their constructors."""
    mapping = {name: name for name in names} | renamed
    return {target: getattr(args, source) for target, source in mapping.items()
            if getattr(args, source) is not None}


def _observer_kwargs(args):
    return _provided(args, "novelty_reference", "comprehension_weight", "comprehension_warmup_steps",
                     "observer_initialization", "training_steps", "learning_rate", "device", "cache_dir",
                     batch_size="observer_batch_size")


def _predictor_kwargs(args):
    return _provided(args, "offspring_aggregation", "gamma", "warmup_targets", "predictor_training_steps",
                     "predictor_batch_size", "predictor_learning_rate")


def _build_evaluator(args):
    from .imagenet import ImageNetEvaluator

    return ImageNetEvaluator(**_provided(args, "device", "cache_dir", model_name="imagenet_model",
                                        weights="imagenet_weights", batch_size="imagenet_batch_size"))


def _random(parser):
    parser.set_defaults(build_strategy=lambda args: RandomSelectionStrategy())


def _novelty(parser):
    _novelty_options(parser)
    parser.set_defaults(build_strategy=lambda args: NoveltySelectionStrategy(**_provided(args, "novelty_reference")))


def _imagenet(parser):
    selection = parser.add_argument_group("Selection")
    selection.add_argument("--epsilon", type=_finite_float(0, 1), default=0.,
                           help="Random-choice probability in [0, 1] (default: 0, greedy).")
    _imagenet_options(parser)
    _runtime_options(parser)
    parser.set_defaults(build_strategy=lambda args: ImageNetSelectionStrategy(
        epsilon=args.epsilon, evaluator=_build_evaluator(args)))


def _novelty_imagenet(parser):
    _novelty_options(parser)
    _selection_options(parser)
    _imagenet_options(parser)
    _runtime_options(parser)
    parser.set_defaults(build_strategy=lambda args: NoveltyImageNetSelectionStrategy(
        **_provided(args, "novelty_reference", "comprehension_weight"), evaluator=_build_evaluator(args)))


def _novelty_predictability(parser):
    _novelty_options(parser)
    _selection_options(parser)
    _observer_options(parser)
    _runtime_options(parser, devices=("cpu", "mps"))
    parser.set_defaults(build_strategy=lambda args: NoveltyPredictabilitySelectionStrategy(**_observer_kwargs(args)))


def _offspring_value(parser):
    _novelty_options(parser)
    _selection_options(parser, offspring=True)
    _observer_options(parser, initializations=("random",))
    _predictor_options(parser)
    _runtime_options(parser, devices=("cpu", "mps"))
    parser.set_defaults(build_strategy=lambda args: OffspringValueSelectionStrategy(
        **_observer_kwargs(args), **_predictor_kwargs(args)))


def _offspring_value_imagenet(parser):
    _novelty_options(parser)
    _selection_options(parser, offspring=True)
    _imagenet_options(parser)
    _predictor_options(parser)
    _runtime_options(parser, devices=("cpu", "mps"))
    parser.set_defaults(build_strategy=lambda args: OffspringValueImageNetSelectionStrategy(
        **_provided(args, "novelty_reference", "comprehension_weight", "device"), **_predictor_kwargs(args),
        evaluator=_build_evaluator(args)))


COMMANDS = (
    ("random", "Choose uniformly without image evaluation.", _random),
    ("novelty", "Maximize pixel novelty relative to a previous grid mean or selected parent.", _novelty),
    ("novelty-imagenet", "Combine pixel novelty and ImageNet confidence.", _novelty_imagenet),
    ("novelty-predictability", "Combine pixel novelty and learned patch predictability.", _novelty_predictability),
    ("offspring-value", "Add offspring forecasts to novelty and patch predictability.", _offspring_value),
    ("offspring-value-imagenet", "Add offspring forecasts to novelty and ImageNet confidence.", _offspring_value_imagenet),
    ("imagenet", "Maximize ImageNet confidence with optional random exploration.", _imagenet),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a selection strategy in the shared nine-image breeding loop.",
        epilog="Use STRATEGY --help for its options. Put all run options after the strategy name.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="selection_strategy", required=True, title="Strategies", metavar="STRATEGY")
    for name, description, configure in COMMANDS:
        command = subparsers.add_parser(name, help=description, description=description, allow_abbrev=False)
        _run_options(command)
        configure(command)
        command.set_defaults(command_parser=command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    parser = args.command_parser
    try:
        settings = ExperimentSettings(seed=args.seed, selection_seed=args.selection_seed,
                                      steps=args.steps, size=args.size, mutation_strength=args.mutation_strength,
                                      topology=not args.no_topology, checkpoint_every=args.checkpoint_every)
    except ValueError as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    kind = "batch" if args.runs is not None else "selection"
    output = args.output or ROOT / "runs" / f"{args.selection_strategy}-{kind}-{stamp}-seed{args.seed}"
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    if args.runs is not None:
        manifest = run_batch(strategy_factory=lambda: args.build_strategy(args), output_dir=output,
                             runs=args.runs, settings=settings)
        print(f"Saved {manifest['completed_runs']}/{args.runs} completed runs to {output.resolve()}")
        if manifest["failed_runs"]:
            raise SystemExit(1)
        return
    try:
        strategy = args.build_strategy(args)
    except ValueError as exc:
        parser.error(str(exc))
    summary = run_experiment(selection_strategy=strategy, output_dir=output, settings=settings)
    print(f"Saved {summary['decisions']} decisions to {output.resolve()}")
