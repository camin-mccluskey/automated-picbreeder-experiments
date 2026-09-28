"""Run the three Experiment 1 conditions with matched seed sets and budgets."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from automated_picbreeder.selection_comparison import run_comparison, write_comparison_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New comparison directory (generated under runs/ if omitted).')
    parser.add_argument('--report-only', type=Path, help='Rebuild reports for an existing comparison; do not run evolution.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 8, 9])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--size', type=int, default=96)
    parser.add_argument('--mutation-strength', type=float, default=.2)
    parser.add_argument('--no-topology', action='store_true')
    parser.add_argument('--comprehension-weight', type=float, default=.5)
    parser.add_argument('--training-steps', type=int, default=20)
    parser.add_argument('--observer-batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu', help='Observer computation device; initialization and sampling stay on CPU.')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.report_only:
        report = write_comparison_report(args.report_only)
        print(f"Report {report['status']}: {args.report_only / 'REPORT.md'}")
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    output = args.output or root / 'runs' / f'comparison-{stamp}'
    try:
        report = run_comparison(output_dir=output, seeds=args.seeds, steps=args.steps, size=args.size,
            mutation_strength=args.mutation_strength, topology=not args.no_topology,
            comprehension_weight=args.comprehension_weight, training_steps=args.training_steps,
            batch_size=args.observer_batch_size, learning_rate=args.learning_rate, cache_dir=root / '.cache/imagenet', device=args.device)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Comparison {report['status']}: {output.resolve() / 'REPORT.md'}")
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
