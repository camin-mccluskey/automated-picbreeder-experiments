"""Compare current-image selection with online offspring-value selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from automated_picbreeder.offspring_comparison import run_offspring_comparison, write_offspring_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    outputs=parser.add_mutually_exclusive_group()
    outputs.add_argument('--output',type=Path,help='New comparison directory; existing runs are never overwritten.')
    outputs.add_argument('--report-only',type=Path,help='Rebuild an existing report without training or evolution.')
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,8,9])
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--size',type=int,default=96)
    parser.add_argument('--mutation-strength',type=float,default=.2)
    parser.add_argument('--no-topology',action='store_true')
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--comprehension-weight',type=float,default=.5)
    parser.add_argument('--training-steps',type=int,default=20)
    parser.add_argument('--observer-batch-size',type=int,default=16)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--warmup-targets',type=int,default=10)
    parser.add_argument('--predictor-training-steps',type=int,default=10)
    parser.add_argument('--predictor-batch-size',type=int,default=16)
    parser.add_argument('--predictor-learning-rate',type=float,default=.001)
    args=parser.parse_args(argv)
    if args.report_only:
        report=write_offspring_report(args.report_only)
        output=args.report_only
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
        output=args.output or Path(__file__).resolve().parents[1]/'runs'/f'offspring-comparison-{stamp}'
        try:
            report=run_offspring_comparison(output_dir=output,seeds=args.seeds,steps=args.steps,size=args.size,
                mutation_strength=args.mutation_strength,topology=not args.no_topology,device=args.device,
                comprehension_weight=args.comprehension_weight,training_steps=args.training_steps,
                batch_size=args.observer_batch_size,learning_rate=args.learning_rate,warmup_targets=args.warmup_targets,
                predictor_training_steps=args.predictor_training_steps,predictor_batch_size=args.predictor_batch_size,
                predictor_learning_rate=args.predictor_learning_rate)
        except (ValueError,FileExistsError) as exc:
            parser.error(str(exc))
    print(f"Comparison {report['status']}: {output.resolve()/'REPORT.md'}")
    if report['status']!='complete': raise SystemExit(1)


if __name__=='__main__':
    main()
