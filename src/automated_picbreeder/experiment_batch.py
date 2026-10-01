"""Sequential independent runs of one strategy configuration."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .experiment import ExperimentSettings, run_experiment
from .experiment_metrics import save_run_metrics, summarize_batch, write_csv
from .persistence import write_json


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save_batch(directory, manifest, reports):
    manifest["completed_runs"] = sum(r["status"] == "complete" for r in manifest["runs"])
    manifest["failed_runs"] = sum(r["status"] == "failed" for r in manifest["runs"])
    write_json(directory / "batch.json", manifest)
    aggregate = summarize_batch(reports)
    aggregate.update(requested_runs=len(manifest["runs"]), failed_runs=manifest["failed_runs"],
                     pending_runs=sum(r["status"] == "pending" for r in manifest["runs"]),
                     batch_status=manifest["status"])
    write_json(directory / "aggregate.json", aggregate)
    summaries = {report["seed"]: report["summary"] for report in reports}
    rows = []
    for run in manifest["runs"]:
        row = {key: run.get(key) for key in ("index", "seed", "selection_seed", "status", "directory",
                                            "metrics", "final_image", "setup_seconds", "wall_seconds")}
        row["error_type"] = run.get("error", {}).get("type")
        row["error_message"] = run.get("error", {}).get("message")
        summary = summaries.get(run["seed"], {})
        for key in ("decisions", "candidate_presentations", "generated_genomes", "classifier_images", "run_seconds", "seconds_per_decision",
                    "displayed_grids", "selection_revisions", "backtracks", "resets", "interaction_seconds",
                    "posthoc_classifier_images", "posthoc_evaluation_seconds"):
            row[key] = summary.get(key)
        for key, stats in summary.get("metrics", {}).items():
            for statistic in ("mean", "final", "max", "total"):
                if statistic in stats:
                    row[f"{key}.{statistic}"] = stats[statistic]
        for group, stats in (summary.get("offspring_prediction") or {}).items():
            row.update({f"offspring_prediction.{group}.{key}": value for key, value in stats.items()})
        rows.append(row)
    write_csv(directory / "runs.csv", rows)
    from .experiment_viewer import write_viewer

    by_seed = {report["seed"]: report for report in reports}
    prepared = {directory / entry["directory"]: by_seed[entry["seed"]]
                for entry in manifest["runs"] if entry["seed"] in by_seed}
    write_viewer(directory, prepared_reports=prepared, prepared_aggregates={directory: aggregate})


def run_batch(*, strategy_factory, output_dir, runs, settings=ExperimentSettings(), progress=print):
    """Run seed+i using a newly constructed strategy each time.

    Explicit selection_seed is also incremented; otherwise use each run's usual
    derived selection seed. Factories must return fresh instances with identical
    describe() configuration. Ordinary run failures are logged and later seeds
    still run. Interruptions are logged and propagated. Never overwrite outputs.
    """
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    batch_start = time.perf_counter()
    run_settings = [replace(settings, seed=settings.seed + index,
                            selection_seed=None if settings.selection_seed is None else settings.selection_seed + index)
                    for index in range(runs)]
    manifest = {
        "schema_version": 1, "status": "running", "started_at": _now(),
        "base_settings": asdict(settings), "selection_strategy": None,
        "seed_schedule": "seed = base_seed + index; explicit selection_seed = base_selection_seed + index, otherwise derived independently per run",
        "runs": [{"index": i, "seed": s.seed, "selection_seed": s.resolved_selection_seed,
                  "directory": f"run-{i:04d}-seed{s.seed}", "status": "pending"} for i, s in enumerate(run_settings)],
    }
    reports = []
    _save_batch(directory, manifest, reports)
    for entry, run_settings_item in zip(manifest["runs"], run_settings, strict=True):
        entry.update(status="running", started_at=_now())
        _save_batch(directory, manifest, reports)
        if progress is not None:
            progress(f"Run {entry['index'] + 1}/{runs}: seed {entry['seed']}")
        run_dir = directory / entry["directory"]
        start = time.perf_counter()
        interrupted = False
        strategy = None
        try:
            strategy = strategy_factory()
            description = strategy.describe()
            if manifest["selection_strategy"] is None:
                manifest["selection_strategy"] = description
            elif description != manifest["selection_strategy"]:
                raise ValueError("Every batch run must use the same strategy configuration.")
            entry["setup_seconds"] = time.perf_counter() - start
            run_experiment(selection_strategy=strategy, output_dir=run_dir, settings=run_settings_item, progress=progress)
            report = json.loads((run_dir / "metrics.json").read_text())
            entry["status"] = "complete"
        except (Exception, KeyboardInterrupt) as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            entry["status"] = "interrupted" if interrupted else "failed"
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report = None
            if (run_dir / "session.json").exists():
                try:
                    report = save_run_metrics(run_dir, status=entry["status"])
                except Exception as reporting_error:
                    entry["metrics_error"] = {"type": type(reporting_error).__name__, "message": str(reporting_error)}
            if progress is not None:
                progress(f"Run seed {entry['seed']} {entry['status']}: {type(exc).__name__}: {exc}")
        finally:
            # Do not keep prior observers/predictors alive between independent runs.
            del strategy
        if report is not None:
            reports.append(report)
            entry["metrics"] = f"{entry['directory']}/metrics.json"
            entry["final_image"] = f"{entry['directory']}/{report['final_image']}" if report["final_image"] else None
            entry["decisions"] = report["summary"]["decisions"]
        entry.update(finished_at=_now(), wall_seconds=time.perf_counter() - start)
        manifest["elapsed_seconds"] = time.perf_counter() - batch_start
        if interrupted:
            manifest.update(status="interrupted", finished_at=_now())
        _save_batch(directory, manifest, reports)
        if interrupted:
            raise KeyboardInterrupt()
    manifest.update(status="completed_with_failures" if manifest["failed_runs"] else "complete",
                    finished_at=_now(), elapsed_seconds=time.perf_counter() - batch_start)
    _save_batch(directory, manifest, reports)
    return manifest
