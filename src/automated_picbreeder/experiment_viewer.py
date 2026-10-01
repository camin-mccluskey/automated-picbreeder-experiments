"""Build cached, offline overview and per-run inspectors from saved experiments."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote
import webbrowser

from .experiment_metrics import image_inventory
from .viewer_cache import atomic_text, batch_aggregate, load_report


ASSETS = Path(__file__).with_name("viewer")


def _discover(path):
    if not path.is_dir():
        raise FileNotFoundError(f"Run, batch or collection directory not found: {path}")
    if (path / "batch.json").is_file() or (path / "session.json").is_file():
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.is_dir() and not child.is_symlink() and not child.name.startswith(".") and child.name not in {"source", "images", "grids", "checkpoints"}:
            yield from _discover(child)


def _url(directory, relative, output):
    target = (directory / relative).resolve()
    if not target.is_relative_to(directory):
        raise ValueError(f"Artifact path points outside the run: {relative}")
    return quote(os.path.relpath(target, output.parent), safe="/")


def _run(directory, output, entry, refresh, prepared=None):
    run = {"label": directory.name, "status": entry.get("status", "incomplete"),
           "seed": entry.get("seed"), "report": None, "images": [], "files": {},
           "error": None, "setup_seconds": entry.get("setup_seconds"), "wall_seconds": entry.get("wall_seconds")}
    if entry.get("error"):
        run["error"] = f"{entry['error']['type']}: {entry['error']['message']}"
    if not (directory / "session.json").exists():
        return run
    try:
        run["files"] = {name: _url(directory, name, output) for name in
                        ("session.json", "metrics.json", "metrics.csv", "posthoc_imagenet.json", "selection_failure.json")
                        if (directory / name).is_file()}
        report = prepared if prepared is not None else load_report(directory, refresh=refresh)
        run["images"] = [item | {"url": _url(directory, item["path"], output)}
                         for item in report["image_inventory"]]
        report = {key: value for key, value in report.items() if key != "image_inventory"}
        if entry.get("status"):
            report["status"] = entry["status"]
        run.update(report=report, status=report["status"], seed=report["seed"])
    except (ValueError, KeyError, OSError, TypeError) as exc:
        message = f"Cannot read saved results: {exc}"
        run["error"] = f"{run['error']}; {message}" if run["error"] else message
        # Preserve image inspection even when diagnostics cannot be extracted.
        try:
            data = json.loads((directory / "session.json").read_text())
            run["images"] = [item | {"url": _url(directory, item["path"], output)}
                             for item in image_inventory(data)]
        except (ValueError, KeyError, OSError, TypeError):
            pass
    return run


def _run_task(arguments):
    return _run(*arguments)


def _write_page(payload, output):
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    css = (ASSETS / "report.css").read_text()
    javascript = (ASSETS / "report.js").read_text()
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(payload['title'])} · Picbreeder results</title><style>{css}</style></head>
<body><header class="masthead"><a href="#" id="home">PICBREEDER EXPERIMENTATION<span>RESEARCH RECORDS</span></a><span class="offline">LOCAL REPORT · OFFLINE</span></header>
<main id="app"></main><dialog id="image-dialog"><button id="close-dialog" aria-label="Close image">Close ×</button><div id="dialog-content"></div></dialog>
<noscript>This report needs JavaScript enabled. The saved JSON, CSV and PNG files remain available alongside it.</noscript>
<script id="report-data" type="application/json">{encoded}</script><script>{javascript}</script></body></html>'''
    if not output.exists() or output.read_text() != html:
        atomic_text(output, html)


def _overview_run(run, directory, detail, output, *, include_series):
    report = run["report"]
    reduced = {key: value for key, value in run.items() if key not in {"report", "images", "files"}}
    reduced.update(viewer_url=quote(os.path.relpath(detail, output.parent), safe="/"),
                   image_count=len(run["images"]), images=[], report=None)
    if report is not None:
        reduced["report"] = {key: report[key] for key in
                             ("selection_strategy", "summary", "final_image", "final_selected_id")}
        if include_series and run["status"] == "complete":
            reduced["series"] = {key: [row["metrics"].get(key) for row in report["generations"]]
                                 for key in report["summary"]["metrics"]}
        reduced["images"] = [item | {"url": _url(directory, item["path"], output)}
                             for item in run["images"] if item["path"] == report["final_image"]]
    return reduced


def write_viewer(sources, *, output=None, refresh=False, jobs=None,
                 prepared_reports=None, prepared_aggregates=None):
    """Reuse unchanged diagnostics; rebuild missing/stale caches independently.

    Single runs have one HTML inspector. Collections have a small overview and
    sibling <output-stem>-runs/ pages; all links work offline. Original session,
    metrics and aggregate records are untouched. The save pipeline can supply
    reports and aggregates it already computed to avoid reading them again.
    """
    paths = [Path(sources)] if isinstance(sources, (str, Path)) else [Path(p) for p in sources]
    if not paths:
        raise ValueError("Supply at least one run, batch or collection directory.")
    output = Path(output) if output is not None else paths[0] / "index.html"
    if output.suffix.lower() != ".html":
        raise ValueError("Viewer output must be an .html file.")
    output = output.resolve()
    if jobs is None:
        jobs = min(4, os.cpu_count() or 1)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer.")
    prepared_reports = {Path(k).resolve(): v for k, v in (prepared_reports or {}).items()}
    prepared_aggregates = {Path(k).resolve(): v for k, v in (prepared_aggregates or {}).items()}
    roots = list(dict.fromkeys(root.resolve() for path in paths for root in _discover(path)))
    roots = [root for root in roots if not any(parent != root and (parent / "batch.json").exists()
                                               and root.is_relative_to(parent) for parent in roots)]
    collection_root = Path(os.path.commonpath([path.resolve() for path in paths]))
    payload = {"title": paths[0].name if len(paths) == 1 else "Experiment collection", "runs": [], "groups": []}
    entries = []
    batches = set()
    for group_index, root in enumerate(roots):
        batch = json.loads((root / "batch.json").read_text()) if (root / "batch.json").exists() else None
        if batch:
            batches.add(group_index)
        experiment = root.parent if not batch and (root.parent / "batch.json").is_file() else root
        experiment_name = (str(experiment.relative_to(collection_root))
                           if experiment != collection_root and experiment.is_relative_to(collection_root)
                           else experiment.name)
        group = {"label": experiment_name, "run_ids": [], "aggregate": None}
        for entry in batch["runs"] if batch else [{}]:
            directory = (root / entry["directory"]).resolve() if batch else root
            if not directory.is_relative_to(root):
                raise ValueError("Batch run directory points outside the batch.")
            group["run_ids"].append(len(entries))
            entries.append((directory, entry, group_index))
        payload["groups"].append(group)
    overview = bool(batches) or len(entries) != 1
    details = [(output.parent / (output.stem + "-runs") /
                (hashlib.sha256(str(directory).encode()).hexdigest()[:16] + ".html")) if overview else output
               for directory, _, _ in entries]
    tasks = [(directory, detail, entry, refresh, prepared_reports.get(directory))
             for (directory, entry, _), detail in zip(entries, details, strict=True)]
    runs = [None] * len(tasks)
    pending = []
    for index, task in enumerate(tasks):
        if task[4] is not None or not (task[0] / "session.json").exists():
            runs[index] = _run_task(task)
        else:
            pending.append(index)
    if jobs > 1 and len(pending) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(pending))) as pool:
            for index, run in zip(pending, pool.map(_run_task, [tasks[i] for i in pending]), strict=True):
                runs[index] = run
    else:
        for index in pending:
            runs[index] = _run_task(tasks[index])
    for run, (_, _, group_index) in zip(runs, entries, strict=True):
        run["experiment"] = payload["groups"][group_index]["label"]
    for group_index in batches:
        group = payload["groups"][group_index]
        root = roots[group_index]
        reports = [runs[i]["report"] for i in group["run_ids"] if runs[i]["report"] is not None]
        aggregate = (prepared_aggregates[root] if root in prepared_aggregates
                     else batch_aggregate(root, reports))
        # The overview only uses metric names, quartile bands and contributing counts.
        group["aggregate"] = {"completed_runs": aggregate["completed_runs"],
                              "run_metrics": dict.fromkeys(aggregate["run_metrics"]),
                              "generations": [{"generation": row["generation"], "metrics": {
                                  key: {stat: value[stat] for stat in ("median", "q25", "q75", "count")}
                                  for key, value in row["metrics"].items()}}
                                  for row in aggregate["generations"]]}
    navigation = [{"experiment": run["experiment"], "label": run["label"], "seed": run["seed"],
                   "strategy": (run["report"] or {}).get("selection_strategy", {}).get("selection_strategy", "Unrecorded strategy"),
                   "url": quote(detail.name, safe="")}
                  for run, detail in zip(runs, details, strict=True)] if overview else []
    for index, (run, detail, (directory, _, group_index)) in enumerate(zip(runs, details, entries, strict=True)):
        run.update(id=index, group=group_index)
        if overview:
            title = run["experiment"] + (f" · {run['label']}" if run["label"] != run["experiment"] else "")
            _write_page({"title": title, "runs": [run | {"id": 0, "group": 0}], "groups": [],
                         "navigation": navigation, "current_run": index,
                         "overview": quote(os.path.relpath(output, detail.parent), safe="/")}, detail)
            payload["runs"].append(_overview_run(run, directory, detail, output, include_series=group_index in batches))
        else:
            payload["runs"].append(run)
    payload["is_overview"] = overview
    _write_page(payload, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and open a cached offline viewer for saved experiments.")
    parser.add_argument("sources", nargs="*", type=Path, default=[Path("runs")],
                        help="Run, batch or collection directories (default: runs/).")
    parser.add_argument("--output", type=Path, help="HTML output (default: first source/index.html).")
    parser.add_argument("--no-open", action="store_true", help="Build without opening a browser.")
    parser.add_argument("--refresh", action="store_true", help="Recompute all diagnostics, even if caches are current.")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel run workers (default: up to 4; use 1 for serial work).")
    args = parser.parse_args(argv)
    try:
        result = write_viewer(args.sources, output=args.output, refresh=args.refresh, jobs=args.jobs)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Open {result.resolve()}")
    if not args.no_open:
        webbrowser.open(result.resolve().as_uri())
