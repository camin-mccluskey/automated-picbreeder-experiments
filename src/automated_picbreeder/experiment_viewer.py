"""Generate a local, offline HTML inspector from saved experiment records."""

import argparse
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote
import webbrowser

from .experiment_metrics import build_run_metrics, summarize_batch


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
    if not target.is_relative_to(directory.resolve()):
        raise ValueError(f"Artifact path points outside the run: {relative}")
    return quote(os.path.relpath(target, output.parent), safe="/")


def _run(directory, output, entry, refresh):
    run = {"label": directory.name, "status": entry.get("status", "incomplete"),
           "seed": entry.get("seed"), "report": None, "images": [], "files": {},
           "error": None, "setup_seconds": entry.get("setup_seconds"), "wall_seconds": entry.get("wall_seconds")}
    if entry.get("error"):
        run["error"] = f"{entry['error']['type']}: {entry['error']['message']}"
    if not (directory / "session.json").exists():
        return run
    try:
        session = json.loads((directory / "session.json").read_text())
        images = [{"genome_id": record["key"], "parent_id": record["parent"], "path": record["image"],
                   "url": _url(directory, record["image"], output)} for record in session["genomes"]]
        files = {name: _url(directory, name, output) for name in ("session.json", "metrics.json", "metrics.csv", "posthoc_imagenet.json", "selection_failure.json")
                 if (directory / name).is_file()}
        run.update(images=images, files=files)
        report = (build_run_metrics(directory, status=entry.get("status")) if refresh else
                  json.loads((directory / "metrics.json").read_text()))
        if entry.get("status"):
            report["status"] = entry["status"]
        run.update(report=report, images=images, files=files, status=report["status"], seed=report["seed"])
    except (ValueError, KeyError, OSError, TypeError) as exc:
        run["error"] = f"Cannot read saved results: {exc}"
    return run


def write_viewer(sources, *, output=None, refresh=True):
    """Write one HTML file; image URLs remain relative to the existing run folders.

    Accept a single run, a batch, a parent collection, or a list of those paths.
    Explicit generation rebuilds diagnostics in memory without altering records.
    The save pipeline uses refresh=False to reuse metrics it just generated.
    """
    paths = [Path(sources)] if isinstance(sources, (str, Path)) else [Path(p) for p in sources]
    if not paths:
        raise ValueError("Supply at least one run, batch or collection directory.")
    output = Path(output) if output is not None else paths[0] / "index.html"
    if output.suffix.lower() != ".html":
        raise ValueError("Viewer output must be an .html file.")
    roots = list(dict.fromkeys(root.resolve() for path in paths for root in _discover(path)))
    # If both a batch and a child were explicitly supplied, show the child once.
    roots = [root for root in roots if not any(parent != root and (parent / "batch.json").exists()
                                               and root.is_relative_to(parent) for parent in roots)]
    payload = {"title": paths[0].name if len(paths) == 1 else "Experiment collection", "runs": [], "groups": []}
    for root in roots:
        group = {"label": root.name, "run_ids": [], "aggregate": None}
        batch = json.loads((root / "batch.json").read_text()) if (root / "batch.json").exists() else None
        entries = batch["runs"] if batch else [{}]
        reports = []
        for entry in entries:
            directory = (root / entry["directory"]).resolve() if batch else root
            if not directory.is_relative_to(root):
                raise ValueError("Batch run directory points outside the batch.")
            run = _run(directory, output, entry, refresh)
            run["id"] = len(payload["runs"])
            run["group"] = len(payload["groups"])
            payload["runs"].append(run)
            group["run_ids"].append(run["id"])
            if run["report"] is not None:
                reports.append(run["report"])
        if batch:
            group["aggregate"] = summarize_batch(reports)
        payload["groups"].append(group)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html)
    temporary.replace(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and open an offline HTML viewer for saved automated or human experiments.")
    parser.add_argument("sources", nargs="*", type=Path, default=[Path("runs")],
                        help="Run, batch or parent collection directories (default: runs/).")
    parser.add_argument("--output", type=Path, help="HTML output (default: first source/index.html).")
    parser.add_argument("--no-open", action="store_true", help="Build the HTML without opening a browser.")
    args = parser.parse_args(argv)
    try:
        result = write_viewer(args.sources, output=args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Open {result.resolve()}")
    if not args.no_open:
        webbrowser.open(result.resolve().as_uri())
