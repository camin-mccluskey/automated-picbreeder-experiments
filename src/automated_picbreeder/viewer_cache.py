"""Disposable viewer diagnostics, separate from original experiment records."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .experiment_metrics import build_run_metrics, image_inventory, summarize_batch


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _write(path, value):
    atomic_text(path, json.dumps(value, separators=(",", ":"), allow_nan=False))


@lru_cache(maxsize=1)
def code_version():
    package = Path(__file__).parent
    return hashlib.sha256(b"".join((package / name).read_bytes() for name in (
        "experiment_metrics.py", "session_history.py", "posthoc_evaluation.py", "viewer_cache.py",
    ))).hexdigest()


def _stamp(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def fingerprint(directory, images):
    # Stat the PNGs too: replacing, removing or editing an image invalidates metrics.
    # No PNG decoding or multi-megabyte session parsing on the unchanged path.
    return {"code": code_version(),
            "records": {name: _stamp(directory / name) for name in
                        ("session.json", "performance.json", "posthoc_imagenet.json")},
            "images": [_stamp(directory / item["path"]) for item in images]}


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def cache_report(directory, report, *, stamp=None):
    directory = Path(directory)
    _write(directory / ".viewer-cache/records.json", {
        "fingerprint": stamp or fingerprint(directory, report["image_inventory"]), "report": report,
    })


def load_report(directory, *, refresh=False):
    directory = Path(directory)
    cached = None if refresh else _read(directory / ".viewer-cache/records.json")
    if isinstance(cached, dict):
        report = cached.get("report")
        if isinstance(report, dict) and isinstance(report.get("image_inventory"), list):
            try:
                if cached.get("fingerprint") == fingerprint(directory, report["image_inventory"]):
                    return report
            except (KeyError, TypeError):
                pass
    # Old runs are indexed once. Do not rewrite their metrics.json or source records.
    session_stamp = _stamp(directory / "session.json")
    data = json.loads((directory / "session.json").read_text())
    images = image_inventory(data)
    root = directory.resolve()
    for item in images:
        if not (directory / item["path"]).resolve().is_relative_to(root):
            raise ValueError(f"Artifact path points outside the run: {item['path']}")
    before = fingerprint(directory, images)
    if before["records"]["session.json"] != session_stamp:
        raise ValueError("Session changed while reading; rebuild the viewer after saving finishes.")
    report = build_run_metrics(directory, data=data)
    if before != fingerprint(directory, images):
        raise ValueError("Saved inputs changed during metrics extraction; rebuild the viewer after saving finishes.")
    cache_report(directory, report, stamp=before)
    return report


def batch_aggregate(directory, reports):
    # Only fields consumed by summarize_batch; avoids reaggregating unchanged runs.
    inputs = [{"status": r["status"], "summary": r["summary"],
               "generations": [g["metrics"] for g in r["generations"]]} for r in reports]
    signature = hashlib.sha256((code_version() + json.dumps(inputs, sort_keys=True)).encode()).hexdigest()
    path = directory / ".viewer-cache/aggregate.json"
    cached = _read(path)
    if isinstance(cached, dict) and cached.get("signature") == signature and isinstance(cached.get("aggregate"), dict):
        return cached["aggregate"]
    aggregate = summarize_batch(reports)
    _write(path, {"signature": signature, "aggregate": aggregate})
    return aggregate
