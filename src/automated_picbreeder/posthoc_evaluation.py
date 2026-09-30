"""Optional frozen ImageNet analysis, separate from selection-time measurements."""

import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import numpy as np

from .persistence import write_json


def evaluate_session_imagenet(directory, *, evaluator, batch_size=64):
    """Evaluate each saved genome once, including discarded branches and rejects.

    Supply the same configured ImageNetEvaluator for comparable runs. This does
    not change the session or automatically regenerate batch aggregates. Repeated
    presentations reuse the saved genome's measurement; distinct genomes with
    identical pixels still count separately. Model construction cost is excluded.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    directory = Path(directory)
    source = (directory / "session.json").read_bytes()
    data = json.loads(source)
    started = time.perf_counter()
    records, names, values = data["genomes"], None, []
    for offset in range(0, len(records), batch_size):
        images = []
        for record in records[offset:offset + batch_size]:
            with Image.open(directory / record["image"]) as image:
                images.append(np.asarray(image.convert("RGB")))
        evaluation = evaluator.evaluate(images)
        if len(evaluation.values) != len(images):
            raise ValueError("Expected one classifier row per saved genome.")
        if names is not None and names != list(evaluation.names):
            raise ValueError("Classifier column identities changed during evaluation.")
        names = list(evaluation.names)
        values.extend(evaluation.values.tolist())
    report = {"schema_version": 1, "session_sha256": hashlib.sha256(source).hexdigest(),
              "evaluator": evaluator.describe(), "names": names, "values": values,
              "genome_ids": [record["key"] for record in records], "evaluated_images": len(records),
              "batch_size": batch_size, "elapsed_seconds": time.perf_counter() - started,
              "scope": "Post-run classification once per saved genome; includes image loading and inference; excludes model construction. Not used during selection."}
    write_json(directory / "posthoc_imagenet.json", report)
    return report


def load_posthoc_evaluation(directory):
    path = directory / "posthoc_imagenet.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    current = hashlib.sha256((directory / "session.json").read_bytes()).hexdigest()
    if record["session_sha256"] != current:
        raise ValueError("Post-run classifier measurements belong to a different session snapshot; evaluate this snapshot again.")
    return record
