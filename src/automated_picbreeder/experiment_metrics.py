"""Versioned diagnostics derived from saved human or automated sessions.

All image distances use full-resolution RGB / 255. Generation indices are zero
based, matching grids and strategy records. None means unavailable, never zero.
"""

import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .persistence import write_json
from .posthoc_evaluation import load_posthoc_evaluation
from .session_history import session_history


SCHEMA_VERSION = 1


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(count=0, mean=None, median=None, min=None, max=None, q25=None, q75=None, std=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max()),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                std=float(values.std(ddof=1)) if len(values) > 1 else None)


def write_csv(path, rows):
    """Blank cells represent missing values. JSON remains the canonical artifact."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _grid_stats(metrics, prefix, values, position):
    if values is None:
        metrics.update({f"{prefix}_{suffix}": None for suffix in ("selected", "grid_mean", "grid_min", "grid_max")})
    else:
        values = np.asarray(values, dtype=float)
        metrics.update({f"{prefix}_selected": float(values[position]),
                        f"{prefix}_grid_mean": float(values.mean()),
                        f"{prefix}_grid_min": float(values.min()),
                        f"{prefix}_grid_max": float(values.max())})


def _measurements(event, candidates, metrics):
    evaluation = event["evaluation"]
    if evaluation is None:
        return {}
    meta = evaluation["metadata"]
    available = meta.get("measurement_available", [True] * len(evaluation["names"]))
    for column, name in enumerate(evaluation["names"]):
        if name not in {"pixel_novelty", "comprehension", "predicted_offspring_value"}:
            continue
        values = [row[column] for row in evaluation["values"]] if available[column] else None
        prefix = {"pixel_novelty": "novelty", "comprehension": "comprehension",
                  "predicted_offspring_value": "offspring_forecast"}[name]
        _grid_stats(metrics, prefix, values, event["position"])
        for i, candidate in enumerate(candidates):
            candidate[name] = None if values is None else values[i]
        if name == "pixel_novelty":
            metrics["novelty_parent_margin"] = (
                float(values[event["position"]] - values[0]) if values is not None else None)
    return meta


def _classification_stats(classification, candidates, metrics, position, previous_class, *, prefix="imagenet", candidate_prefix=""):
    values = np.asarray(classification["values"])
    indices = values.argmax(axis=1)
    maxima = values.max(axis=1)
    for candidate, index, confidence in zip(candidates, indices, maxima, strict=True):
        candidate[f"{candidate_prefix}top_class"] = {"index": int(index), "name": classification["names"][index]}
        candidate[f"{candidate_prefix}max_class_confidence"] = float(confidence)
    selected = candidates[position][f"{candidate_prefix}top_class"]
    metrics.update({f"{prefix}_selected_confidence": float(maxima[position]),
                    f"{prefix}_grid_mean_confidence": float(maxima.mean()),
                    f"{prefix}_grid_min_confidence": float(maxima.min()),
                    f"{prefix}_grid_max_confidence": float(maxima.max()),
                    f"{prefix}_top_class_changed": None if previous_class is None else int(selected["index"] != previous_class["index"])})
    return selected


def _classifier(event, candidates, metrics, strategy_name, previous_class):
    evaluation = event["evaluation"]
    classification = (evaluation if strategy_name == "imagenet" else
                      (evaluation or {}).get("metadata", {}).get("classifier_evaluation"))
    metrics["classifier_images"] = 0
    if classification is None:
        return None
    metrics["classifier_images"] = len(candidates)
    return _classification_stats(classification, candidates, metrics, event["position"], previous_class)


def _value_components(meta, config, candidates, metrics, position):
    quality_key = "confidence_ranks" if "confidence_ranks" in meta else "comprehension_ranks"
    if quality_key not in meta:
        return
    weight = meta.get("effective_comprehension_weight", config["comprehension_weight"])
    novelty = (1 - weight) * np.asarray(meta["novelty_ranks"])
    quality = weight * np.asarray(meta[quality_key])
    metrics.update(value_quality_weight=weight, value_novelty_contribution=float(novelty[position]),
                   value_quality_contribution=float(quality[position]),
                   value_current=float(novelty[position] + quality[position]),
                   choice_differs_from_novelty=(int(position != int(np.argmax(meta["novelty_ranks"])))
                                                if meta["reference_available"] else None),
                   choice_differs_from_quality=(int(position != int(np.argmax(meta[quality_key])))
                                                if meta.get("comprehension_available", True) else None))
    for i, candidate in enumerate(candidates):
        candidate["value_components"] = {"novelty": float(novelty[i]), "quality": float(quality[i])}


def _observer(meta, metrics, candidates, position):
    if "masked_mse" not in meta:
        return
    errors = meta["masked_mse"] if meta["comprehension_available"] else None
    _grid_stats(metrics, "prediction_mse", errors, position)
    for i, candidate in enumerate(candidates):
        candidate["prediction_mse"] = None if errors is None else errors[i]
    training = meta["training"]
    metrics.update(comprehension_warmup_complete=int(meta["comprehension_warmup_complete"]),
                   observer_scoring_seconds=meta["scoring_seconds"],
                   observer_training_seconds=meta["training_seconds"],
                   observer_masked_inputs=meta["masked_inputs_scored"],
                   observer_training_updates=training["updates"],
                   observer_training_examples=training.get("sampled_examples"),
                   observer_training_loss=training.get("mean_loss"),
                   observer_replay_images=meta["replay_size_after"])


def _offspring(meta, metrics, candidates, position):
    record = meta.get("offspring")
    if record is None:
        return None
    contributions = np.asarray(record["forecasts"]) * record["gamma"] if record["forecast_used"] else np.zeros(9)
    for i, candidate in enumerate(candidates):
        candidate["value_components"]["offspring"] = float(contributions[i])
    training = record["training"] or {}
    metrics.update(value_offspring_contribution=float(contributions[position]),
                   offspring_forecast_used=int(record["forecast_used"]),
                   offspring_warmup_complete=int(record["warmup_complete"]),
                   offspring_target_eligible=int(record["target_eligible"]),
                   offspring_choice_changed=int(record["choice_changed"]),
                   offspring_completed_targets=record["completed_targets"],
                   offspring_target=None, offspring_error=None, offspring_absolute_error=None,
                   offspring_squared_error=None, offspring_running_mean_squared_error=None,
                   offspring_current_value_squared_error=None,
                   offspring_target_scoring_seconds=record["target_scoring_seconds"],
                   offspring_predictor_training_seconds=record["predictor_training_seconds"],
                   offspring_predictor_inference_seconds=record["inference_seconds"],
                   offspring_snapshot_seconds=record["snapshot_seconds"],
                   offspring_target_masked_inputs=record.get("target_masked_inputs_scored", 0),
                   offspring_classifier_measurements_reused=record.get("target_classifier_measurements_reused", 0),
                   offspring_predictor_training_updates=training.get("updates", 0),
                   offspring_predictor_training_examples=training.get("sampled_examples"),
                   offspring_predictor_training_loss=training.get("mean_loss"))
    return record["feedback"]


def _align_offspring_feedback(rows, feedbacks):
    """Outcome belongs to the original parent decision, not its arrival grid."""
    for feedback in feedbacks:
        row = rows[feedback["origin_generation"]]
        row["offspring_outcome"] = {
            "received_generation": feedback["received_generation"],
            "forecast": feedback["forecast"], "forecast_used": feedback["forecast_used"],
            "running_mean_forecast": feedback["running_mean_forecast"],
            "current_value": feedback["current_value"],
            "child_names": feedback["child_names"], "child_values": feedback["child_values"],
        }
        for name in ("target", "error", "absolute_error", "squared_error",
                     "running_mean_squared_error", "current_value_squared_error"):
            row["metrics"][f"offspring_{name}"] = feedback[name]
    errors = []
    for row in rows:
        metrics = row["metrics"]
        if "offspring_forecast_used" not in metrics:
            continue
        if metrics["offspring_target"] is not None:
            errors.append(metrics)
        metrics["offspring_observed_targets"] = len(errors)
        metrics["offspring_cumulative_mae"] = float(np.mean([e["offspring_absolute_error"] for e in errors])) if errors else None
        for prefix in ("", "running_mean_", "current_value_"):
            metrics[f"offspring_{prefix}cumulative_rmse"] = (
                float(np.sqrt(np.mean([e[f"offspring_{prefix}squared_error"] for e in errors]))) if errors else None)


def _forecast_summary(rows):
    result = {}
    for group in ("all", "forecast_used", "forecast_not_used"):
        eligible = [row["metrics"] for row in rows if row["metrics"].get("offspring_target") is not None
                    and (group == "all" or bool(row["metrics"]["offspring_forecast_used"]) == (group == "forecast_used"))]
        result[group] = {"count": len(eligible),
                         "mae": float(np.mean([m["offspring_absolute_error"] for m in eligible])) if eligible else None}
        for prefix in ("", "running_mean_", "current_value_"):
            result[group][f"{prefix}rmse"] = float(np.sqrt(np.mean([
                m[f"offspring_{prefix}squared_error"] for m in eligible]))) if eligible else None
    return result


def image_inventory(data):
    """Small viewer records; no network parameters or classifier matrices."""
    return [{"genome_id": g["key"], "parent_id": g["parent"], "path": g["image"]}
            for g in data["genomes"]]


def build_run_metrics(directory, *, status=None, data=None):
    """Read saved records and PNGs; never load strategy models or perform inference.

    Rows follow explicit selection events, including human reselections. Display
    visits and final ancestry are separate, so no discarded exploration is hidden.
    Historical full-image MSE costs O(S² * pixels); compare in bounded chunks and
    deduplicate exact earlier images without changing the minimum distance.
    """
    directory = Path(directory)
    if data is None:
        data = json.loads((directory / "session.json").read_text())
    config = data["metadata"]["selection_strategy"]
    human = config["selection_strategy"] == "human"
    if not human and (any(e["action"] == "back" for e in data["events"]) or sum(e["action"] == "reset" for e in data["events"]) != 1):
        raise ValueError("Expected an automated trajectory without resets or backtracking.")
    selections, displays, ancestry = session_history(data)
    ancestry_ids = {node["genome_id"] for node in ancestry}
    records = {genome["key"]: genome for genome in data["genomes"]}

    @lru_cache(maxsize=32)
    def load_image(key):
        with Image.open(directory / records[key]["image"]) as image:
            return np.asarray(image.convert("RGB"))

    @lru_cache(maxsize=2)
    def display_mean(index):
        return np.asarray([load_image(key) for key in displays[index]["displayed"]], dtype=float).mean(axis=0)/255

    performance_path = directory / "performance.json"
    performance = json.loads(performance_path.read_text()) if performance_path.exists() else {}
    timings = {entry["generation"]: entry for entry in performance.get("generations", [])}
    interaction = {entry["event_index"]: entry for entry in performance.get("events", [])}
    posthoc = load_posthoc_evaluation(directory)
    posthoc_values = dict(zip(posthoc["genome_ids"], posthoc["values"], strict=True)) if posthoc else {}
    rows, history, seen, feedbacks = [], [], set(), []
    previous = previous_id = previous_class = None
    previous_posthoc_class = None
    streak = explored = 0
    for generation, context in enumerate(selections):
        event = context["event"]
        ids, position = event["displayed"], event["position"]
        if not human and (len(ids) != 9 or (generation and (ids[0] != previous_id or any(records[k]["parent"] != previous_id for k in ids[1:])))):
            raise ValueError("Expected nine chronological candidates with the selected parent retained first.")
        images = [load_image(key) for key in ids]
        selected = images[position]
        identity = selected.tobytes()
        unchanged = None if previous is None else bool(np.array_equal(selected, previous))
        streak = streak + 1 if unchanged else 1
        distance = nearest = None
        if previous is not None:
            pixels = selected.astype(float) / 255
            distance = float(np.mean((pixels - previous.astype(float) / 255)**2))
            nearest = 0.0 if identity in seen else min(float(np.mean((pixels - np.asarray(history[i:i+32], dtype=float)/255)**2,
                                        axis=(1, 2, 3)).min()) for i in range(0, len(history), 32))
        if identity not in seen:
            history.append(selected)
            seen.add(identity)
        decision = event["decision"]
        decision_metadata = (decision or {}).get("metadata", {})
        scores = None if decision is None else decision["scores"]
        parent = context["retained_parent"]
        metrics = dict(pixel_mse_previous=distance, pixel_mse_nearest_earlier=nearest,
                       parent_retained=None if parent is None else int(event["genome"] == parent),
                       image_unchanged=None if unchanged is None else int(unchanged), unchanged_image_streak=streak,
                       children_identical_to_parent_fraction=(sum(np.array_equal(im, images[0]) for im in images[1:])/8 if parent is not None else None),
                       distinct_child_images=len({im.tobytes() for im in images[1:]}) if parent is not None else None,
                       selected_score=None if scores is None else scores[position])
        for name in ("breeding_seconds", "rendering_seconds", "selection_seconds", "saving_seconds", "decision_seconds"):
            metrics[name] = timings.get(generation, {}).get(name)
        metrics.update(decision_metadata.get("inference", {}))
        if human:
            metrics["interaction_elapsed_seconds"] = interaction.get(context["event_index"], {}).get("elapsed_seconds")
        candidates = [{"position": i, "genome_id": key, "image": records[key]["image"],
                       "score": None if scores is None else scores[i]} for i, key in enumerate(ids)]
        display_index = context["display_index"]
        reference_mean = display_mean(display_index - 1) if display_index else None
        display_novelty = np.mean((np.asarray(images, dtype=float)/255 - reference_mean)**2, axis=(1, 2, 3)) if display_index else None
        display_mean(display_index)  # Keep this visit's mean for the next selection.
        _grid_stats(metrics, "display_novelty", display_novelty, position)
        for i, candidate in enumerate(candidates):
            candidate["display_novelty"] = float(display_novelty[i]) if display_novelty is not None else None
        meta = _measurements(event, candidates, metrics)
        selected_class = _classifier(event, candidates, metrics, config["selection_strategy"], previous_class)
        posthoc_class = None
        if posthoc:
            posthoc_class = _classification_stats({"values": [posthoc_values[key] for key in ids], "names": posthoc["names"]},
                candidates, metrics, position, previous_posthoc_class, prefix="posthoc_imagenet", candidate_prefix="posthoc_")
        if config["selection_strategy"] == "imagenet":
            exploratory = int(event["decision"]["mode"] == "random")
            explored += exploratory
            metrics.update(exploratory_choice=exploratory, exploratory_choice_frequency=explored/(generation + 1))
        _value_components(meta, config, candidates, metrics, position)
        _observer(meta, metrics, candidates, position)
        feedback = _offspring(meta, metrics, candidates, position)
        if feedback is not None:
            feedbacks.append(feedback)
        rows.append(dict(generation=generation, selected_id=event["genome"], selected_position=position,
                         selected_image=records[event["genome"]]["image"], selected_class=selected_class,
                         posthoc_selected_class=posthoc_class,
                         selection_mode="human" if decision is None else decision["mode"], grid=f"grids/{generation:06d}.png",
                         candidates=candidates, metrics=metrics, on_final_ancestry=event["genome"] in ancestry_ids,
                         **({"decision_metadata": decision_metadata} if decision_metadata else {}),
                         **{key: context[key] for key in ("event_index", "display_index", "round", "mutation", "preceding_actions", "candidate_presentations")}))
        previous, previous_id, previous_class = selected, event["genome"], selected_class
        previous_posthoc_class = posthoc_class
    _align_offspring_feedback(rows, feedbacks)
    keys = sorted({key for row in rows for key in row["metrics"]})
    summary = {
        "decisions": len(rows), "candidate_presentations": 9*len(displays), "displayed_grids": len(displays),
        "selection_revisions": sum(max(0, len(d["selection_generations"]) - 1) for d in displays),
        "backtracks": sum(d["action"] == "back" for d in displays),
        "resets": sum(d["action"] == "reset" for d in displays) - 1,
        "generated_genomes": len(records),
        "classifier_images": sum(row["metrics"]["classifier_images"] for row in rows),
        "run_seconds": None if human else performance.get("elapsed_seconds"),
        "interaction_seconds": performance.get("elapsed_seconds") if human else None,
        "posthoc_classifier_images": posthoc["evaluated_images"] if posthoc else 0,
        "posthoc_evaluation_seconds": posthoc["elapsed_seconds"] if posthoc else None,
        "seconds_per_decision": performance.get("elapsed_seconds") / len(rows) if not human and rows and performance.get("elapsed_seconds") is not None else None,
        "metrics": {key: distribution([row["metrics"].get(key) for row in rows]) | {"final": rows[-1]["metrics"].get(key)} for key in keys},
        "offspring_prediction": _forecast_summary(rows) if "offspring_forecast_used" in keys else None,
    }
    for key in keys:
        if key.startswith("api_"):
            values = [row["metrics"].get(key) for row in rows]
            summary["metrics"][key]["total"] = sum(values) if all(v is not None for v in values) else None
        elif (key.endswith("_seconds") and key != "interaction_elapsed_seconds") or key.endswith(("_inputs", "_updates", "_examples", "_reused")):
            values = [row["metrics"].get(key) for row in rows]
            summary["metrics"][key]["total"] = sum(v for v in values if v is not None) if any(v is not None for v in values) else None
    return dict(schema_version=SCHEMA_VERSION, status=status or (data.get("summary") or {}).get("status", "complete" if human else "incomplete"),
                session="session.json", seed=data["seed"], settings=data["metadata"]["settings"],
                selection_strategy=config, generation_index="zero_based", pixel_scale="RGB / 255, full resolution",
                trajectory_axis="selection_event", display_history=displays, final_ancestry=ancestry,
                posthoc_evaluation={key: value for key, value in posthoc.items() if key not in {"values", "genome_ids", "names"}} if posthoc else None,
                final_selected_id=data["selected"],
                final_image=records[data["selected"]]["image"] if data["selected"] is not None else None,
                generations=rows, summary=summary, image_inventory=image_inventory(data))


def save_run_metrics(directory, *, status=None):
    directory = Path(directory)
    report = build_run_metrics(directory, status=status)
    if report["selection_strategy"]["selection_strategy"] == "human":
        from .experiment_reporting import contact_sheet
        from .selection_strategies import SelectionDecision

        (directory / "grids").mkdir(exist_ok=True)
        for row in report["generations"]:
            path = directory / row["grid"]
            if not path.exists():
                images = []
                for candidate in row["candidates"]:
                    with Image.open(directory / candidate["image"]) as image:
                        images.append(np.asarray(image.convert("RGB")))
                contact_sheet(images, [c["genome_id"] for c in row["candidates"]],
                              SelectionDecision(position=row["selected_position"], mode="human")).save(path)
    write_json(directory / "metrics.json", report)
    write_csv(directory / "metrics.csv", [
        {"generation": row["generation"], "event_index": row["event_index"], "display_index": row["display_index"],
         "round": row["round"], "candidate_presentations": row["candidate_presentations"],
         "on_final_ancestry": row["on_final_ancestry"], "selected_id": row["selected_id"],
         "selected_position": row["selected_position"], "selected_image": row["selected_image"],
         "selection_mode": row["selection_mode"],
         "selected_class_index": (row["selected_class"] or {}).get("index"),
         "selected_class_name": (row["selected_class"] or {}).get("name"),
         "posthoc_class_index": (row["posthoc_selected_class"] or {}).get("index"),
         "posthoc_class_name": (row["posthoc_selected_class"] or {}).get("name"), **row["metrics"]}
        for row in report["generations"]])
    from .viewer_cache import cache_report
    from .experiment_viewer import write_viewer

    cache_report(directory, report)
    write_viewer(directory, prepared_reports={directory.resolve(): report})
    return report


def summarize_batch(reports):
    """Each completed run contributes once; failed/partial trajectories are excluded."""
    complete = [report for report in reports if report["status"] == "complete"]
    keys = sorted({key for report in complete for key in report["summary"]["metrics"]})
    metrics = {key: {stat: distribution([r["summary"]["metrics"].get(key, {}).get(stat) for r in complete])
                     for stat in ("mean", "final", "max", "total")} for key in keys}
    run_summary = {key: distribution([r["summary"].get(key) for r in complete])
                   for key in ("decisions", "candidate_presentations", "generated_genomes", "classifier_images",
                               "run_seconds", "seconds_per_decision", "displayed_grids", "selection_revisions",
                               "backtracks", "resets", "interaction_seconds", "posthoc_classifier_images",
                               "posthoc_evaluation_seconds")}
    offspring = {}
    if any(r["summary"].get("offspring_prediction") is not None for r in complete):
        for group in ("all", "forecast_used", "forecast_not_used"):
            offspring[group] = {key: distribution([
                (r["summary"].get("offspring_prediction") or {}).get(group, {}).get(key) for r in complete])
                for key in ("count", "mae", "rmse", "running_mean_rmse", "current_value_rmse")}
    generations = []
    for generation in range(max((len(r["generations"]) for r in complete), default=0)):
        rows = [r["generations"][generation]["metrics"] for r in complete if generation < len(r["generations"])]
        generations.append({"generation": generation,
                            "metrics": {key: distribution([row.get(key) for row in rows]) for key in keys}})
    return dict(schema_version=SCHEMA_VERSION, completed_runs=len(complete),
                aggregation="Completed runs only; equal run weight; sample standard deviation; no confidence intervals.",
                run_summary=run_summary, run_metrics=metrics, offspring_prediction=offspring or None,
                generations=generations)
