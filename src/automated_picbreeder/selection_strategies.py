"""Choose the next image; evaluation and scoring belong to the selection strategy."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json
import time
from numbers import Integral, Real
from random import Random
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .evaluation import Evaluation, ImageArray, validate_image

if TYPE_CHECKING:
    from .imagenet import ImageNetEvaluator


class SelectionError(RuntimeError):
    """A failed choice with JSON-compatible diagnostics for the runner to save."""

    def __init__(self, message, *, metadata):
        super().__init__(message)
        self.metadata = metadata


@dataclass(frozen=True)
class SelectionDecision:
    """One choice, with optional measurements explaining how it was made.

    Scores, when present, contain one preference value per candidate. They are
    distinct from the raw evaluation matrix. Mode records the operation used
    (e.g. 'random' or 'greedy'), even when both would choose the same candidate.
    Call validate with the current grid size before recording a decision.
    """

    position: int
    mode: str
    evaluation: Evaluation | None = None
    scores: NDArray[np.floating] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.position, bool) or not isinstance(self.position, Integral) or self.position < 0:
            raise ValueError("Selection position must be a non-negative integer.")
        object.__setattr__(self, "position", int(self.position))
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("Selection mode must be a non-empty string.")
        if not isinstance(self.metadata, dict):
            raise ValueError("Decision metadata must be a JSON-compatible dictionary.")
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))
        if self.scores is not None:
            scores = np.asarray(self.scores)
            if scores.ndim != 1 or scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
                raise ValueError("Selection scores must be a one-dimensional array of finite real numbers.")
            object.__setattr__(self, "scores", scores.copy())

    def validate(self, candidate_count: int) -> None:
        """Check the decision against the images supplied to the selection strategy."""
        if self.position >= candidate_count:
            raise ValueError("Selection position is outside the displayed grid.")
        if self.scores is not None and len(self.scores) != candidate_count:
            raise ValueError("Expected one selection score per displayed candidate.")
        if self.evaluation is not None and len(self.evaluation.values) != candidate_count:
            raise ValueError("Evaluation must contain one row per displayed candidate.")


class SelectionStrategy(Protocol):
    """Complete image selection, independent of breeding and persistence.

    Preserve image order and contents. Own any evaluation and scoring needed
    for the choice, and return their results for inspection. Use the supplied
    RNG for selection randomness rather than global or persistent RNG state.
    Stateful strategies advance on each choose call. Create a fresh instance for
    each independent run or inspection replay; a preview consumes its history.
    """

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision: ...

    def describe(self) -> dict:
        """Return JSON-compatible identification/configuration for run records."""
        ...


class RandomSelectionStrategy:
    """Uniformly select a candidate, including the retained parent, without scoring."""

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        if not images:
            raise ValueError("Cannot select from an empty candidate grid.")
        return SelectionDecision(position=rng.randrange(len(images)), mode="random")

    def describe(self) -> dict:
        return {"selection_strategy": "random", "rule": "uniform over all candidates, including the parent"}


class VLMSelectionStrategy:
    """Choose directly from nine labelled PNGs in one stateless OpenRouter request."""

    def __init__(self, *, model=None, prompt="choose the most interesting image to you",
                 temperature=0.0, max_completion_tokens=1024, timeout=120.0,
                 max_retries=5, env_file=None, client=None):
        from .vlm import OpenRouterSelection

        self._selector = OpenRouterSelection(
            model=model, prompt=prompt, temperature=temperature,
            max_completion_tokens=max_completion_tokens, timeout=timeout,
            max_retries=max_retries, env_file=env_file, client=client,
        )

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        # Fixed presentation order and no local random choice: do not consume rng.
        position, metadata = self._select(images)
        return SelectionDecision(position=position, mode="vlm", metadata=metadata)

    def _select(self, images, **kwargs):
        if len(images) != 9:
            raise ValueError("VLM selection requires exactly nine RGB images.")
        for image in images:
            validate_image(image)
            if image.ndim != 3:
                raise ValueError("VLM selection requires RGB images.")
        return self._selector.select(images, **kwargs)

    def describe(self) -> dict:
        return {"selection_strategy": "vlm", **self._selector.describe()}


class VLMScratchpadSelectionStrategy(VLMSelectionStrategy):
    """Choose from the current grid with one model-written, rewritable note."""

    def __init__(self, *, model=None, prompt="find something interesting",
                 temperature=0.0, max_completion_tokens=1024, timeout=120.0,
                 max_retries=5, env_file=None, client=None):
        from .vlm import OpenRouterScratchpadSelection

        self._selector = OpenRouterScratchpadSelection(
            model=model, prompt=prompt, temperature=temperature,
            max_completion_tokens=max_completion_tokens, timeout=timeout,
            max_retries=max_retries, env_file=env_file, client=client,
        )
        self._scratchpad = ""

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        position, metadata = self._select(images, scratchpad=self._scratchpad)
        decision = SelectionDecision(position=position, mode="vlm-scratchpad", metadata=metadata)
        # A failed request or invalid response must not replace the previous note.
        self._scratchpad = metadata["vlm"]["scratchpad"]
        return decision

    def describe(self) -> dict:
        return {"selection_strategy": "vlm-scratchpad", **self._selector.describe(),
                "memory": "one rewritable text note, initially empty; no historical images"}


def _percentile_ranks(values: NDArray) -> NDArray:
    """Ascending ranks in [0, 1], averaging exact ties; singleton is neutral."""
    if len(values) == 1:
        return np.full(1, .5)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2)[inverse] / (len(values) - 1)


def _image_hash(image: ImageArray) -> str:
    """Identify exact pixels including their layout, not their genome or class."""
    digest = hashlib.sha256(f"{image.shape}:{image.dtype}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _validate_novelty_reference(novelty_reference):
    if novelty_reference not in ("previous-grid-mean", "previous-parent"):
        raise ValueError("novelty_reference must be previous-grid-mean or previous-parent.")


class NoveltySelectionStrategy:
    """Prefer pixel distance from the preceding grid mean or selected parent.

    The default mean weights every displayed image equally, including duplicates.
    Previous-parent mode uses the preceding choice's actual selected image.
    Scores are average-tie percentile ranks of RGB/grayscale mean squared distance
    in [0, 1]. The first grid has no reference and is selected uniformly. Later
    ties favour the first candidate. No model, resizing or cropping is involved.

    Construct a fresh instance per run: choose advances the reference even in
    notebook inspection. Image shapes must stay constant within an instance.
    """

    def __init__(self, *, novelty_reference="previous-grid-mean"):
        _validate_novelty_reference(novelty_reference)
        self.novelty_reference = novelty_reference
        self._reference_image = None
        self._generation = 0
        self._seen_hashes: set[str] = set()

    def _measure(self, images):
        if not images:
            raise ValueError("Cannot select from an empty candidate grid.")
        for image in images:
            validate_image(image)
        shape = images[0].shape if self._reference_image is None else self._reference_image.shape
        if any(image.shape != shape for image in images):
            raise ValueError("Image shape must stay the same within a strategy instance.")

        # Convert before arithmetic to prevent uint8 subtraction/summation overflow.
        pixels = np.stack(images).astype(np.float64) / 255
        available = self._reference_image is not None
        novelty = (np.mean((pixels - self._reference_image) ** 2, axis=tuple(range(1, pixels.ndim)))
                   if available else np.zeros(len(images)))
        hashes = [_image_hash(image) for image in images]
        metadata = {
            "generation": self._generation,
            "reference_generation": self._generation - 1 if available else None,
            "reference_available": available,
            "novelty_reference": self.novelty_reference,
            "measurement_available": [available],
            "image_hashes": hashes,
            "seen_before": [key in self._seen_hashes for key in hashes],
        }
        return pixels, novelty, metadata

    def _remember(self, pixels, hashes, position):
        self._reference_image = (pixels.mean(axis=0) if self.novelty_reference == "previous-grid-mean"
                                 else pixels[position].copy())
        self._seen_hashes.update(hashes)
        self._generation += 1

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        pixels, novelty, metadata = self._measure(images)
        available = metadata["reference_available"]
        scores = _percentile_ranks(novelty)
        decision = SelectionDecision(
            position=int(scores.argmax()) if available else rng.randrange(len(images)),
            mode="greedy" if available else "random",
            evaluation=Evaluation(novelty[:, None], ("pixel_novelty",), metadata), scores=scores,
        )
        # Advance only after the whole grid has been scored against one reference.
        self._remember(pixels, metadata["image_hashes"], decision.position)
        return decision

    def describe(self) -> dict:
        return {
            "selection_strategy": "novelty", "version": 1,
            "novelty_reference": self.novelty_reference,
            "measurement": "mean squared pixel distance from the configured preceding-decision reference image",
            "reference": ("all previous-grid images equally weighted, including duplicates; replaced each decision"
                          if self.novelty_reference == "previous-grid-mean"
                          else "actual image selected on the previous decision; replaced each decision"),
            "preprocessing": "full-resolution uint8 pixels converted to float64 and divided by 255; no crop or resize",
            "score": "ascending percentile ranks with average ties; singleton and all ties score 0.5",
            "rule": "uniform first choice (no reference); otherwise first maximum score",
            "first_grid": "zero raw novelty, neutral ranks, reference_available=false",
            "lifecycle": "fresh instance per independent run; each choose advances state",
            "image_hash": "sha256 of ASCII shape:dtype followed by C-order pixel bytes",
        }


class NoveltyImageNetSelectionStrategy(NoveltySelectionStrategy):
    """Combine pixel novelty ranks with frozen maximum-class confidence ranks.

    Confidence is available immediately; the first grid has neutral novelty.
    Weight zero preserves novelty's uniform first choice. No training or warm-up
    is needed. Construct a fresh strategy per run to reset the novelty reference.
    """

    def __init__(self, comprehension_weight=.5, *, evaluator: "ImageNetEvaluator | None" = None,
                 novelty_reference="previous-grid-mean"):
        if isinstance(comprehension_weight, bool) or not isinstance(comprehension_weight, Real) or not 0 <= comprehension_weight <= 1:
            raise ValueError("comprehension_weight must be in [0, 1].")
        super().__init__(novelty_reference=novelty_reference)
        self.comprehension_weight = float(comprehension_weight)
        if evaluator is None:
            from .imagenet import ImageNetEvaluator

            evaluator = ImageNetEvaluator()
        self.evaluator = evaluator

    def _score_current(self, images, rng):
        pixels, novelty, metadata = self._measure(images)
        classification = self.evaluator.evaluate(images)
        if len(classification.values) != len(images):
            raise ValueError("Evaluation must contain one row per displayed candidate.")
        confidence = classification.values.max(axis=1)
        novelty_ranks = _percentile_ranks(novelty)
        confidence_ranks = _percentile_ranks(confidence)
        weight = self.comprehension_weight
        scores = (1 - weight) * novelty_ranks + weight * confidence_ranks
        random_first = not metadata["reference_available"] and weight == 0
        metadata.update({
            "measurement_available": [metadata["reference_available"], True],
            "novelty_ranks": novelty_ranks.tolist(),
            "confidence_ranks": confidence_ranks.tolist(),
            "classifier_evaluation": {
                "values": classification.values.tolist(), "names": list(classification.names),
                "metadata": deepcopy(classification.metadata),
            },
        })
        position = rng.randrange(len(images)) if random_first else int(scores.argmax())
        return pixels, novelty, confidence, scores, position, metadata

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        pixels, novelty, confidence, scores, position, metadata = self._score_current(images, rng)
        random_first = not metadata["reference_available"] and self.comprehension_weight == 0
        decision = SelectionDecision(
            position=position, mode="random" if random_first else "greedy",
            evaluation=Evaluation(np.column_stack((novelty, confidence)),
                                  ("pixel_novelty", "imagenet_confidence"), metadata),
            scores=scores,
        )
        self._remember(pixels, metadata["image_hashes"], position)
        return decision

    def describe(self):
        return super().describe() | {
            "selection_strategy": "novelty-imagenet", "comprehension_weight": self.comprehension_weight,
            "score": "(1-weight)*novelty_rank + weight*confidence_rank; average ties",
            "comprehension": "maximum frozen ImageNet class probability; available from the first grid",
            "rule": "first maximum score; uniform first choice only when comprehension_weight=0",
            "first_grid": "zero raw novelty, neutral novelty ranks, reference_available=false; confidence available",
            "evaluator": self.evaluator.describe(),
        }


class NoveltyPredictabilitySelectionStrategy(NoveltySelectionStrategy):
    """Combine pixel novelty and pre-update masked prediction accuracy.

    Every choose scores one grid, fixes the choice, then trains on unique displayed
    images from this and earlier grids. Construct a fresh instance per run.
    """

    def __init__(self, comprehension_weight=.5, *, comprehension_warmup_steps=10, observer_initialization="random",
                 training_steps=20, batch_size=16, learning_rate=.001, cache_dir=None, device="cpu",
                 novelty_reference="previous-grid-mean"):
        if isinstance(comprehension_weight, bool) or not isinstance(comprehension_weight, Real) or not 0 <= comprehension_weight <= 1:
            raise ValueError("comprehension_weight must be in [0, 1].")
        if observer_initialization not in ("random", "imagenet"):
            raise ValueError("observer_initialization must be random or imagenet.")
        for name, value, minimum in (("comprehension_warmup_steps", comprehension_warmup_steps, 0),
                                    ("training_steps", training_steps, 1), ("batch_size", batch_size, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        super().__init__(novelty_reference=novelty_reference)
        from .image_predictability import MaskedImageObserver

        self.comprehension_weight = float(comprehension_weight)
        self.comprehension_warmup_steps = comprehension_warmup_steps
        self.training_steps, self.batch_size = training_steps, batch_size
        self.observer = MaskedImageObserver(initialization=observer_initialization,
                                          learning_rate=learning_rate, cache_dir=cache_dir, device=device)
        self._initialization_seed = None
        self._replay = {}

    def _score_current(self, images, rng):
        """Measure with the live pre-update observer; preserve the baseline RNG order."""
        start = time.perf_counter()
        pixels, novelty, metadata = self._measure(images)
        available = metadata["reference_available"]
        position = None if available else rng.randrange(len(images))
        if self._initialization_seed is None:
            self._initialization_seed = rng.getrandbits(63)
            self.observer.initialize(self._initialization_seed)
        state_before = self.observer.state_hash()
        errors = self.observer.predict(images)[0] if available else np.zeros(len(images))
        if errors.shape != (len(images),) or not np.isfinite(errors).all() or np.any((errors < 0) | (errors > 1)):
            raise ValueError("Observer must return one finite masked MSE in [0, 1] per image.")
        comprehension = 1 - errors if available else np.zeros(len(images))
        novelty_ranks = _percentile_ranks(novelty)
        comprehension_ranks = _percentile_ranks(comprehension)
        warmup_complete = self._generation >= self.comprehension_warmup_steps
        weight = self.comprehension_weight if available and warmup_complete else 0.0
        scores = (1 - weight) * novelty_ranks + weight * comprehension_ranks
        if available:
            position = int(scores.argmax())
        metadata.update({
            "comprehension_available": available,
            "comprehension_warmup_complete": warmup_complete,
            "effective_comprehension_weight": weight,
            "measurement_available": [available, available],
            "masked_mse": errors.tolist() if available else [None] * len(images),
            "novelty_ranks": novelty_ranks.tolist(), "comprehension_ranks": comprehension_ranks.tolist(),
            "observer_initialization_seed": self._initialization_seed,
            "observer_state_before": state_before,
            "scoring_seconds": time.perf_counter() - start,
            "masked_inputs_scored": len(images) * 16 if available else 0,
        })
        return pixels, novelty, comprehension, scores, position, metadata

    def _update_observer(self, images, pixels, metadata, rng, position):
        replay_before = len(self._replay)
        for key, image in zip(metadata["image_hashes"], images, strict=True):
            if key not in self._replay:
                self._replay[key] = self.observer.prepare(image)
        seed = rng.getrandbits(63)
        start = time.perf_counter()
        update = self.observer.train(list(self._replay.values()), steps=self.training_steps,
                                     batch_size=self.batch_size, seed=seed)
        metadata.update({
            "observer_state_after": self.observer.state_hash(),
            "observer": self.observer.describe(), "replay_size_before": replay_before,
            "replay_size_after": len(self._replay), "update_seed": seed, "training": update,
            "training_seconds": time.perf_counter() - start,
        })
        self._remember(pixels, metadata["image_hashes"], position)

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        pixels, novelty, comprehension, scores, position, metadata = self._score_current(images, rng)
        self._update_observer(images, pixels, metadata, rng, position)
        return SelectionDecision(position=position, mode="greedy" if metadata["reference_available"] else "random",
            evaluation=Evaluation(np.column_stack((novelty, comprehension)),
                                  ("pixel_novelty", "comprehension"), metadata), scores=scores)

    def describe(self):
        return super().describe() | {
            "selection_strategy": "novelty-predictability", "comprehension_weight": self.comprehension_weight,
            "comprehension_warmup_steps": self.comprehension_warmup_steps,
            "score": "(1-weight)*novelty_rank + weight*comprehension_rank; weight zero during comprehension warmup; first grid neutral",
            "comprehension": "1 - masked MSE before current-grid training; initially unavailable",
            "observer": self.observer.describe(), "training_steps": self.training_steps,
            "batch_size": self.batch_size, "replay": "all distinct displayed images in first-seen order, including rejects",
        }


class _OffspringValueMixin:
    """Shared delayed targets, predictor replay and additive forecast selection.

    Concrete strategies supply current values, frozen targets, predictor context
    and seeds, and their end-of-grid update. This is internal implementation reuse;
    callers still configure one complete SelectionStrategy.
    """

    def __init__(self, *, gamma=1., warmup_targets=10, offspring_aggregation="max",
                 predictor_training_steps=10, predictor_batch_size=16, predictor_learning_rate=.001,
                 predictor_device="cpu", predictor_value_source="predictability", **value_options):
        if offspring_aggregation not in ("max", "mean"):
            raise ValueError("offspring_aggregation must be max or mean.")
        for name, value, minimum in (("warmup_targets", warmup_targets, 1),
                ("predictor_training_steps", predictor_training_steps, 1),
                ("predictor_batch_size", predictor_batch_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        if (isinstance(predictor_learning_rate, bool) or not isinstance(predictor_learning_rate, Real)
                or not np.isfinite(predictor_learning_rate) or predictor_learning_rate <= 0):
            raise ValueError("predictor_learning_rate must be positive and finite.")
        _validate_novelty_reference(value_options.get("novelty_reference", "previous-grid-mean"))
        from .offspring_prediction import OffspringValuePredictor
        self.predictor = OffspringValuePredictor(learning_rate=predictor_learning_rate,
            device=predictor_device, value_source=predictor_value_source,
            offspring_aggregation=offspring_aggregation)
        super().__init__(**value_options)
        self.offspring_aggregation = offspring_aggregation
        self.gamma, self.warmup_targets = float(gamma), warmup_targets
        self.predictor_training_steps, self.predictor_batch_size = predictor_training_steps, predictor_batch_size
        self._predictor_seed = None
        self._transitions = []
        self._pending = None
        self._expected_parent_hash = None

    def _receive_feedback(self, images, **measurement_options):
        if self._pending is None:
            return None, None, 0., 0.
        pending = self._pending
        start = time.perf_counter()
        observed = pending["target_context"].evaluate_offspring(images, **measurement_options)
        target_seconds = time.perf_counter() - start
        target = observed.metadata["offspring_value"]
        record = pending["record"]
        feedback = record | {
            "received_generation": self._generation, "target": target,
            "error": record["forecast"] - target,
            "absolute_error": abs(record["forecast"] - target),
            "squared_error": (record["forecast"] - target)**2,
            "running_mean_squared_error": (record["running_mean_forecast"] - target)**2,
            "current_value_squared_error": (record["current_value"] - target)**2,
            "child_names": list(observed.names), "child_values": observed.values.tolist(),
            "target_context": observed.metadata,
        }
        self._transitions.append({"inputs": pending["inputs"], "target": target,
                                  "origin_generation": record["origin_generation"]})
        # Release the previous target context before creating the next one.
        self._pending = None
        del pending
        seed = self._predictor_run_seed("update", self._generation)
        before = self.predictor.state_hash()
        start = time.perf_counter()
        update = self.predictor.train(self._transitions, steps=self.predictor_training_steps,
                                      batch_size=self.predictor_batch_size, seed=seed)
        training_seconds = time.perf_counter() - start
        training = update | {"seed": seed, "state_before": before, "state_after": self.predictor.state_hash()}
        return feedback, training, target_seconds, training_seconds

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        if len(images) != 9:
            raise ValueError("Expected nine RGB candidates: retained parent followed by eight children.")
        for image in images:
            validate_image(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.shape != images[0].shape:
                raise ValueError("Expected nine matching RGB image shapes.")
            if self._reference_image is not None and image.shape != self._reference_image.shape:
                raise ValueError("Image shape changed during a chronological run.")
        if self._expected_parent_hash is not None and _image_hash(images[0]) != self._expected_parent_hash:
            raise ValueError("Retained parent does not match the previous selected parent.")

        measured, received = self._measure_and_receive(images, rng)
        feedback, training, target_seconds, training_seconds = received
        pixels, novelty, comprehension, current, base_position, metadata = measured
        if self._predictor_seed is None:
            self._predictor_seed = self._predictor_run_seed("initialize", 0)
            self.predictor.initialize(self._predictor_seed)
        available = metadata["reference_available"]
        target_eligible = self._target_eligible(metadata)
        forecasts = np.zeros(9)
        contexts = inputs = None
        inference_seconds = snapshot_seconds = 0.
        if available:
            contexts = self._prediction_context(novelty, comprehension, current)
            inputs = self.predictor.prepare(images, self._reference_image, contexts)
            start = time.perf_counter()
            forecasts = np.asarray(self.predictor.predict(inputs), dtype=np.float64)
            inference_seconds = time.perf_counter() - start
            if forecasts.shape != (9,) or not np.isfinite(forecasts).all() or np.any((forecasts < 0) | (forecasts > 1)):
                raise ValueError("Predictor must return nine finite forecasts in [0, 1].")
        ready = len(self._transitions) >= self.warmup_targets
        used = bool(target_eligible and ready and self.gamma > 0)
        scores = current + self.gamma*forecasts if used else current.copy()
        position = int(scores.argmax()) if available else base_position
        predictor_state = self.predictor.state_hash()
        pending_record = None
        # Eligibility belongs to the parent-selection time, not child arrival.
        if target_eligible:
            start = time.perf_counter()
            frozen = self._freeze_target(novelty, comprehension, images[position])
            snapshot_seconds = time.perf_counter() - start
            pending_record = {
                "origin_generation": self._generation, "selected_position": position,
                "parent_image_hash": metadata["image_hashes"][position],
                "reference_image_hash": frozen.describe()["reference_image_hash"],
                "offspring_aggregation": self.offspring_aggregation,
                "novelty_reference": self.novelty_reference,
                "forecast": float(forecasts[position]), "forecast_used": used,
                "predictor_state_hash": predictor_state, "context": contexts[position].tolist(),
                "current_value": float(current[position]),
                "running_mean_forecast": float(np.mean([t["target"] for t in self._transitions])) if self._transitions else .5,
                "completed_targets_at_forecast": len(self._transitions),
                "target_context": frozen.describe(),
            }
            self._pending = {"record": pending_record, "inputs": inputs[position], "target_context": frozen}
        self._expected_parent_hash = metadata["image_hashes"][position]
        metadata["offspring"] = {
            "offspring_aggregation": self.offspring_aggregation,
            "forecasts": forecasts.tolist(), "forecast_available": bool(available), "forecast_used": used,
            "gamma": self.gamma, "warmup_targets": self.warmup_targets, "warmup_complete": ready,
            "completed_targets": len(self._transitions), "replay_size": len(self._transitions),
            "target_eligible": target_eligible,
            "current_values": current.tolist(), "contexts": contexts.tolist() if available else None,
            "base_position": base_position, "choice_changed": position != base_position,
            "initialization_seed": self._predictor_seed, "predictor_state": predictor_state,
            "feedback": feedback, "training": training, "pending": pending_record,
            "target_scoring_seconds": target_seconds, "predictor_training_seconds": training_seconds,
            "inference_seconds": inference_seconds, "snapshot_seconds": snapshot_seconds,
            **self._target_cost(feedback),
        }
        metadata["offspring"] = deepcopy(metadata["offspring"])
        self._finish_grid(images, pixels, metadata, rng, position)
        metadata["measurement_available"] += [self._current_value_available(metadata), bool(available)]
        return SelectionDecision(position, self._selection_mode(metadata),
            Evaluation(np.column_stack((novelty, comprehension, current, forecasts)),
                       ("pixel_novelty", self._quality_name, "current_value", "predicted_offspring_value"), metadata), scores)

    def describe(self):
        return super().describe() | {
            "score": "current_value + gamma*predicted_offspring_value after warmup; otherwise current_value",
            "offspring_aggregation": self.offspring_aggregation,
            "gamma": self.gamma, "warmup_targets": self.warmup_targets,
            "predictor": self.predictor.describe(), "predictor_training_steps": self.predictor_training_steps,
            "predictor_batch_size": self.predictor_batch_size,
            "lifecycle": "fresh instance; nine RGB candidates, retained selected parent first; chronological calls only",
        }


class OffspringValueSelectionStrategy(_OffspringValueMixin, NoveltyPredictabilitySelectionStrategy):
    """Predict max or mean child value under selection-time predictability and novelty."""

    _quality_name = "comprehension"

    def __init__(self, comprehension_weight=.5, *, comprehension_warmup_steps=10, gamma=1., warmup_targets=10,
                 offspring_aggregation="max",
                 predictor_training_steps=10, predictor_batch_size=16, predictor_learning_rate=.001,
                 observer_initialization="random", training_steps=20, batch_size=16,
                 learning_rate=.001, cache_dir=None, device="cpu", novelty_reference="previous-grid-mean"):
        if observer_initialization != "random":
            raise ValueError("offspring-value uses a scratch observer; observer_initialization must be random.")
        super().__init__(comprehension_weight=comprehension_weight, comprehension_warmup_steps=comprehension_warmup_steps,
            novelty_reference=novelty_reference,
            observer_initialization=observer_initialization, training_steps=training_steps, batch_size=batch_size,
            learning_rate=learning_rate, cache_dir=cache_dir, device=device, predictor_device=device,
            gamma=gamma, warmup_targets=warmup_targets, offspring_aggregation=offspring_aggregation,
            predictor_training_steps=predictor_training_steps,
            predictor_batch_size=predictor_batch_size, predictor_learning_rate=predictor_learning_rate)

    def _predictor_run_seed(self, purpose, generation):
        payload = f"offspring-value-v1:{self._initialization_seed}:{purpose}:{generation}"
        return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)

    def _measure_and_receive(self, images, rng):
        feedback = self._receive_feedback(images)
        return self._score_current(images, rng), feedback

    def _target_eligible(self, metadata):
        return bool(metadata["reference_available"] and metadata["comprehension_warmup_complete"])

    def _prediction_context(self, novelty, comprehension, current):
        shared = np.r_[np.sort(novelty), np.sort(comprehension),
                       np.log1p(self._generation*self.training_steps), np.log1p(len(self._replay))]
        return np.column_stack((novelty, comprehension, current, np.tile(shared, (9, 1))))

    def _freeze_target(self, novelty, comprehension, selected_parent):
        from .offspring_value import FrozenOffspringValue

        return FrozenOffspringValue(reference_image=self._reference_image,
            novelty_reference=novelty, comprehension_reference=comprehension, observer=self.observer,
            selected_parent=selected_parent, origin_generation=self._generation,
            comprehension_weight=self.comprehension_weight, offspring_aggregation=self.offspring_aggregation)

    def _target_cost(self, feedback):
        return {"target_masked_inputs_scored": 128 if feedback is not None else 0}

    def _finish_grid(self, images, pixels, metadata, rng, position):
        self._update_observer(images, pixels, metadata, rng, position)

    def _current_value_available(self, metadata):
        return bool(metadata["reference_available"])

    def _selection_mode(self, metadata):
        return "greedy" if metadata["reference_available"] else "random"

    def describe(self):
        return super().describe() | {
            "selection_strategy": "offspring-value", "version": 2,
            "predictor_replay": "completed transitions from parents selected after comprehension warmup, in chronological order; no deduplication or relabeling",
            "seed_derivation": "SHA256 ASCII offspring-value-v1:{observer_seed}:{initialize|update}:{generation}; first eight bytes big-endian masked to 63 bits",
            "target": f"{self.offspring_aggregation} value of eight actual children under selection-time frozen observer, configured reference image and nine fixed rank references",
            "missing_targets": "first grid and parents selected during comprehension warmup ineligible; final selection unobserved; max(S-max(comprehension_warmup_steps,1)-1,0) targets",
        }


class OffspringValueImageNetSelectionStrategy(_OffspringValueMixin, NoveltyImageNetSelectionStrategy):
    """Predict max or mean child value under fixed ImageNet confidence and novelty."""

    _quality_name = "imagenet_confidence"

    def __init__(self, comprehension_weight=.5, *, evaluator: "ImageNetEvaluator | None" = None,
                 gamma=1., warmup_targets=10, offspring_aggregation="max", predictor_training_steps=10, predictor_batch_size=16,
                 predictor_learning_rate=.001, device="cpu", novelty_reference="previous-grid-mean"):
        super().__init__(comprehension_weight=comprehension_weight, evaluator=evaluator,
            novelty_reference=novelty_reference,
            gamma=gamma, warmup_targets=warmup_targets, offspring_aggregation=offspring_aggregation,
            predictor_training_steps=predictor_training_steps,
            predictor_batch_size=predictor_batch_size, predictor_learning_rate=predictor_learning_rate,
            predictor_device=device, predictor_value_source="imagenet")
        self._seed_material = None

    def _predictor_run_seed(self, purpose, generation):
        payload = f"offspring-value-imagenet-v1:{self._seed_material}:{purpose}:{generation}"
        return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)

    def _measure_and_receive(self, images, rng):
        if self._seed_material is None:
            # Read the supplied run RNG without consuming selection draws.
            self._seed_material = hashlib.sha256(repr(rng.getstate()).encode("ascii")).hexdigest()
        measured = self._score_current(images, rng)
        return measured, self._receive_feedback(images, confidence=measured[2][1:])

    def _target_eligible(self, metadata):
        return bool(metadata["reference_available"])

    def _prediction_context(self, novelty, confidence, current):
        shared = np.r_[np.sort(novelty), np.sort(confidence)]
        return np.column_stack((novelty, confidence, current, np.tile(shared, (9, 1))))

    def _freeze_target(self, novelty, confidence, selected_parent):
        from .offspring_value import FrozenImageNetOffspringValue

        return FrozenImageNetOffspringValue(reference_image=self._reference_image,
            novelty_reference=novelty, confidence_reference=confidence, evaluator_metadata=self.evaluator.describe(),
            selected_parent=selected_parent, origin_generation=self._generation,
            comprehension_weight=self.comprehension_weight, offspring_aggregation=self.offspring_aggregation)

    def _target_cost(self, feedback):
        return {"target_classifier_images_scored": 0,
                "target_classifier_measurements_reused": 8 if feedback is not None else 0}

    def _finish_grid(self, images, pixels, metadata, rng, position):
        self._remember(pixels, metadata["image_hashes"], position)

    def _current_value_available(self, metadata):
        return True

    def _selection_mode(self, metadata):
        return "random" if not metadata["reference_available"] and self.comprehension_weight == 0 else "greedy"

    def describe(self):
        return super().describe() | {
            "selection_strategy": "offspring-value-imagenet", "version": 2,
            "predictor_replay": "all completed eligible selected-parent transitions in chronological order; no deduplication or relabeling",
            "seed_derivation": "seed_material=SHA256 ASCII repr(initial selection RNG state); SHA256 ASCII offspring-value-imagenet-v1:{seed_material}:{initialize|update}:{generation}; first eight bytes big-endian masked to 63 bits; no RNG draws",
            "target": f"{self.offspring_aggregation} value of eight actual children under frozen classifier and selection-time configured reference image and nine fixed rank references",
            "missing_targets": "first grid ineligible; final selection unobserved; max(S-2,0) targets",
        }


class ImageNetSelectionStrategy:
    """Select by maximum class probability, with optional random exploration.

    Epsilon=0 is greedy; epsilon=1 always selects uniformly. Every candidate is
    evaluated on every choice, including exploratory choices. Exact greedy
    ties favour the first candidate (the retained parent after initialization).

    By default construct one ImageNetEvaluator and reuse it. Supply a loaded
    evaluator to share its classifier across notebook experiments or configure
    its model, checkpoint, device, batch size and cache directory.
    """

    def __init__(self, epsilon: float = 0.0, *, evaluator: "ImageNetEvaluator | None" = None):
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be a finite number between 0 and 1.")
        self.epsilon = float(epsilon)
        if evaluator is None:
            from .imagenet import ImageNetEvaluator

            evaluator = ImageNetEvaluator()
        self.evaluator = evaluator

    def choose(self, images: Sequence[ImageArray], *, rng: Random) -> SelectionDecision:
        if not images:
            raise ValueError("Cannot select from an empty candidate grid.")
        evaluation = self.evaluator.evaluate(images)
        if len(evaluation.values) != len(images):
            raise ValueError("Evaluation must contain one row per displayed candidate.")
        scores = evaluation.values.max(axis=1)
        explore = self.epsilon == 1 or (self.epsilon > 0 and rng.random() < self.epsilon)
        position = rng.randrange(len(images)) if explore else int(scores.argmax())
        return SelectionDecision(position=position, mode="random" if explore else "greedy",
                                 evaluation=evaluation, scores=scores)

    def describe(self) -> dict:
        return {
            "selection_strategy": "imagenet", "epsilon": self.epsilon,
            "score": "maximum ImageNet class probability per candidate",
            "rule": "uniform random with probability epsilon; otherwise first maximum score",
            "ties": "first candidate; retained parent is first after initialization",
            "evaluator": self.evaluator.describe(),
        }
