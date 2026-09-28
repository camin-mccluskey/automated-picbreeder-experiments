"""Run a selection strategy in the shared nine-image breeding loop."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from random import Random
import time

import numpy as np

from .breeding import BreedingSession
from .cppn import png_bytes, render
from .experiment_reporting import contact_sheet, progress_message
from .persistence import SessionWriter
from .selection_strategies import SelectionStrategy


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 7
    steps: int = 100
    size: int = 96
    mutation_strength: float = 0.2
    topology: bool = True
    checkpoint_every: int = 10
    selection_seed: int | None = None

    def __post_init__(self):
        for name in ("seed", "selection_seed"):
            value = getattr(self, name)
            if name == "selection_seed" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        for name in ("steps", "size", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.size < 2:
            raise ValueError("size must be at least 2.")
        if not np.isfinite(self.mutation_strength) or self.mutation_strength < 0:
            raise ValueError("mutation_strength must be finite and non-negative.")

    @property
    def resolved_selection_seed(self) -> int:
        """A stable, separate seed; never use Python's process-dependent hash()."""
        if self.selection_seed is not None:
            return self.selection_seed
        digest = sha256(f"automated-picbreeder:selection:{self.seed}".encode()).digest()
        return int.from_bytes(digest[:8], "big")


def run_experiment(
    *, selection_strategy: SelectionStrategy, output_dir: str | Path,
    settings: ExperimentSettings = ExperimentSettings(), progress=print,
) -> dict:
    """Choose from nine images, breed eight children, and record every decision.

    The initial grid counts as one decision. S decisions present 9*S candidates
    and generate 9+8*(S-1) genomes. Evaluation is owned by the selection strategy:
    random selection evaluates none, while ImageNet evaluates all nine each turn.
    A fresh selection RNG is created per run, independent of breeding's RNG.
    """
    run_settings = asdict(settings) | {"selection_seed": settings.resolved_selection_seed}
    writer = SessionWriter(output_dir, size=settings.size,
                           selection_strategy=selection_strategy.describe(), settings=run_settings)
    directory = writer.directory
    (directory / "grids").mkdir()
    session = BreedingSession(seed=settings.seed)
    selection_rng = Random(settings.resolved_selection_seed)
    names = None
    start = time.perf_counter()
    evaluated_images = candidate_presentations = 0
    for step in range(settings.steps):
        if step:
            session.evolve(strength=settings.mutation_strength, topology=settings.topology)
        ids = session.candidates.copy()
        images = [render(session.genomes[key], session.config, settings.size) for key in ids]
        # Preserve the current grid even if the selection strategy fails.
        writer.save(session, images={key: png_bytes(pixels) for key, pixels in zip(ids, images)})
        decision = selection_strategy.choose(images, rng=selection_rng)
        decision.validate(len(ids))
        if decision.evaluation is not None:
            if names is None:
                names = decision.evaluation.names
            elif names != decision.evaluation.names:
                raise ValueError("Evaluator column names/order changed during the run.")
            evaluated_images += len(ids)
        session.select(decision.position, decision=decision)
        candidate_presentations += len(ids)
        decisions = step + 1
        summary = {
            "status": "complete" if decisions == settings.steps else "running",
            "decisions": decisions, "candidate_presentations": candidate_presentations,
            "evaluated_images": evaluated_images, "unique_candidates": len(session.genomes),
            "selected_id": session.selected_id, "selection_mode": decision.mode,
            "selected_score": None if decision.scores is None else float(decision.scores[decision.position]),
            "elapsed_seconds": time.perf_counter() - start,
        }
        contact_sheet(images, ids, decision).save(directory / "grids" / f"{step:06d}.png")
        checkpoint = step if decisions % settings.checkpoint_every == 0 or decisions == settings.steps else None
        writer.save(session, summary=summary, checkpoint=checkpoint)
        if progress is not None:
            progress(progress_message(summary, settings.steps))
    return summary
