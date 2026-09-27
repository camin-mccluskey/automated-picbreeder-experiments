"""Run and record automated choices in the shared nine-image breeding loop."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .breeding import BreedingSession
from .cppn import png_bytes, render
from .evaluation import ImageEvaluator
from .selection import Selector
from .persistence import SessionWriter


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 7
    steps: int = 100
    size: int = 96
    mutation_strength: float = 0.2
    topology: bool = True
    checkpoint_every: int = 10

    def __post_init__(self):
        for name in ("steps", "size", "checkpoint_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.size < 2:
            raise ValueError("size must be at least 2.")
        if not np.isfinite(self.mutation_strength) or self.mutation_strength < 0:
            raise ValueError("mutation_strength must be finite and non-negative.")


def _contact_sheet(images, candidate_ids, selected_position, evaluation):
    """An inspection view; scoring always uses the original image arrays."""
    sheet = Image.new("RGB", (660, 3 * 222), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (pixels, key) in enumerate(zip(images, candidate_ids)):
        x, y = (i % 3) * 220, (i // 3) * 222
        sheet.paste(Image.fromarray(pixels).convert("RGB").resize((160, 160)), (x + 8, y + 8))
        column = int(evaluation.values[i].argmax())
        label = evaluation.names[column]
        draw.text((x + 8, y + 173), f"{i + 1}. genome #{key}" + (" SELECTED" if i == selected_position else ""), fill="black")
        draw.text((x + 8, y + 189), f"{label[:27]} [{column}]", fill="black")
        draw.text((x + 8, y + 203), f"max score: {evaluation.values[i, column]:.6f}", fill="black")
        if i == selected_position:
            draw.rectangle((x + 2, y + 2, x + 216, y + 219), outline="#267744", width=3)
    return sheet


def run_experiment(
    evaluator: ImageEvaluator, selector: Selector, output_dir: str | Path,
    settings: ExperimentSettings = ExperimentSettings(), *, progress=print,
) -> dict:
    """Evaluate nine candidates, select one, breed eight children, repeat.

    A step includes one decision, starting with the initial nine roots. Every
    displayed image is rescored, including the parent: 9 evaluations per step,
    and 9 + 8*(steps-1) distinct candidate genomes. This deliberately avoids a
    stationary-score cache so other evaluators can score the current choice set.
    """
    writer = SessionWriter(output_dir, size=settings.size, selector=selector.describe(),
                           settings=asdict(settings))
    directory = writer.directory
    (directory / "grids").mkdir()
    session = BreedingSession(seed=settings.seed)
    names = None
    start = time.perf_counter()
    decisions = 0
    for step in range(settings.steps):
        if step:
            session.evolve(strength=settings.mutation_strength, topology=settings.topology)
        ids = session.candidates.copy()
        images = [render(session.genomes[key], session.config, settings.size) for key in ids]
        # Preserve the current choice set even if evaluation or selection fails.
        writer.save(session, images={key: png_bytes(pixels) for key, pixels in zip(ids, images)})
        evaluation = evaluator.evaluate(images)
        if evaluation.values.shape[0] != 9:
            raise ValueError("The evaluator must return exactly one row per displayed candidate.")
        if names is None:
            names = evaluation.names
        elif names != evaluation.names:
            raise ValueError("Evaluator column names/order changed during the run.")
        position = selector.select(evaluation)
        session.select(position, evaluation=evaluation)
        best_classes = evaluation.values.argmax(axis=1)
        decisions += 1
        summary = {
            "status": "complete" if decisions == settings.steps else "running",
            "decisions": decisions, "evaluated_images": decisions * 9,
            "unique_candidates": len(session.genomes), "selected_id": session.selected_id,
            "selected_class_index": int(best_classes[position]),
            "selected_class_name": names[int(best_classes[position])],
            "selected_score": float(evaluation.values[position].max()),
            "elapsed_seconds": time.perf_counter() - start,
        }
        _contact_sheet(images, ids, position, evaluation).save(directory / "grids" / f"{step:06d}.png")
        checkpoint = step if decisions % settings.checkpoint_every == 0 or decisions == settings.steps else None
        writer.save(session, summary=summary, checkpoint=checkpoint)
        if progress is not None:
            progress(f"Step {decisions}/{settings.steps}: selected #{session.selected_id}, "
                     f"{summary['selected_class_name']} "
                     f"{summary['selected_score']:.6f}; {9 * decisions} image evaluations")
    return summary
