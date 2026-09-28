"""One session format and writer for human and automated breeding runs."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

from .cppn import CONFIG_PATH, genome_to_dict, png_bytes, render


def write_json(path, data):
    """Replace a snapshot only after its complete JSON has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_snapshot(directory):
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "src" / "automated_picbreeder").glob("*.py")),
             *sorted((root / "src" / "automated_picbreeder").glob("*.cfg")),
             *sorted((root / "experiments").glob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    hashes = {}
    for path in paths:
        if path.is_file():
            relative = path.relative_to(root)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


class SessionWriter:
    """Own a new run directory; save version-2 sessions for any selection strategy.

    Image paths in session and checkpoint documents are relative to the run
    directory. Snapshots are audit records, not resumable RNG/innovation state.
    Genomes are immutable after registration, so their PNGs are written once.
    """

    def __init__(self, directory, *, size, selection_strategy=None, settings=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "images").mkdir()
        self.size = size
        self.image_records = {}
        self.metadata = {
            "export_started_at": datetime.now(timezone.utc).isoformat(),
            "selection_strategy": selection_strategy, "settings": settings,
            "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": _source_snapshot(self.directory),
            "checkpoint_scope": "audit snapshots, not resumable sessions",
        }

    def save(self, session, *, images=None, summary=None, checkpoint=None):
        """Save all genomes/events, optionally using already rendered PNG bytes."""
        for key, genome in session.genomes.items():
            if key not in self.image_records:
                image = images.get(key) if images is not None else None
                if image is None:
                    image = png_bytes(render(genome, session.config, self.size))
                relative = f"images/{key:08d}.png"
                (self.directory / relative).write_bytes(image)
                self.image_records[key] = {
                    "image": relative, "image_sha256": hashlib.sha256(image).hexdigest(),
                }
        config_path = self.directory / "cppn.cfg"
        session.config.save(str(config_path))
        data = {
            "format_version": 2, "seed": session.seed, "size": self.size,
            "rendering": "x,y in [-1,1]; y increases down; r=hypot(x,y); mutable H,S,B outputs; "
                         "H=(h+1)%1; S=clip(s,0,1); B=clip(abs(b),0,1); "
                         "colorsys HSV to RGB; int(255*clip(channel,0,1)+0.5)",
            "output_encoding": "HSB",
            "output_mapping": session.config.output_mapping,
            "image_mode": "RGB",
            "versions": {p: version(p) for p in ("automated-picbreeder", "neat-python", "numpy", "pillow", "ipywidgets")},
            "config": config_path.read_text(), "initial_config": CONFIG_PATH.read_text(),
            "genomes": [genome_to_dict(g) | {"parent": session.parents[k]} | self.image_records[k]
                        for k, g in session.genomes.items()],
            "events": session.events, "displayed": session.candidates, "selected": session.selected_id,
            "metadata": self.metadata, "summary": summary,
        }
        selected_path = self.directory / "selected.png"
        if session.selected_id is None:
            selected_path.unlink(missing_ok=True)
        else:
            shutil.copyfile(self.directory / self.image_records[session.selected_id]["image"], selected_path)
        write_json(self.directory / "session.json", data)
        if checkpoint is not None:
            checkpoints = self.directory / "checkpoints"
            checkpoints.mkdir(exist_ok=True)
            write_json(checkpoints / f"{checkpoint:06d}.json", data)
        return self.directory
