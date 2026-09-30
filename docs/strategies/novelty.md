# Pixel novelty selection

[All strategies](../../README.md#automated-selection-experiments)

Choose uniformly on the first grid; subsequently choose the first candidate
maximising full-resolution pixel MSE to the configured reference. The default
reference is the previous displayed grid's mean image. This measures local visual
change; an old image may be rewarded again, including an unchanged retained parent.

```sh
uv run python experiments/run_selection.py novelty --steps 100 --seed 7
uv run python experiments/run_selection.py novelty --novelty-reference previous-parent --steps 100 --seed 7
uv run python experiments/run_selection.py novelty --help
```

[All run options](../run-options.md) apply. The only strategy-specific option is
[`--novelty-reference`](options.md#novelty-reference), default
`previous-grid-mean`, alternative `previous-parent`. It requires no Torch.
Python: `NoveltySelectionStrategy(novelty_reference="previous-grid-mean")`.
Construct a fresh instance for every run or independent inspection.

Records include raw distances, percentile ranks, the reference, pixel hashes and
previously-seen flags. Initial distances are unavailable and ranks neutral;
flags do not affect the choice. Previous-parent mode gives the retained parent
zero raw novelty, but it can still win if every candidate is identical.
Historical nearest-image comparison is deferred. Notebook 05 inspects novelty
and references. [Shared metrics](../batch-experiments.md#shared-measurements)
keep previous-grid `display_novelty` distinct from configurable strategy novelty.
