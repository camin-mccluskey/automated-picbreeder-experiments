# Automated Picbreeder

A small research project exploring how CPPNs generate images and how selection
shapes their evolution. Human and automated runs share one loop: select one of
nine images, retain its network unchanged, and generate eight mutated children.
The project provides breeding, image diagnostics and network inspection tools.
It does not yet establish unified factored representations (UFR): independently
controllable image properties and shared computation underlying repeated structure.

## Start

Requires Python 3.13+ and `uv`. From the repository root:

```sh
uv sync --locked
uv run jupyter lab notebooks/01_cppn_selection.ipynb
```

Select an image, click **Mutate selected**, and use **Save session** before closing
the kernel. [Usage and notebooks](docs/usage.md) explains exploration, model
inspection, saved sessions, interpretation, rendering and development checks.

## Automated selection experiments

Each guide explains the selection rule, its limitations, runnable examples and
all applicable options. Use a fresh strategy and output directory for each run.

| Strategy guide | Selection rule |
| --- | --- |
| [Random](docs/strategies/random.md) | Uniform choice, no image evaluation |
| [Novelty](docs/strategies/novelty.md) | Pixel distance from the previous grid mean or previous selected parent |
| [ImageNet](docs/strategies/imagenet.md) | Maximum class confidence, with optional random exploration |
| [Novelty + ImageNet](docs/strategies/novelty-imagenet.md) | Combine novelty and confidence ranks |
| [Novelty + predictability](docs/strategies/novelty-predictability.md) | Combine novelty and learned masked-pixel accuracy ranks |
| [Offspring value](docs/strategies/offspring-value.md) | Add predicted best-child or mean-child value using patch predictability |
| [Offspring value + ImageNet](docs/strategies/offspring-value-imagenet.md) | Add the same offspring forecast using frozen classification confidence |

```sh
uv run python experiments/run_selection.py random --steps 100 --seed 7
uv run python experiments/run_selection.py --help
```

Put the strategy before all options. Model strategies require
`uv run --extra imagenet`; checkpoints may download on first use.
[Run options and reproducibility](docs/run-options.md) covers seeds, batches,
rendering and budgets. [Shared model options](docs/strategies/options.md) covers
observer, classifier and predictor settings.

Every export includes an offline `index.html` viewer. To collect saved results:

```sh
uv run python experiments/view_results.py runs/
```

[Batch experiments and metrics](docs/batch-experiments.md) explains diagnostics,
failures, human-session comparisons and sharing reports with their image folders.
The [experiment brief](docs/experiment-brief.md) explains the scientific objective
and limits of the current tools.

Project code is [MIT licensed](LICENSE). The bundled Picbreeder skull reference
retains its [Apache-2.0 licence and attribution](src/automated_picbreeder/reference_data/skull/README.md).
