# Usage, saved sessions and interpretation

[Project overview](../README.md) · [Selection guides](../README.md#automated-selection-experiments) · [Run options](run-options.md)

Run terminal commands from the repository root.

## Start

```sh
uv sync --locked
uv run jupyter lab notebooks/01_cppn_selection.ipynb
```

Run the notebook cells in order. Select an image with the button underneath it,
then click **Mutate selected**. The selected parent stays in position 1 and eight
offspring appear alongside it. **Back** returns to an earlier grid so you can
branch; **New random images** creates new roots without discarding session history.
The kernel must remain running for the buttons to work.

If using VS Code, choose this project's `.venv/bin/python` notebook kernel and
enable the Python/Jupyter extensions. If widgets display only as text, try the
JupyterLab command above; the required widget extensions are included in the lockfile.

**Save session** writes all candidate genomes (including unselected alternatives),
parent IDs, ordered displays and decisions, the configuration, package versions,
and PNGs of every generated image to `runs/`, using the shared session format
described below. The notebook shows how to render an
exported genome again. Exports are inspectable records, not resumable evolution
checkpoints. Sessions remain in memory until saved; save before restarting the kernel.

## Small experiments

1. Choose a pattern and generate several rounds. Which image properties persist?
2. Turn off **Allow structure / activation changes**. Compare mutation σ = 0.05
   with σ = 0.5. This changes the standard deviation of weight and bias perturbations;
   it does not directly set image difference.
3. Turn structure changes back on and inspect hidden-node and enabled-connection
   counts. A structural mutation is probabilistic, so counts need not change each time.
4. Use the notebook's single-weight sweep to separate one intervention from the
   many simultaneous changes in an offspring.

These are observations about rendering and mutation, not evidence of semantic
factorisation or shared computation.

## Inspect measurements and selection strategies

```sh
uv sync --locked --extra imagenet
uv run --extra imagenet jupyter lab notebooks/02_image_evaluation.ipynb
```

The `imagenet` extra keeps PyTorch optional for the original playground and for
random and pixel-novelty experiments. Observer and predictor strategies also
require this extra. Include `--extra imagenet` in `uv run` commands that
need it, so uv retains these optional dependencies in the environment.

The notebook renders nine CPPNs, shows the classifier's top labels, inspects its
actual input crop, and demonstrates a simple contrast evaluator. It then compares
selection strategies on the same grid and runs a short breeding experiment.
Contrast is an interface example, not a proposed measure of interestingness.

```python
from automated_picbreeder.evaluation import ImageEvaluator
from automated_picbreeder.imagenet import ImageNetEvaluator

evaluator: ImageEvaluator = ImageNetEvaluator(cache_dir=".cache/imagenet")
result = evaluator.evaluate(images)  # list of uint8 grayscale or RGB arrays
result.values                      # shape (number of images, number of measurements)
result.names                       # names in column order
result.metadata                    # checkpoint, preprocessing, device and versions
```

An evaluator is a helper for obtaining measurements; it does not choose parents
or mutate genomes. `evaluate(images) -> Evaluation` returns finite measurements
with rows in input order, including duplicates. Column position identifies each
measurement because display names can repeat. Measurements need not be probabilities.

**A selection strategy is the experiment's interchangeable component.** It owns
any evaluation, scoring and final choice. The ImageNet selection strategy reuses
this evaluator internally; the random selection strategy needs no evaluator.
There are no separate scorer or selector arguments to the experiment runner.

The adapter defaults to **ResNet-18 / IMAGENET1K_V1**, with all parameters frozen,
evaluation mode and gradient-free inference. It returns all 1,000 softmax class
scores, not just a top label. The first load downloads ~45 MB of official weights;
later loads use the cache. Package versions are locked in `uv.lock`; result
metadata records the concrete weight enum, checkpoint SHA-256 and preprocessing.
`DEFAULT` weight aliases are rejected to avoid silently changing checkpoints.
CPU is the default; `device="mps"` or `device="cuda"` explicitly opts into an
available accelerator. Changing model, device, library version or batching can
change numerical results; fix these before main runs.

CPPNs render as RGB from HSB outputs. The classifier adapter also accepts
grayscale arrays and replicates them across RGB channels. The default checkpoint's official transform
resizes the shorter side to 256, centre-crops to 224, rescales pixels and applies
ImageNet channel normalization. **This crops the image edges**; the notebook shows
what is retained. No per-image min/max normalization is applied. Inputs must be
uint8 H×W or H×W×3 arrays; float ranges and other channel layouts are rejected.

ImageNet is the training dataset/category set; ResNet-18 is the default classifier. Its
scores are not validated naturalness, novelty or interestingness measures, and
abstract CPPNs are outside its ordinary natural-image setting. These scores are an
exploratory selection signal.

### Configure experiments in Python or notebooks

```python
from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.selection_strategies import RandomSelectionStrategy

summary = run_experiment(
    selection_strategy=RandomSelectionStrategy(),
    output_dir="runs/my-random-run",  # Must be a new directory.
    settings=ExperimentSettings(seed=7, steps=3),
)
```

To use ImageNet, pass `ImageNetSelectionStrategy()` or
`ImageNetSelectionStrategy(epsilon=0.1)` as `selection_strategy`. Its default
constructor loads a classifier once. To configure the device, cache or model,
or share one loaded classifier across experiments:

```python
from automated_picbreeder.imagenet import ImageNetEvaluator
from automated_picbreeder.selection_strategies import ImageNetSelectionStrategy

classifier = ImageNetEvaluator(
    model_name="resnet18", weights="IMAGENET1K_V1",
    device="cpu", cache_dir=".cache/imagenet",
)
greedy = ImageNetSelectionStrategy(evaluator=classifier)
exploring = ImageNetSelectionStrategy(epsilon=0.1, evaluator=classifier)
```

For a new selection strategy, implement `choose(images, *, rng) -> SelectionDecision`
and `describe() -> dict`. The strategy receives the ordered image arrays and a
run-owned random generator. Return a zero-based `position`, a `mode` identifying
the choice operation, and optional `evaluation` and per-candidate `scores`.
`describe()` supplies JSON-compatible configuration. Preserve the input images
and use the supplied RNG; breeding and persistence remain the runner's concern.
Evaluators remain available for standalone inspection, but are not runner arguments.

### Saved sessions

Default outputs go into new directories named
`runs/<strategy>-selection-<timestamp>-seed<seed>/`. Existing directories
are never overwritten. Both human and automated runs use `SessionWriter` and
version-2 `session.json` records:

- `session.json`: configuration, rendering, versions, all genomes and ancestry,
  image paths/hashes, ordered displays, events, selection, metadata and summary.
- `images/<8-digit-genome-id>.png`: every generated candidate, including rejected
  alternatives and discarded branches.
- `cppn.cfg`, `source/`: configuration and a hashed package-source snapshot.
  Exports from a checkout also include CLI scripts, package metadata, the licence
  and dependency lockfile.
- `selected.png`: a convenience copy of the current selection.
- Both interfaces export `metrics.json`, `metrics.csv`, `index.html`, `grids/`
  and `performance.json`. Human timing includes deliberation and idle time.
- Automated runs also save `checkpoints/`: audit snapshots, not resumable sessions.
  Grids distinguish selection mode and preference scores from raw measurements.

Every selection event records its `position`, `genome`, ordered `displayed` IDs,
`evaluation` and `decision`:

| Choice | `evaluation` | `decision` |
| --- | --- | --- |
| Human | `null` | `null` |
| Pure random | `null` | `mode: "random"`, `scores: null` |
| Pixel novelty | Raw distances, configured reference mode, reference generation (zero-based; initially null), ordered pixel hashes and previously-seen flags | First `"random"`, then `"greedy"`; percentile-rank scores |
| Novelty plus ImageNet | Raw novelty/confidence, both ranks and full classifier provenance | Weighted novelty/confidence rank score |
| Novelty plus predictability | Raw novelty/comprehension, pre-update errors, ranks and observer learning records | First `"random"`, then `"greedy"`; weighted rank scores |
| Offspring value | Current-image measurements, frozen child targets, original forecasts, delayed errors and predictor training records | Current-image value plus the weighted forecast after warm-up |
| ImageNet | Full measurement values, names and provenance | `mode: "greedy"` or `"random"`, one maximum-class score per candidate |
| VLM | `null` | `mode: "vlm"`, `scores: null`, metadata with selected index/reason, request settings, responses and API costs |
| VLM + scratchpad | `null` | `mode: "vlm-scratchpad"`, `scores: null`, VLM metadata plus `scratchpad_before` and the accepted replacement `scratchpad` |

Measurements and scores belong to decisions, so repeated parent evaluations are
recorded separately. Measurement rows follow `displayed`; column positions follow
`names`. `metadata.selection_strategy` describes the selection strategy. Human
exports record the seed, size and current controls in `metadata.settings`, with
mutation settings retained per evolution event; their `summary` is null. Automated summaries include
candidate presentations, evaluated images, unique candidates, selected score
(null when unscored) and selection mode.

Snapshots are written before calling the selection strategy and after recording
its decision. A strategy failure leaves the current grid and earlier decisions
available, with a null summary. Image paths in sessions and checkpoints are
relative to the run directory.
VLM API failures additionally save `selection_failure.json` with the failed
grid's request diagnostics. The [VLM guide](strategies/vlm.md) explains setup,
structured selection, costs and remote replay limitations.

Existing version-2 HSB sessions remain readable by the interpretation notebooks,
including older records with `metadata.selector` and no `decision` field. They
are not migrated or rewritten. The metrics exporter and viewer require current strategy and decision metadata;
older records can still be inspected with the network loader.

This remains a local single-parent experiment. It does not implement the shared
archives, branching or critic agents in [Sakana's AI Picbreeder](https://pub.sakana.ai/picbreeder-vlm/).
Class confidence is a proposed preference proxy, not a validated naturalness or
interestingness metric. These selection strategies are available for exploration;
no comparison result or frozen experimental protocol is claimed.

## Interpret a saved network

```sh
uv run jupyter lab notebooks/03_cppn_interpretability.ipynb
```

Save a human selection first, or use an automated run's `session.json`. Set
`SESSION_PATH` in notebook 03; it defaults to the genome selected at export.
An explicit `GENOME_ID` can select another candidate. The loader checks the saved
PNG hash and reproduces its pixels using the embedded configuration before
inspection. It supports the current version-2 HSB format.

The independent `interpretability.py` module loads networks, traces NEAT's native
evaluator, changes individual weights/biases, disables connections, clamps node
activations and runs independent parameter sweeps. `interpretability_notebook.py`
supplies the graph, selected-connection activation maps, original/modified/difference
images and widget controls. Selecting a weight shows its source activation and the
destination's original/modified activations, with descriptive neuron labels and
original/current weights. Neither requires a running breeding session or classifier.
All interventions start from the original genome and operate on copies.

Activation-map scales remain fixed to each node's baseline range; values beyond
that range saturate the display, with numerical ranges shown. Node clamps replace
post-activation values at every pixel. A zero clamp is not necessarily neutral.
The graph distinguishes evaluated computation from unused and disabled genes.

**Save experiment** writes a new directory under `runs/interpretability/`, including
the baseline genome, configuration, source identity, exact interventions, observation
and PNGs. `replay_experiment(directory)` recomputes and checks those images without
requiring the original run. These records support exploratory causal tests; they
are not UFR scores or a frozen comparison protocol. Shared influence on two regions
alone does not establish that their structure has a shared representation.

### Original Picbreeder skull

```sh
uv run jupyter lab notebooks/04_picbreeder_skull.ipynb
```

Notebook 04 loads the authors' bundled **human-evolved skull CPPN**, reproduces
the published mouth-opening and eye-winking sweeps, then opens the same inspector
with presets for mouth opening, eye winking, eye width and jaw width. It needs no
ImageNet, JAX, downloading or retraining. The authors' layered representation is a
computation-preserving conversion of the evolved graph; their SGD-trained imitation
is a separate model. Our reference viewer uses the compact original graph, with
verified mappings from the published parameter IDs to its original connections.

`picbreeder_reference.py` preserves the reference model's activation functions,
scaled radius and explicit bias input independently of our breeding settings.
The tests compare baseline/intervention activations against the published layered
weights. The standalone published PNG has a small unresolved mismatch (mean
0.63/255, maximum 12/255 at 256×256), documented in the notebook and the
[reference provenance](../src/automated_picbreeder/reference_data/skull/README.md).
Saved reference experiments are self-contained and replay through the common API.

## Implementation choices

`cppn.cfg` explicitly declares `[CPPNRendering] output_mapping = picbreeder_hsb_v1`.
This is the only supported mapping. Configuration loading and rendering reject
unknown mappings; exported configurations and `session.json` preserve the setting.


- Three inputs per pixel: x and y in [-1, 1], plus Euclidean radius in [0, √2].
  y increases down the image; neural biases supply constant offsets.
- Three mutable outputs, ordered hue, saturation and brightness (HSB/HSV).
  Following Picbreeder-VLM, H = (h + 1) % 1, S = clip(s, 0, 1), and
  B = clip(abs(b), 0, 1). Python colorsys converts HSV to RGB; channels are
  rounded to bytes with int(255 * channel + 0.5). Output and hidden activations
  can be sine, Gaussian, tanh, sigmoid, identity or absolute value. NEAT-Python's built-in
  scaling applies (e.g. its sine is sin(clip(5z, -60, 60))).
- Two initial hidden nodes; initially fully connected with direct input-output links.
- Images are evaluated at 96 × 96 by the native `neat.nn.FeedForwardNetwork`.
  No per-image normalization or image filtering. Flat images and duplicate outputs
  are retained. The UI displays the images at 160 × 160.
- Each step clones one parent and applies NEAT-Python mutation. No crossover,
  speciation, or fitness-based population selection: this is **not full NEAT**.
- Disabling structure/activation changes freezes connections' enabled states,
  graph structure and activation functions, leaving weight/bias perturbations.
- Seeds isolate NEAT's global random state for a single-threaded notebook session.
  The same seed and action sequence reproduce the candidates with locked dependencies.
  Back restores the earlier choices, but does not rewind randomness: the next click
  produces a fresh branch. Mutation σ = 0 only guarantees unchanged images when
  structure/activation changes are also disabled.

The existing `neat-python==2.0.0` dependency supplies the genome and mutation
implementation. This is a small learning prototype, not a replication of Picbreeder
or Innovation Engines. Those systems' rendering, initialization and mutation
choices differ from our implementation. We match Picbreeder-VLM’s mapping from
HSB network outputs to RGB pixels, while retaining our activation-function
definitions and mutation/reproduction rules. We do not adopt its extra output
transforms, custom mutation operators or colour/brightness subnetworks.

## Files and checks

- `notebooks/01_cppn_selection.ipynb`: guided, runnable first experiment.
- `notebooks/02_image_evaluation.ipynb`: classifier inspection, selection-strategy comparison and a short run.
- `notebooks/07_offspring_value.ipynb`: offspring targets, online forecasts, ancestry and replay inspection.
- `notebooks/05_novelty_predictability.ipynb`: pixel novelty and pre-update masked prediction inspection for both observer initializations.
- `src/automated_picbreeder/cppn.py`: native NEAT genomes, mutation, rendering and JSON conversion.
- `src/automated_picbreeder/cppn.cfg`: explicit starting configuration.
- `src/automated_picbreeder/notebook.py`: human selection interface.
- `src/automated_picbreeder/persistence.py`: shared session metadata, schema and file writer.
- `src/automated_picbreeder/evaluation.py`: generic batch result and evaluator protocol.
- `src/automated_picbreeder/imagenet.py`: optional frozen Torchvision adapter.
- `src/automated_picbreeder/breeding.py`: shared human/automated breeding state.
- `src/automated_picbreeder/selection_strategies.py`: selection strategies and decision records.
- `src/automated_picbreeder/experiment.py`: the shared automated runner and settings.
- `src/automated_picbreeder/experiment_reporting.py`: saved grids and progress formatting.
- `experiments/run_selection.py`: entry point for strategy subcommands.
- `src/automated_picbreeder/selection_cli.py`: shared CLI option groups, strategy construction and dispatch. To add a strategy, define its configure function and register it in `COMMANDS`; keep model construction inside its builder.

```sh
uv run pytest
uv run --extra imagenet pytest  # also run offline classifier adapter checks
```

Classifier unit tests use controlled logits and the real preprocessing transform;
they never download weights. To check the downloaded model itself, run notebook 02.


## References

- [Scientific objective and limitations](experiment-brief.md).
- [NEAT-Python configuration](https://neat-python.readthedocs.io/en/latest/config_file.html).
  The installed 2.0.0 source is authoritative for this prototype; online docs may
  describe newer options.
- [ipywidgets button and callback API](https://ipywidgets.readthedocs.io/en/latest/reference/ipywidgets.html).
- [Torchvision ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Torchvision model and checkpoint API](https://docs.pytorch.org/vision/stable/models.html).
- [Earle et al. Picbreeder implementation](https://github.com/smearle/picbreeder-vlm),
  a reference for the local interaction loop and output-to-pixel mapping.
