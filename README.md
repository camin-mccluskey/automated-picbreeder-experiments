# Automated Picbreeder

A small research project for learning how CPPNs generate images, how NEAT-Python
mutates them, and how different selection strategies shape their evolution.
Human and automated selection share the same nine-image breeding loop.

Automated runs support uniform random selection, pixel novelty, novelty plus
online predictability, greedy ImageNet selection and
ImageNet selection with occasional random exploration. The notebooks support
classifier inspection, comparison of selection strategies and interventions on
saved networks. Representation assessment and a frozen scientific protocol remain
unfinished; attractive images or high classifier scores are not evidence of UFR.

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
checkpoints or a finished human-choice data collection protocol. Sessions remain
in memory until saved; save before restarting the kernel.

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
non-classifier experiments. Include `--extra imagenet` in `uv run` commands that
need it, so uv retains these optional dependencies in the environment.

The notebook renders nine CPPNs, shows the classifier's top labels, inspects its
actual input crop, and demonstrates a simple contrast evaluator. It then compares
all three selection modes on the same grid and runs a short breeding experiment.
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

The first adapter uses **ResNet-18 / IMAGENET1K_V1**, with all parameters frozen,
evaluation mode and gradient-free inference. It returns all 1,000 softmax class
scores, not just a top label. The first load downloads ~45 MB of official weights;
later loads use the cache. Package versions are locked in `uv.lock`; result
metadata records the concrete weight enum, checkpoint SHA-256 and preprocessing.
`DEFAULT` weight aliases are rejected to avoid silently changing checkpoints.
CPU is the default; `device="mps"` or `device="cuda"` explicitly opts into an
available accelerator. Changing model, device, library version or batching can
change numerical results; fix these before main runs.

CPPNs render as RGB from HSB outputs. The classifier adapter also accepts
grayscale arrays and replicates them across RGB channels. The checkpoint's official transform
resizes the shorter side to 256, centre-crops to 224, rescales pixels and applies
ImageNet channel normalization. **This crops the image edges**; the notebook shows
what is retained. No per-image min/max normalization is applied. Inputs must be
uint8 H×W or H×W×3 arrays; float ranges and other channel layouts are rejected.

ImageNet is the training dataset/category set; ResNet-18 is the classifier. Its
scores are not validated naturalness, novelty or interestingness measures, and
abstract CPPNs are outside its ordinary natural-image setting. This is an
exploratory choice, not an exact Innovation Engine replication.

## Automated selection experiments

From the repository root:

```sh
# Uniform random selection; no Torch or classifier required.
uv run python experiments/run_selection.py --selection-strategy random --steps 100 --seed 7

# Pixel distance from the previous generation's mean; no Torch required.
uv run python experiments/run_selection.py --selection-strategy novelty --steps 100 --seed 7

# Greedy selection by maximum ImageNet class probability.
uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet --steps 100 --seed 7

# ImageNet selection with a 10% probability of a random choice each decision.
uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet --epsilon 0.1 --steps 100 --seed 7

# Add an online masked-image observer (random or pretrained backbone). Experiment balances novelty with predictability.
uv run --extra imagenet python experiments/run_selection.py --selection-strategy novelty-predictability --observer-initialization random --steps 3 --seed 7
uv run --extra imagenet python experiments/run_selection.py --selection-strategy novelty-predictability --observer-initialization imagenet --steps 3 --seed 7


uv run python experiments/run_selection.py --help
```

The runner starts with nine random CPPNs, asks the selection strategy to choose
one, retains it unchanged in position 1, and generates eight independently mutated
children. It repeats this process without resets or backtracking. Human selection
uses the same `BreedingSession`, rendering and mutation code.

| Selection strategy | Behaviour |
| --- | --- |
| `RandomSelectionStrategy()` | Uniform choice from all nine candidates, including the retained parent; no inference or scores |
| `NoveltySelectionStrategy()` | Uniform first choice, then greatest mean squared pixel distance from the previous grid's mean image |
| `NoveltyPredictabilitySelectionStrategy(observer_initialization="random")` | Combine novelty rank with pre-update masked-pixel accuracy rank; train ResNet18 online after each choice |
| `NoveltyPredictabilitySelectionStrategy(observer_initialization="imagenet")` | Same architecture, head and training; start the backbone from pretrained ImageNet weights |
| `ImageNetSelectionStrategy()` | Score each candidate by its highest probability across all 1,000 classes and choose the first maximum |
| `ImageNetSelectionStrategy(epsilon=0.1)` | With probability 0.1 choose uniformly; otherwise use the same greedy ImageNet rule |

Exact greedy ties favour the first candidate, retaining the parent after
initialization. The winning class can change between decisions. With fixed,
deterministic measurements, greedy selection cannot decrease the maximum class
score; exploratory choices can. A random choice may still pick the greedy winner,
and the record retains which operation was used.

Epsilon must be in [0, 1]. `epsilon=0` is greedy; `epsilon=1` always chooses
randomly while still recording ImageNet measurements. ImageNet selection evaluates
all nine candidates on every turn, including exploratory turns. Pure random
selection performs no classifier evaluation. Neither path filters duplicate images.

Pixel novelty uses full-resolution pixels in [0, 1], averaging all nine previous
images equally, including duplicates. Raw distances are recorded as `pixel_novelty`;
preference scores are ascending percentile ranks with averaged ties. First-grid
distances are zero, scores are 0.5, and `reference_available` is false. Later ties
choose the first candidate. A repeated image can still score highly if it differs
from the previous mean. This is local visual change, not novelty across all history
or a validated interestingness measure.

Create a **fresh `NoveltySelectionStrategy()` for each run**: every `choose` call
advances its reference, including notebook previews. Import it from
`automated_picbreeder.selection_strategies` and pass it to the same runner shown
below. [Notebook 05](notebooks/05_novelty_predictability.ipynb) shows synthetic
examples, reference means, a short CPPN run and matched observer predictions.

### Novelty plus predictability

`NoveltyPredictabilitySelectionStrategy` retains the same pixel novelty reference,
and also learns to reconstruct hidden 8x8 tiles of 32x32 images. It combines
novelty and comprehension percentile ranks with `comprehension_weight=0.5` by
default. Comprehension is `1 - masked_mse`, computed before training on the current
grid. Both initializations train the entire ResNet18 backbone and an identically
seeded new reconstruction head, with a zero-initialized fourth input channel for
the visibility mask. Scoring uses evaluation mode so BatchNorm stays fixed.

After each choice, all new distinct displayed images enter replay, including
rejected candidates. Twenty Adam updates (batch size 16, learning rate 0.001) sample
from the full replay dataset. The first choice is uniform, with unavailable
comprehension; a fresh instance is required for each run. The per-run selection RNG
supplies recorded model/update seeds without changing the breeding RNG. A retained
image may already have been trained on; repeat flags separate it from fresh images.

Python constructor options are `comprehension_weight`, `observer_initialization`,
`training_steps`, `batch_size` (at least 2 for BatchNorm), `learning_rate`, and
`cache_dir`, and `device` (`cpu` or `mps`). The CLI exposes `--comprehension-weight`, `--observer-initialization`,
`--training-steps`, `--observer-batch-size`, and `--learning-rate` only for this
strategy. `--device mps` uses the Apple GPU; CPU remains the default.
Initialization and training-example sampling stay on CPU for both devices.
An unavailable requested device fails explicitly, without silently falling back.
Novelty alone still needs no Torch. Pretrained observer weights reuse `.cache/imagenet`
in CLI/notebook runs; the scratch observer downloads nothing.

Per-decision evaluation metadata records both raw measurements, their ranks,
pre-update errors, observer model hashes, seeds, replay sizes, sampled-example
hashes, loss traces and scoring/training times. These records support replay from
the beginning in the same environment, not checkpoint resumption. Model hashes
include BatchNorm buffers; optimizer state is recreated by replaying updates.
Short fresh replays matched exactly on CPU and MPS in the tested environment;
this does not promise equality across devices, library versions or longer runs.
The `evaluated_images` counter counts candidate rows, not the sixteen masked
inputs per image or training examples. Availability of the strategies is not
evidence that they produce more interesting images.

### Run the Experiment 1 comparison

From the repository root, install the observer dependencies and start all three
conditions with matched seeds and budgets:

```sh
uv sync --locked --extra imagenet
uv run --extra imagenet python experiments/compare_selection.py --device mps --output runs/my-experiment1
```

Defaults are **9 runs**: novelty alone, novelty plus the randomly initialized
observer, and novelty plus the pretrained observer, each with seeds 7, 8 and 9.
Every run uses 100 decisions, 96×96 rendering, mutation strength 0.2, topology
changes enabled, and its own fresh strategy and derived selection RNG. Both
observers use weight 0.5, 20 Adam updates of batch size 16 per decision, and learning
rate 0.001. No random-selection baseline or offspring prediction is included.
The runs execute sequentially. The command above uses Apple Silicon MPS; omit
`--device mps` to use CPU. On this machine, a 20-update observer benchmark took
about 0.4 seconds on MPS versus 5–6 seconds on a busy CPU. Whole runs also render
and save images, so this is not the expected speedup for the complete experiment.

For a quick workflow check with a deliberately smaller budget:

```sh
uv run --extra imagenet python experiments/compare_selection.py --device mps --seeds 7 --steps 3 --size 32 --training-steps 2 --observer-batch-size 4 --output runs/my-experiment1-smoke
```

Use a **new output directory every time**, or omit `--output` for a timestamped
directory. The command does not resume interrupted runs. Each child directory
contains the normal `session.json`, all candidate images, grids and checkpoints.
At the comparison root:

- `comparison.json` records the configuration and status of every scheduled run.
- `REPORT.md` and `final-images.png` show every final selection, with failures or
  incomplete runs explicitly identified.
- `report.json` records per-run counts, repeat rates, fresh-image errors, common
  pixel-diversity measurements and computational costs. “Fresh” means absent from
  earlier grids; duplicate presentations within the current grid each count.
- `trajectories.csv` contains per-decision measurements for further inspection.

Failures do not erase completed runs or silently remove scheduled conditions.
Rebuild a report from existing artifacts without running evolution:

```sh
uv run python experiments/compare_selection.py --report-only runs/my-experiment1
uv run --extra imagenet jupyter lab notebooks/06_selection_comparison.ipynb
```

In notebook 06, set `comparison_dir` to your output directory. It reads reports by
default; `RUN_COMPARISON` and `RUN_DIAGNOSTICS` are explicit opt-ins for new work.
The diagnostics train on one synthetic image set and evaluate a separate set,
and can feed the same saved grids to all strategies to compare choices under
identical exposure. Independent breeding runs see different images after their
choices diverge; their prediction errors alone do not isolate pretraining effects.

For one condition rather than the full batch, use `experiments/run_selection.py`
with `--selection-strategy novelty` or `novelty-predictability` as shown above.
Experiment 2 adds online offspring-value prediction, described below.

The initial nine-run MPS pilot is complete. See [WRAP_UP.md](WRAP_UP.md) for the
settings, verification and findings, and the local
[full report](runs/experiment1-pilot-mps/REPORT.md) for every final image. The combined
strategies reduced pixel diversity in all three seeds; both converged near black
for seed 9. These outcomes are included rather than treated as failed runs or omitted.

### Experiment 2: predict offspring value online

`OffspringValueSelectionStrategy` learns which selected parents produce valuable
children. A separate, initially random scalar predictor takes a parent image and
its scoring context and forecasts the mean value of its next eight children.
When those children arrive, the strategy records the original forecast's error
and trains on that transition. All completed transitions remain in replay,
including repeated parents. Rejected parents have no offspring labels.

The target combines novelty and comprehension using the **parent-selection-time**
observer, previous-grid mean image and original nine measurement references.
Those are frozen until the actual children arrive. The retained parent is excluded;
duplicate children each count. This avoids reranking siblings against themselves.
The live comprehension observer continues its normal training on all distinct
shown images, including rejected candidates.

Run a short inspection with Apple GPU access:

```sh
uv run --extra imagenet python experiments/run_selection.py \
  --selection-strategy offspring-value --gamma 1 --device mps \
  --steps 14 --seed 7 --size 96 --output runs/my-offspring-inspection
```

Use `--gamma 0` and a different output directory for the current-image control.
Both configurations train and log the predictor; gamma zero ignores it when
choosing and reproduces Experiment 1's scratch-observer trajectory. Use `--device
cpu` without Apple GPU access. The observer is scratch-initialized in this batch;
`--observer-initialization imagenet` is deliberately rejected for this strategy.

Defaults are ten valid targets before forecasts affect selection, gamma 1,
ten predictor Adam updates per target, batch size 16 and learning rate 0.001.
The choice score is `current_value + gamma * predicted_offspring_value`; forecasts
are not reranked. Configure the new model with `--warmup-targets`,
`--predictor-training-steps`, `--predictor-batch-size` and
`--predictor-learning-rate`. Existing `--training-steps`, `--observer-batch-size`
and `--learning-rate` configure the separate comprehension observer.

The first transition has no informed value function; the final selected parent
has no observed children. There are `max(S-2, 0)` training targets in `S` decisions.
Fourteen decisions give twelve targets and three decisions with the forecast term
enabled. No additional offspring are generated. Use a fresh strategy instance
and chronological nine-image grids; backtracking and arbitrary grid playback are
unsupported. Saved chronological replay is tested on CPU and MPS within the same
environment; cross-device equality and checkpoint resumption are not promised.

```sh
uv run --extra imagenet jupyter lab notebooks/07_offspring_value.ipynb
```

Notebook 07 first inspects the target with three six-decision runs, then executes
two fourteen-decision online runs, verifies ancestry and exact fresh replay, and
shows all forecasts, outcomes and both final images. It uses MPS by default and
fresh timestamped `runs/offspring-target-inspection-*` directories. Reports include
`REPORT.md` for target inspection and `ONLINE_REPORT.md` for prediction inspection,
with machine-readable JSON and the normal session artifacts. Errors use the
original forecast, not a prediction recomputed after observing the target.
The two comparison forecasts use the past target mean and current parent value.

Only selected parents reveal outcomes, so these errors cannot establish that
rejected alternatives were ranked correctly. Improved training loss alone is not
evidence of useful selection. The comparison is Phase 3 of
[PLAN_EXPERIMENT_2.md](PLAN_EXPERIMENT_2.md); progress and verification are recorded
in [TODO_EXPERIMENT_2.md](TODO_EXPERIMENT_2.md).

#### Run the Experiment 2 comparison

```sh
uv run --extra imagenet python experiments/compare_offspring_value.py \
  --device mps --output runs/my-experiment2
```

This runs both gamma configurations for seeds 7, 8 and 9, with 100 decisions,
size 96, mutation strength 0.2 and topology changes enabled. Observer and predictor
budgets match the defaults above. It creates six fresh runs, each with an independent observer and predictor;
existing output directories are rejected. Each completed run has 900 candidate
presentations, 801 genomes and 98 observed offspring targets. It may take several
minutes; interruptions and failures remain visible in the manifest and report.

```sh
uv run python experiments/compare_offspring_value.py --report-only runs/my-experiment2
```

Reporting requires no Torch or additional evolution. `REPORT.md` includes all six
final images and complete selected-image trajectories, prediction MAE/MSE during
and after the ten-target warm-up, both simple baselines, ten-outcome error windows,
repeat rates, pixel diversity, target saturation and computational costs.
`report.json`, `forecasts.csv`, `prediction-windows.csv` and `trajectories.csv`
contain the underlying measurements; each run preserves all nine-image grids,
ancestry, source, checkpoints and rejected candidates.

Notebook 07's final section reads an existing comparison via `comparison_dir`.
It does not launch the larger batch. The earlier notebook sections still run the
short target/prediction inspections when executed.

The first Phase 2 inspection is complete: both fourteen-decision MPS runs
replayed exactly and took approximately 18 seconds each. Forecast MSE was 0.0271,
versus 0.0162 for the past-target-mean baseline. Gamma 1 was enabled for three
decisions and changed none of them in this seed-7 inspection. This establishes
working online feedback and reproducibility, not useful prediction yet. See the
[inspection report](runs/offspring-target-inspection-20260928T222120-932319Z/ONLINE_REPORT.md)
for every forecast and both final images.

#### Experiment 2 pilot findings

The six-run MPS pilot is complete: 600 decisions, 4,806 generated genomes and
588 observed offspring targets. All 219 tests pass. All three gamma-zero controls
exactly reproduce Experiment 1's 100 choices, observer hashes and 801 PNGs per run.
The six runs took approximately 15.7 minutes in total.

| Seed | Forecast-driven changes / 89 enabled decisions | Later predictor MSE | Later running-mean MSE | Pixel diversity: control → forecast-driven |
| --- | --- | --- | --- | --- |
| 7 | 19 / 89 | 0.02064 | 0.01724 | 0.09213 → 0.40952 |
| 8 | 11 / 89 | 0.01630 | 0.01268 | 0.22096 → 0.30149 |
| 9 | 22 / 89 | 0.01789 | 0.01391 | 0.00040 → 0.24891 |

“Later” means the 88 observed forecasts made after ten targets were available;
the final enabled decision has no observed brood. Forecast MSE fell from the first
20 to the last 20 outcomes in five of six runs, but that includes changing target
difficulty. Among forecast-driven runs, seed 7 beat the running mean in its final
20 outcomes, seed 8 became worse, and seed 9 approximately tied it. Across the
whole post-warm-up period, none of those three runs beat the running mean.

The forecast term changed 52 choices, of which 43 resolved current-value ties.
Pixel diversity increased in all three seeds, but that does not establish better
forecasting or greater interestingness. Seed 9 avoided the control's near-black
endpoint but produced a visibly noisy pattern. All six endpoints and full
trajectories are retained; no unattractive outcome was dropped.

[Full comparison report](runs/experiment2-pilot-mps/REPORT.md) ·
[Forecasts and learning windows](runs/experiment2-pilot-mps/prediction-windows.csv) ·
[Artifact and control-parity audit](runs/experiment2-pilot-mps/artifact-audit.json) ·
[Additional pilot measurements](runs/experiment2-pilot-mps/pilot-findings.json).
The notebook's comparison-reading section was executed and saved as
[comparison-inspection.ipynb](runs/experiment2-pilot-mps/comparison-inspection.ipynb).

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

### Settings, reproducibility and budgets

CLI and Python defaults match: 100 decisions, seed 7, image size 96, mutation
strength 0.2, topology changes enabled and a checkpoint every 10 decisions.
`--selection-strategy` is required. Other settings are `--size`,
`--mutation-strength`, `--no-topology`, `--checkpoint-every`, `--selection-seed`
and `--output`. `--epsilon` applies only to ImageNet selection. `--device` applies
to ImageNet and novelty-predictability selection; both default to CPU.

Breeding and selection use separate random generators. The selection seed is
stably derived from the breeding seed by default, can be overridden with
`--selection-seed` or `ExperimentSettings(selection_seed=...)`, and is saved in
`metadata.settings.selection_seed`. Each run creates a fresh selection RNG, so
reusing a selection strategy does not continue a previous run's random sequence.
It does retain any strategy-owned history; use fresh stateful strategies per run.
Reproduction requires the same strategy, settings and environment, subject to
classifier numerical determinism. Restart notebook kernels after changing imported
modules; rerunning imports alone does not refresh existing objects.

`--steps` includes the initial choice. For S decisions:

| Quantity | Count |
| --- | --- |
| Candidate presentations | 9 × S |
| Unique generated genomes | 9 + 8 × (S − 1) |
| ImageNet image evaluations | 9 × S, including repeated parents |
| Novelty image evaluations | 9 × S recorded rows, including the initial grid's unavailable-reference placeholders; no model inference |
| Pure random image evaluations | 0 |

A 100-decision run therefore presents 900 candidates and creates 801 genomes.
Match decision budgets, seeds, rendering and mutation settings when comparing
selection strategies; record evaluation cost separately. Matching seeds does not
keep candidate sets identical after choices diverge.

### Saved sessions

Default outputs go into new directories named
`runs/<strategy>-selection-<timestamp>-seed<seed>/`. Existing directories
are never overwritten. Both human and automated runs use `SessionWriter` and
version-2 `session.json` records:

- `session.json`: configuration, rendering, versions, all genomes and ancestry,
  image paths/hashes, ordered displays, events, selection, metadata and summary.
- `images/<8-digit-genome-id>.png`: every generated candidate, including rejected
  alternatives and discarded branches.
- `cppn.cfg`, `source/`: configuration and a hashed implementation/lockfile snapshot.
- `selected.png`: a convenience copy of the current selection.
- Automated runs also have `grids/` and `checkpoints/`. Grids distinguish selection
  mode and preference scores from raw measurements. Checkpoints are audit snapshots,
  not resumable sessions; there is no resume command.

Every selection event records its `position`, `genome`, ordered `displayed` IDs,
`evaluation` and `decision`:

| Choice | `evaluation` | `decision` |
| --- | --- | --- |
| Human | `null` | `null` |
| Pure random | `null` | `mode: "random"`, `scores: null` |
| Pixel novelty | Raw distances, reference generation (zero-based; initially null), ordered pixel hashes and previously-seen flags | First `"random"`, then `"greedy"`; percentile-rank scores |
| Novelty plus predictability | Raw novelty/comprehension, pre-update errors, ranks and observer learning records | First `"random"`, then `"greedy"`; weighted rank scores |
| ImageNet | Full measurement values, names and provenance | `mode: "greedy"` or `"random"`, one maximum-class score per candidate |

Measurements and scores belong to decisions, so repeated parent evaluations are
recorded separately. Measurement rows follow `displayed`; column positions follow
`names`. `metadata.selection_strategy` describes the selection strategy. Human
exports have null `metadata.settings` and `summary`; automated summaries include
candidate presentations, evaluated images, unique candidates, selected score
(null when unscored) and selection mode.

Snapshots are written before calling the selection strategy and after recording
its decision. A strategy failure leaves the current grid and earlier decisions
available, with a null summary. Image paths in sessions and checkpoints are
relative to the run directory.

Existing version-2 HSB sessions remain readable by the interpretation notebooks,
including older records with `metadata.selector` and no `decision` field. They
are not migrated or rewritten. Version-1 exports and older rendering conventions
are not supported by that reader. The previous experiment API and command have
been replaced; use the API and CLI above for new runs.

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
[reference provenance](src/automated_picbreeder/reference_data/skull/README.md).
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

## Experimental sequence

| Step | Build and learn | What to establish before proceeding |
| --- | --- | --- |
| 1. CPPN playground (current) | Render, select, mutate, inspect genomes and sweep a weight | Understand coordinate inputs, activations, weights and topology |
| 2. Automated search (current) | Random, greedy ImageNet and epsilon-greedy ImageNet selection in the shared nine-image loop | Check shared breeding behavior, full choice-set records, costs and selection trajectories |
| 3. Pilot assessment | Graphs, activation maps, selective attribute interventions and tests of shared computation | Define measurable attributes, preservation criteria, thresholds and held-out intervention settings |
| 4. Frozen experiment | Independent seeds and fixed decision budgets, with evaluation costs, checkpoints and network-sampling rules recorded | Measure both selective semantic control and shared computation; include duplicates, failures and uncertainty |
| 5. Human-choice follow-up | Same CPPN implementation and matched candidate sets; record position, alternatives, ancestry, branching and resets | Specify a classifier choice rule; predictive agreement does not establish the same mechanism |

The primary hypothesis concerns whether the complete automated system can produce
UFR evidence. Class confidence is not a naturalness measure. Changes in the selected
image's top label are classifier-label transitions. The prototype's weight sweep
is exploratory and is not a validated UFR metric or standard DCI.

## Files and checks

- `notebooks/01_cppn_selection.ipynb`: guided, runnable first experiment.
- `notebooks/02_image_evaluation.ipynb`: classifier inspection, selection-strategy comparison and a short run.
- `notebooks/05_novelty_predictability.ipynb`: pixel novelty and pre-update masked prediction inspection for both observer initializations.
- `notebooks/06_selection_comparison.ipynb`: all-condition reports, trajectories, held-out-image diagnostics and common-grid inspection.
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
- `experiments/run_selection.py`: thin CLI for the same Python selection strategies.
- `experiments/compare_selection.py`: run the three Experiment 1 conditions across a seed set, or rebuild their report.

```sh
uv run pytest
uv run --extra imagenet pytest  # also run offline classifier adapter checks
```

Classifier unit tests use controlled logits and the real preprocessing transform;
they never download weights. To check the downloaded model itself, run notebook 02.

## References

- [Experiment brief](docs/experiment-brief.md), supplied by the project owner.
- [NEAT-Python configuration](https://neat-python.readthedocs.io/en/latest/config_file.html).
  The installed 2.0.0 source is authoritative for this prototype; online docs may
  describe newer options.
- [ipywidgets button and callback API](https://ipywidgets.readthedocs.io/en/latest/reference/ipywidgets.html).
- [Torchvision ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Torchvision model and checkpoint API](https://docs.pytorch.org/vision/stable/models.html).
- [Earle et al. Picbreeder implementation](https://github.com/smearle/picbreeder-vlm),
  a reference for the local interaction loop and output-to-pixel mapping.
