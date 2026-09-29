# Automated Picbreeder

A small research project for learning how CPPNs generate images, how NEAT-Python
mutates them, and how different selection strategies shape their evolution.
Human and automated selection share the same nine-image breeding loop.

Automated runs support uniform random selection, pixel novelty, novelty plus
ImageNet confidence, novelty plus online predictability, predicted offspring value
under either value function, greedy ImageNet selection and
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
abstract CPPNs are outside its ordinary natural-image setting. This is an
exploratory choice, not an exact Innovation Engine replication.

## Automated selection experiments

`experiments/run_selection.py` is the single CLI for all automated selection runs.
Run each strategy/seed configuration with a new output directory. From the repository root:

```sh
# Uniform random selection; no Torch or classifier required.
uv run python experiments/run_selection.py random --steps 100 --seed 7

# Pixel distance from the previous generation's mean; no Torch required.
uv run python experiments/run_selection.py novelty --steps 100 --seed 7

# Balance pixel novelty with frozen ImageNet classification confidence.
uv run --extra imagenet python experiments/run_selection.py novelty-imagenet --comprehension-weight 0.5 --steps 100 --seed 7

# Greedy selection by maximum ImageNet class probability.
uv run --extra imagenet python experiments/run_selection.py imagenet --steps 100 --seed 7

# ImageNet selection with a 10% probability of a random choice each decision.
uv run --extra imagenet python experiments/run_selection.py imagenet --epsilon 0.1 --steps 100 --seed 7

# Balance novelty with predictability from an online masked-image observer.
uv run --extra imagenet python experiments/run_selection.py novelty-predictability --observer-initialization random --steps 3 --seed 7
uv run --extra imagenet python experiments/run_selection.py novelty-predictability --observer-initialization imagenet --steps 3 --seed 7

# Add online prediction of the selected parent's offspring value.
uv run --extra imagenet python experiments/run_selection.py offspring-value --gamma 1 --steps 14 --seed 7

# Predict offspring value using novelty plus frozen ImageNet confidence.
uv run --extra imagenet python experiments/run_selection.py offspring-value-imagenet --gamma 1 --steps 14 --seed 7

uv run python experiments/run_selection.py --help
uv run python experiments/run_selection.py offspring-value-imagenet --help
```

The runner starts with nine random CPPNs, asks the selection strategy to choose
one, retains it unchanged in position 1, and generates eight independently mutated
children. It repeats this process without resets or backtracking. Human selection
uses the same `BreedingSession`, rendering and mutation code.

| Selection strategy | Behaviour |
| --- | --- |
| `RandomSelectionStrategy()` | Uniform choice from all nine candidates, including the retained parent; no inference or scores |
| `NoveltySelectionStrategy()` | Uniform first choice, then greatest mean squared pixel distance from the previous grid's mean image |
| `NoveltyImageNetSelectionStrategy()` | Combine novelty rank and frozen maximum ImageNet class-confidence rank; confidence available immediately |
| `NoveltyPredictabilitySelectionStrategy(observer_initialization="random")` | Combine novelty rank with pre-update masked-pixel accuracy rank; train ResNet18 online after each choice |
| `NoveltyPredictabilitySelectionStrategy(observer_initialization="imagenet")` | Same architecture, head and training; start the backbone from pretrained ImageNet weights |
| `OffspringValueSelectionStrategy()` | Add predicted mean offspring value to current-image value after warm-up; train on actual selected-parent transitions |
| `OffspringValueImageNetSelectionStrategy()` | Same offspring prediction and selection, using novelty plus frozen ImageNet confidence as the value function |
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

### Novelty plus ImageNet confidence

`NoveltyImageNetSelectionStrategy(comprehension_weight=0.5, evaluator=None)` uses
the same pixel novelty reference and the same frozen classifier as ImageNet
selection. It scores each candidate as `(1-weight)*novelty_rank + weight*confidence_rank`,
where confidence is its maximum class probability. Both ranks use average ties
within the current grid. Ranking discards the magnitude of differences: a small
confidence gap can affect selection as much as a large one.

The first grid has neutral novelty ranks and available classifier confidence.
Positive weights therefore choose by confidence immediately; weight zero makes
the first choice uniform. Later choices take the first maximum combined score.
Weight zero matches pure novelty choices; weight one matches greedy ImageNet
choices. There is no training, replay dataset or comprehension warm-up.

Pass a configured `ImageNetEvaluator` to reuse a classifier or change its model.
The CLI accepts `--comprehension-weight`, `--imagenet-model`, `--imagenet-weights`,
`--imagenet-batch-size`, `--device` and `--cache-dir`. Observer, predictor, warm-up
and epsilon options do not apply. Create a fresh strategy for each run to reset
its novelty reference; the frozen evaluator can be shared.

Each decision records `pixel_novelty` and `imagenet_confidence`, both ranks, and
the full classifier values, class names and provenance in
`evaluation.metadata.classifier_evaluation`. All nine candidates are classified
on every decision, even at weight zero: `9*S` classifier evaluations for `S`
decisions. Novelty uses full-resolution pixels while the classifier uses its
checkpoint's preprocessing, including the default centre crop. Classification
confidence replaces patch predictability as the proposed comprehension proxy;
it is not a validated measure of comprehension or recognisability.

### Novelty plus predictability

`NoveltyPredictabilitySelectionStrategy` retains the same pixel novelty reference,
and also learns to reconstruct hidden 8x8 tiles of 32x32 images. It combines
novelty and comprehension percentile ranks with `comprehension_weight=0.5` by
default. Comprehension is `1 - masked_mse`, computed before training on the current
grid. `comprehension_warmup_steps=10` sets the effective comprehension weight to
zero for the first ten decisions: the first choice is random, then selection uses
novelty alone. Comprehension is still measured after the first grid and the observer
trains after every decision. The configured weight first applies at decision 11.
Set the warm-up to 0 to use comprehension as soon as its reference is available.
Both initializations train the entire ResNet18 backbone and an identically
seeded new reconstruction head, with a zero-initialized fourth input channel for
the visibility mask. Scoring uses evaluation mode so BatchNorm stays fixed.

After each choice, all new distinct displayed images enter replay, including
rejected candidates. Twenty Adam updates (batch size 16, learning rate 0.001) sample
from the full replay dataset. The first choice is uniform, with unavailable
comprehension; a fresh instance is required for each run. The per-run selection RNG
supplies recorded model/update seeds without changing the breeding RNG. A retained
image may already have been trained on; repeat flags separate it from fresh images.

Python constructor options are `comprehension_weight`, `comprehension_warmup_steps` (integer >= 0), `observer_initialization`,
`training_steps`, `batch_size` (at least 2 for BatchNorm), `learning_rate`,
`cache_dir`, and `device` (`cpu` or `mps`). The CLI exposes `--comprehension-weight`,
`--comprehension-warmup-steps`, `--observer-initialization`, `--training-steps`, `--observer-batch-size`,
`--learning-rate`, `--cache-dir` and `--device` for novelty-predictability or
offspring-value selection. `--device mps` uses the Apple GPU; CPU remains the default.
Initialization and training-example sampling stay on CPU for both devices.
An unavailable requested device fails explicitly, without silently falling back.
Novelty alone still needs no Torch. Pretrained observer weights default to `.cache/imagenet`
in CLI/notebook runs; the scratch observer downloads nothing.

Per-decision evaluation metadata records comprehension warm-up completion,
the effective comprehension weight, both raw measurements, their ranks,
pre-update errors, observer model hashes, seeds, replay sizes, sampled-example
hashes, loss traces and scoring/training times. These records support replay from
the beginning in the same environment, not checkpoint resumption. Model hashes
include BatchNorm buffers; optimizer state is recreated by replaying updates.
Short fresh replays matched exactly on CPU and MPS in the tested environment;
this does not promise equality across devices, library versions or longer runs.
The `evaluated_images` counter counts candidate rows, not the sixteen masked
inputs per image or training examples. Availability of the strategies is not
evidence that they produce more interesting images.

### Predict offspring value online

`OffspringValueSelectionStrategy` learns which selected parents produce valuable
children. A separate, initially random scalar predictor takes a parent image and
its scoring context and forecasts the mean value of its next eight children.
When those children arrive, the strategy records the original forecast's error
and trains on that transition, provided the parent was selected after comprehension
warm-up. Earlier parents never supply training targets, even if their children arrive
after warm-up ends. All completed eligible transitions remain in replay,
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
  offspring-value --gamma 1 --device mps \
  --steps 23 --seed 7 --size 96 --output runs/my-offspring-inspection
```

Use `--gamma 0` and a different output directory for the current-image control.
Both configurations train and log the predictor; gamma zero ignores it when
choosing and reproduces novelty-predictability's scratch-observer trajectory
with matching settings. Use `--device
cpu` without Apple GPU access. The observer is scratch-initialized in this strategy;
`--observer-initialization imagenet` is deliberately rejected for this strategy.

The warm-ups run sequentially. Defaults are ten comprehension warm-up decisions,
then ten eligible targets before forecasts affect selection, gamma 1,
ten predictor Adam updates per target, batch size 16 and learning rate 0.001.
The choice score is `current_value + gamma * predicted_offspring_value`; forecasts
are not reranked. Configure the new model with `--warmup-targets`,
`--predictor-training-steps`, `--predictor-batch-size` and
`--predictor-learning-rate`. Existing `--training-steps`, `--observer-batch-size`
and `--learning-rate` configure the separate comprehension observer.

The first grid has no informed value function; the final selected parent
has no observed children. With comprehension warm-up `W`, there are
`max(S-max(W, 1)-1, 0)` training targets in `S` decisions. With both defaults at 10,
decision 11 selects the first eligible parent, decision 12 receives the first target,
and decision 21 receives the tenth target and can use forecasts immediately.
Twenty-three decisions give twelve targets and three decisions with the forecast term
enabled. Target eligibility and completion counts are saved per decision. No
additional offspring are generated. Use a fresh strategy instance
and chronological nine-image grids; backtracking and arbitrary grid playback are
unsupported. Saved chronological replay is tested on CPU and MPS within the same
environment; cross-device equality and checkpoint resumption are not promised.

```sh
uv run --extra imagenet jupyter lab notebooks/07_offspring_value.ipynb
```

Notebook 07 first inspects the target with three six-decision runs, then executes
two fourteen-decision online runs, verifies ancestry and exact fresh replay, and
shows all forecasts, outcomes and both final images. It uses MPS by default and
explicitly disables comprehension warm-up to keep these short inspections useful;
normal strategy and CLI defaults remain ten decisions. It uses
fresh timestamped `runs/offspring-target-inspection-*` directories. Reports include
`REPORT.md` for target inspection and `ONLINE_REPORT.md` for prediction inspection,
with machine-readable JSON and the normal session artifacts. Errors use the
original forecast, not a prediction recomputed after observing the target.
The two comparison forecasts use the past target mean and current parent value.

Only selected parents reveal outcomes, so these errors cannot establish that
rejected alternatives were ranked correctly. Improved training loss alone is not
evidence of useful selection.

### Predict offspring value with ImageNet confidence

`OffspringValueImageNetSelectionStrategy` uses the same delayed-target, predictor
training and selection machinery as `OffspringValueSelectionStrategy`. Its current
value comes from `NoveltyImageNetSelectionStrategy`: a weighted sum of pixel
novelty rank and maximum ImageNet class-confidence rank. After predictor warm-up,
selection uses `current_value + gamma * predicted_mean_offspring_value`, without
reranking forecasts. Defaults remain `comprehension_weight=0.5`, `gamma=1`, ten
completed targets, ten predictor updates per target, batch size 16 and learning
rate 0.001.

The ImageNet classifier stays frozen. Only the offspring predictor trains; there
is no patch observer or comprehension warm-up. The predictor receives the same
parent RGB and previous-grid mean images, plus 21 context scalars: raw novelty,
confidence, current value, and the two sorted nine-candidate measurement vectors.
The patch version retains its 23 inputs, including two observer training-history
statistics. Predictor seeds are derived from the initial supplied selection RNG
state without consuming selection draws.

Each selected parent's target keeps its selection-time previous-grid mean and
nine novelty/confidence rank references. On the next grid, the eight actual
children are scored against those fixed references and averaged, excluding the
retained parent and counting duplicate children individually. The current grid's
classifier output supplies their confidence values, so no extra inference or
offspring generation is needed: all nine candidates are classified exactly once
per decision, for `9*S` classifier evaluations. Every eligible transition remains
in predictor replay, including repeated parents.

The first choice follows novelty-imagenet immediately, but that first parent's
transition is ineligible because no previous-grid novelty reference exists.
The final parent has no observed children. There are `max(S-2, 0)` completed targets:
decision 2 selects the first eligible parent, decision 3 receives the first target,
and decision 12 receives the tenth target and can first use forecasts with default
settings. `--gamma 0` still trains and logs, while exactly preserving matching
novelty-imagenet choices and selection RNG use.

```sh
uv run --extra imagenet python experiments/run_selection.py \
  offspring-value-imagenet --comprehension-weight 0.5 \
  --gamma 1 --warmup-targets 10 --steps 14 --seed 7 --device cpu
```

The CLI accepts the ImageNet model/checkpoint/batch/cache options and all offspring
predictor options. `--device` applies to both models and supports CPU or MPS;
the predictor does not support CUDA. Patch-observer options, including
`--comprehension-warmup-steps`, and epsilon are rejected. Python callers can pass
an `evaluator=ImageNetEvaluator(...)`; `device` on the strategy configures the
predictor, while the evaluator owns its classifier device.

Records preserve full classifier outputs and provenance, current values, original
forecasts, fixed target references, realised child values, forecast errors,
predictor seeds, hashes and training records. Use a fresh strategy for each run
and chronological nine-image grids with the retained parent first. As with the
patch version, saved genome links establish ancestry; matching pixels alone
cannot. Forecast errors describe selected parents, not rejected alternatives.

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
The strategy subcommand is required and comes before all options. Other settings
are `--size`, `--mutation-strength`, `--no-topology`, `--checkpoint-every`, `--selection-seed`
and `--output`. `--epsilon` applies only to ImageNet selection. `--device` applies
to all strategies using a model; all default to CPU.

All strategy constructor settings and ImageNet evaluator settings are exposed:

| Applies to | CLI options (defaults) |
| --- | --- |
| All runs | `--steps 100`, `--seed 7`, `--selection-seed` (derived), `--size 96`, `--mutation-strength 0.2`, `--no-topology` (off), `--checkpoint-every 10`, `--output` (new timestamped directory) |
| All model-based strategies | `--device cpu`, `--cache-dir` (repository `.cache/imagenet`) |
| ImageNet only | `--epsilon 0` |
| ImageNet, novelty-imagenet and offspring-value-imagenet | `--imagenet-model resnet18`, `--imagenet-weights IMAGENET1K_V1`, `--imagenet-batch-size 16` |
| All combined novelty strategies | `--comprehension-weight 0.5` |
| Novelty-predictability and offspring-value | `--comprehension-warmup-steps 10`, `--observer-initialization random`, `--training-steps 20`, `--observer-batch-size 16`, `--learning-rate 0.001` |
| Offspring-value and offspring-value-imagenet | `--gamma 1`, `--warmup-targets 10`, `--predictor-training-steps 10`, `--predictor-batch-size 16`, `--predictor-learning-rate 0.001` |

Top-level `--help` lists strategies; `STRATEGY --help` shows only that strategy's
options, grouped by purpose. Random and novelty have only run settings. Options
for an unrelated strategy and abbreviated option names are rejected. The former
`--selection-strategy NAME` syntax has been replaced by the `NAME` subcommand.
Observer initialization can be `random` or `imagenet` for
novelty-predictability; offspring-value requires `random`. Observers and predictors
support CPU/MPS; the frozen ImageNet evaluator also supports CUDA. ImageNet model
and weight options configure only the frozen classifier; observers use ResNet18.
Choose an explicit checkpoint version supported by the selected ImageNet-1K model;
`DEFAULT` is rejected. `--cache-dir` controls downloaded weight storage.

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
| Offspring value | Current-image measurements, frozen child targets, original forecasts, delayed errors and predictor training records | Current-image value plus the weighted forecast after warm-up |
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

- [Experiment brief](docs/experiment-brief.md), supplied by the project owner.
- [NEAT-Python configuration](https://neat-python.readthedocs.io/en/latest/config_file.html).
  The installed 2.0.0 source is authoritative for this prototype; online docs may
  describe newer options.
- [ipywidgets button and callback API](https://ipywidgets.readthedocs.io/en/latest/reference/ipywidgets.html).
- [Torchvision ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Torchvision model and checkpoint API](https://docs.pytorch.org/vision/stable/models.html).
- [Earle et al. Picbreeder implementation](https://github.com/smearle/picbreeder-vlm),
  a reference for the local interaction loop and output-to-pixel mapping.
