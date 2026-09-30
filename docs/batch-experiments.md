# Batch experiments, metrics and viewer

This work has three phases:

1. **Implemented:** run one strategy configuration across independent seeds and
   save per-generation diagnostics, per-run summaries and batch statistics.
2. **Implemented:** apply the same reporting standard to human notebook
   sessions, distinguishing chronological exploration from final ancestry.
3. **Implemented:** a static HTML viewer with linked charts, generation
   navigation, full candidate grids and final-image galleries.

There are no human ratings or overall interestingness scores. These measurements
describe image changes, selection behaviour and computational cost; they do not
establish creativity, naturalness or UFR.

## Run a batch

```sh
uv run python experiments/run_selection.py novelty --runs 10 --seed 7 --steps 100 --output runs/novelty-batch
uv run --extra imagenet python experiments/run_selection.py offspring-value-imagenet --runs 10 --seed 7 --steps 100 --device mps --output runs/offspring-imagenet-batch
```

All strategy options are the same as for a standalone run. Put the strategy name
before the options. Omitting `--runs` keeps the standalone directory layout;
`--runs 1` explicitly creates a one-run batch. Both now save metrics.

Runs execute sequentially with identical settings except seeds. Run index `i`
uses `seed + i`. By default its selection seed is derived from that run's breeding
seed using the existing stable derivation. An explicit `--selection-seed K`
instead uses `K + i`. Every run constructs a new strategy, observer, predictor
and evaluator; state and training do not carry between runs. Downloaded model
weights may use the same cache. No extra offspring or model evaluations are
performed to produce the reports.

```python
from automated_picbreeder.experiment import ExperimentSettings
from automated_picbreeder.experiment_batch import run_batch
from automated_picbreeder.selection_strategies import NoveltySelectionStrategy

batch = run_batch(
    strategy_factory=NoveltySelectionStrategy,
    output_dir="runs/novelty-batch-python",
    runs=10,
    settings=ExperimentSettings(seed=7, steps=100),
)
```

The factory must return a fresh instance with the same `describe()` configuration
each time. Use the same seed schedule, image size, mutation settings and decision
budget when comparing strategies. Equal seeds do not preserve candidate sets
after choices diverge.

Existing output directories are rejected. Each run's status is saved before and
after execution. Ordinary exceptions are recorded and later seeds continue; the
CLI exits with code 1 if any run failed. Ctrl-C records interruption and stops the
batch, leaving later seeds pending. Saved partial sessions are retained, with
partial metrics where extraction succeeds. These are audit artifacts, not
resumable evolution sessions.

## Artifacts

```text
batch.json                         configuration, seed schedule, status, links
aggregate.json                     equal-weight statistics across completed runs
runs.csv                           one row per requested run, including failures
index.html                         offline batch viewer
run-0000-seed7/
    session.json                   original genomes, ancestry, decisions, scores
    images/                        every generated image, including rejects
    grids/                         each generation's full candidate grid
    selected.png                   final selected image
    checkpoints/                   existing audit snapshots
    source/                        source and dependency snapshot
    performance.json               stage timings for completed decisions
    metrics.json                   canonical per-generation and per-run diagnostics
    metrics.csv                    scalar metrics with image and class references
    index.html                     offline run inspector (also saved for human sessions)
```

`batch.json` contains each run's final selected-image path. `metrics.json` contains
the selected image and nine ordered candidates for every generation, with paths
relative to the run directory. This is sufficient for the later galleries without
copying images or ranking different generations by incomparable internal scores.
A failed run's image link, if present, is its last completed selection and remains
labelled failed in the manifest.

Metrics and batch artifacts have `schema_version: 1`. Session format remains 2.
Generation indices and candidate positions are **zero based**, matching the
saved grid filenames and strategy records. The trajectory axis is an explicit
selection event: human reselections get their own rows and may share a grid.
`display_index` identifies the grid visit and `round` records the notebook round.
JSON `null` means unavailable; CSV
uses an empty cell. Zero is an actual measured value. The original `session.json`
remains the source of full classifier vectors and model provenance.

Metrics can be rebuilt from a saved human or automated session without model inference:

```python
from automated_picbreeder.experiment_metrics import save_run_metrics

save_run_metrics("runs/novelty-batch/run-0000-seed7")
# For a partial failed run, preserve the known batch status explicitly:
# save_run_metrics(path, status="failed")
```

Extraction reads saved RGB PNGs and decisions. It does not alter the original
session or use any selection RNG. If timing data is absent it stays unavailable.
Human sessions use the same exporter and diagnostics. Their additional history
and optional post-run measurements are described below.

## Open the interactive viewer

New automated runs, batches and notebook saves automatically write `index.html`.
Open it directly in a browser. It needs no running server, external JavaScript,
internet connection or model inference. A batch's viewer is updated as runs
finish; reload the file to see the latest snapshot.

Build or refresh a viewer for existing results:

```sh
# A run, a batch, or all experiments beneath a collection directory.
uv run python experiments/view_results.py runs/novelty-batch
uv run python experiments/view_results.py runs/

# Combine chosen automated and human sessions in a single report.
uv run python experiments/view_results.py runs/novelty-batch runs/cppn-SAVED-TIMESTAMP --output runs/comparison.html
```

With no `--output`, the file is `index.html` inside the first source directory.
Discovery recognizes batch boundaries so batch runs are not listed twice.
Explicit generation rebuilds diagnostics in memory from the saved session without
rewriting the original session, metrics or batch aggregates. Failed, pending,
interrupted and unreadable entries remain visible with their status or error.
The automatic save path reuses the metrics it just produced.

The collection overview shows every run's final saved image, a strategy filter,
a run ledger and batch trajectory plots. Batch plots offer a scalar metric
selector, median and interquartile band, and optional individual trajectories.
Hover a trajectory to identify its seed, or click it to open that run at the
corresponding selection. Completed runs have equal weight; missing values and
failed runs are excluded, with contributing counts shown. A mixed collection is
not silently pooled into one aggregate: each original batch retains its group.

The run inspector provides:

- A selection slider, previous/next buttons and left/right arrow navigation.
- The selected image and its complete nine-candidate grid. Click any image for
  a larger preview and its saved measurements.
- Linked strategy-specific plots. Clicking a plot changes the active selection
  and moves the shared marker on every chart. Unavailable values remain gaps.
- An additional-metric selector and an exact scalar-value table for every
  recorded metric, plus full run summaries, settings and provenance.
- ImageNet class timelines when available; post-run classification remains
  explicitly separate from selection-time classification.
- **Selected timeline**, **All display visits**, **Final ancestry**, and
  **All generated images** galleries. The latter includes every rejected genome
  and abandoned branch; larger galleries are paginated.

Manual sessions use exactly the same inspector. Back/reset markers, notebook
rounds and display indices remain visible. The saved final image is shown
separately when it differs from the last chronological choice. A session with
no selection still exposes its displayed grids and generated images.
If metric extraction fails, the inspector shows the error and still exposes
the saved images and source-file links when the session record can be read.

The HTML embeds report data and its CSS/JavaScript, but references existing PNGs
through relative paths. Share the report **with its referenced run folders**
(for example, zip the batch directory). A standalone HTML file without those
folders retains charts but cannot display the images. These are static snapshots,
not a live dashboard; regenerate after changing or adding results.

## Human notebook sessions

In notebook 01, **Save session** (or `playground.save()`) saves `index.html`, `metrics.json`,
`metrics.csv`, `performance.json` and a selected contact sheet for every selection
event, along with the original complete session. `playground.last_saved_dir`
points to the latest export. No classifier is loaded and no additional mutations
or selection-RNG draws occur. Restart an existing notebook kernel after updating
the imported implementation.

There are three distinct records, present in both human and automated reports:

- `generations`: chronological explicit selection events, including revisions
  made on the same grid. It preserves every click; no discarded branch is removed.
- `display_history`: every reset, evolution and Back visit, including grids on
  which no selection was made. Each visit has its ordered candidates, selected
  state, round, originating mutation settings and associated selection indices.
- `final_ancestry`: actual genome parent links from the final selected genome to
  its root, listed root first. Repeated selection of one genome is represented by
  its `selection_generations`, not duplicated ancestry nodes.

`final_selected_id` and `final_image` describe the session's **current** selection.
After Back this may differ from the last chronological selection; after a reset
with no new choice it is null. `on_final_ancestry` marks which selected genomes
belong to that ancestry; it does not imply that every occurrence was on the final
chronological branch. A saved human snapshot is labelled complete as an export,
not as completion of a predetermined experimental budget.

Candidate presentations count nine per **display visit**, not nine per click.
Repeated selection on one grid adds a decision but no presentations. Back and
reset add display visits; generated genomes count only newly created genomes.
Mutation without another selection click leaves the retained parent selected,
but does not invent a human decision. `selection_revisions`, `backtracks`,
`resets` (excluding the initial roots) and `displayed_grids` make these differences
explicit. Use presentation/genome budgets, as well as settings, when comparing
human exploration with automated trajectories. A simple straight human run with
one choice per grid has the same counters and shared measurements as automation.

`preceding_actions` marks branch/reset transitions before each selection. Shared
pixel-change measurements still compare chronological selections, so changes
caused by Back or reset must not be interpreted as mutation effects.
`parent_retained` means choosing the actual retained parent of that grid; it is
null on every root grid, including revisits and resets. Child metrics also apply
only to actual offspring grids. Each row records the mutation settings that
created the grid, rather than treating the notebook's current slider values as
fixed settings for the whole session.

Human `performance.json` records elapsed interaction time and event timestamps
since playground construction. It includes rendering, deliberation and idle time;
it excludes the current export. It appears as `interaction_seconds` and
`interaction_elapsed_seconds`. Automated runtime and stage timings remain null
for human sessions. Missing historical timing data is not reconstructed.

## Optional shared post-run ImageNet evaluation

Use the same explicit frozen evaluator on human **and** automated saved sessions:

```python
from automated_picbreeder.imagenet import ImageNetEvaluator
from automated_picbreeder.posthoc_evaluation import evaluate_session_imagenet
from automated_picbreeder.experiment_metrics import save_run_metrics

evaluator = ImageNetEvaluator(model_name="resnet18", weights="IMAGENET1K_V1", device="cpu")
evaluate_session_imagenet(saved_directory, evaluator=evaluator)
metrics = save_run_metrics(saved_directory)
```

This optional operation needs the ImageNet extra and may download weights. It
classifies each saved genome once, including rejected images and discarded
branches, and reuses that result for repeated presentations of that genome.
Distinct genomes with identical images are still evaluated separately.
`posthoc_imagenet.json` preserves the full measurements, positional class names,
evaluator provenance, evaluated-image count, batch size, elapsed time and a hash
of the source session. Extraction rejects results from a different session
snapshot. The original session's choices and selection-time evaluations remain
unchanged.

The shared `posthoc_imagenet_*` metrics describe selected maximum confidence,
grid mean/range of per-candidate maxima, and selected top-class transitions.
Candidates have `posthoc_top_class` and `posthoc_max_class_confidence`; generation
rows have `posthoc_selected_class`. Costs are separate as
`posthoc_classifier_images` and `posthoc_evaluation_seconds`. Confidence is not
silently assigned as a human selection score. There are no human observer or
offspring forecasts to reconstruct.

For comparisons, fix checkpoint, preprocessing, device and batching. Report this
analysis cost separately from selection cost. Rebuilding metrics later reuses
the saved measurements without inference. This per-session operation does not
rewrite an existing batch's manifest or aggregates; rebuilding its viewer will
use current post-run results without changing those source files. New reports can be
passed to `summarize_batch` to summarize a matched set of human or automated runs.

## Shared measurements

All pixel MSEs use full-resolution RGB values divided by 255, averaged over pixels
and channels. There is no resizing, crop or per-image normalization.

| Metric | Definition |
| --- | --- |
| `pixel_mse_previous` | Selected image versus the previous selected image |
| `pixel_mse_nearest_earlier` | Minimum MSE to all earlier selected images, including the immediate predecessor |
| `parent_retained` | Whether the selected genome is the grid's retained parent (null on root grids) |
| `image_unchanged` | Whether the selected image's pixels equal the previous selection |
| `unchanged_image_streak` | Number of consecutive identical selected images, including the current image; starts at 1 |
| `children_identical_to_parent_fraction` | Fraction of eight children exactly matching the retained parent's pixels |
| `distinct_child_images` | Number of distinct pixel arrays among the eight children; the retained parent is excluded |
| `display_novelty_selected` | Selected image's pixel MSE to the immediately previous displayed grid's mean, derived for every strategy |

Transition metrics are unavailable for the first selection. Child/parent metrics
are unavailable on root grids. `display_novelty` also has grid mean/min/max fields
and candidate values; it is unavailable only on the first display, and uses the
previous display visit (including Back/reset/unselected grids) for human sessions.
Reselecting on one grid keeps its reference fixed. On normal automated runs it
matches recorded strategy `pixel_novelty` only for `previous-grid-mean` selection.
With `previous-parent`, strategy novelty uses the preceding selected image while
`display_novelty` continues using the previous displayed grid mean.
Means of binary flags are frequencies over available decisions. The maximum
`unchanged_image_streak` is the longest unchanged streak. Distinct genome IDs do
not imply distinct images. Historical nearest-image MSE has quadratic cost in
the decision count; comparisons use bounded chunks and deduplicate exactly equal
earlier images without changing the minimum.

Run summaries also record decisions, candidate presentations, generated genomes,
classifier image evaluations, runtime and seconds per decision. Classifier counts
are separate from observer masked-input counts and training examples. All nine
images contribute equally to grid means/ranges, including duplicates and the
retained parent. Random selection has no classifier scores or evaluations.

## Strategy diagnostics

### Novelty

`novelty_selected`, `novelty_grid_mean`, `novelty_grid_min` and
`novelty_grid_max` describe recorded raw distances from the **configured novelty
reference**: previous displayed grid mean (default) or previous selected parent.
`novelty_parent_margin` compares the chosen candidate with
the retained parent under that same reference. These differ from historical
nearest-image MSE. All are unavailable on the first grid.

### ImageNet

`imagenet_selected_confidence` is the chosen image's maximum class probability.
`imagenet_grid_mean_confidence` averages each of the nine candidates' maximum
probabilities; corresponding minimum/maximum fields describe their range.
It is not the mean over all 1,000 classes, which would be constant.

Each candidate has `top_class: {index, name}` and `max_class_confidence`.
The selected class also appears on the generation row. Class identity uses the
column index because names may repeat. `imagenet_top_class_changed` compares
successive selected class indices, with the first comparison unavailable.

Pure ImageNet also records `exploratory_choice` and cumulative
`exploratory_choice_frequency`. These describe the actual random-choice
operation, even when it picks the greedy winner. A novelty strategy's random
first choice is not labelled epsilon exploration.

### Combined current-image values

Each chosen image has `value_novelty_contribution`, `value_quality_contribution`
and `value_current`, with `value_quality_weight` recording the effective weight:

```text
novelty contribution = (1 - effective weight) * novelty rank
quality contribution = effective weight * confidence or comprehension rank
current value = novelty contribution + quality contribution
```

The same components are stored for every candidate. Initial neutral ranks remain
numeric because they really enter the decision, while unavailable raw measurements
remain null. Observer warm-up uses its recorded effective weight, not the eventual
configured weight. `choice_differs_from_novelty` and `choice_differs_from_quality`
compare with each component's first maximum on that grid, only when available.
Ranks describe relative choice pressure; they are not absolute progress measures.

### Masked-image predictability

`prediction_mse_selected` and its grid mean/min/max use errors recorded **before
the current grid's observer training**. The first grid is unavailable.
`comprehension_warmup_complete` marks the configured boundary. Observer replay
size, optimizer updates, sampled training examples and training loss are separate
diagnostics. Training loss is not substituted for pre-update prediction error.
Error can decline through improved prediction or selection of easier images.

### Offspring prediction

`offspring_forecast_selected` and its grid statistics describe pre-outcome
forecasts. `value_offspring_contribution` is `gamma * forecast` only when forecasts
actually influence selection, otherwise zero. Candidate contributions sum to the
recorded decision score. Eligibility, completed-target count, predictor warm-up,
forecast use and whether it changed the current-value choice are explicit fields.

`offspring_target`, signed/absolute/squared error and baseline squared errors are
assigned to the **parent's selection generation**. `offspring_outcome` records the
arrival generation, original forecast and eight actual child measurements under
the frozen selection-time references. New runs record `offspring_aggregation`: `max`
(default) or `mean`; `offspring_target` follows that choice. Existing saved mean
targets retain their original meaning. Max predicts the expected best value in an
eight-child brood; it is not the mean-child diagnostic. Repeated children count
individually, and the retained parent is excluded. The frozen reference is the
active selection-time image (grid mean or previous parent), not the next grid's
reference. The first/ineligible parents and the final unobserved parent have null targets.

Cumulative MAE/RMSE and running-mean/current-value baseline RMSE use only observed
eligible targets. These are retrospective curves indexed by forecast generation;
an outcome is available only when the following generation arrives. Final run
summaries split observed targets into all, forecast-used and forecast-not-used
groups. The latter includes warm-up and gamma-zero controls. No outcome or error
is invented for rejected parents.

## Timing and aggregation

`performance.json` records breeding, rendering, selection and saving times per
decision. Selection includes all strategy work: inference, training and audit
metadata. Saving includes decision bookkeeping, session writes and contact sheets.
Automated run elapsed time covers the evolution loop; it excludes strategy construction and
metrics extraction. Batch entries separately record strategy setup and full run
wall time, including metric extraction. Batch elapsed time includes reporting
between runs. Stage timings use host wall clocks, not accelerator profiling.

Observer and offspring strategy timing fields provide finer breakdowns where
already measured. They are nested inside selection time; **do not add them to it**.
For frozen ImageNet, selection time measures the classifier plus selection and
metadata overhead; it is not labelled pure inference time.

Each per-run scalar has count, mean, median, minimum, maximum, quartiles and sample
standard deviation over available generations, plus the final generation's value
(null if unavailable there). Time/work counters additionally have totals. These
within-run statistics are descriptive, not independent statistical samples.

Batch aggregation gives **each completed run equal weight**, including for aligned
generation trajectories. It reports count, mean, median, range, quartiles and
sample standard deviation. A single observation has null standard deviation.
Missing values are excluded with counts retained; failures and partial runs remain
in the manifest/CSV but are excluded from completed-run statistics. Do not treat
generations as independent replicates. No confidence intervals or significance
claims are produced.
