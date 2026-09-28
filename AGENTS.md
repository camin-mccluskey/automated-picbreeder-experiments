# Automated Picbreeder: agent guide

## Purpose and current scope

This is a small research and learning project for evolving compositional pattern-producing networks (CPPNs). The user experiments in stages, inspecting human-selected and automatically selected images before extending the system.

The longer-term question is whether automatically evolved CPPNs develop unified factored representations (UFR): meaningful properties that can be varied independently, and repeated structure that depends on shared computation. The working breeding tools are implemented; the quantitative representation assessment and frozen experimental protocol are not yet complete. High classifier confidence is not evidence of naturalness, creativity or UFR.

Read [docs/experiment-brief.md](docs/experiment-brief.md) for the scientific objective and [README.md](README.md) for usage and implementation details. The current experiment is the single-parent loop below. The earlier MAP-Elites proposal is historical context, not the implemented method.

## Current experiment

- Begin with nine random CPPNs. Select one parent, retain it unchanged in the first position, and generate eight independently mutated offspring. Repeat.
- Human selection happens in the notebook. Automated runs accept one `SelectionStrategy`: random, pixel novelty, novelty-predictability, offspring-value or ImageNet.
- Novelty selects uniformly on the first grid, then maximizes full-resolution pixel MSE from the previous displayed grid's mean image. All previous candidates contribute equally, including duplicates. It records raw distance and percentile-rank scores, requires no Torch, and retains state: construct a fresh instance for each run/inspection.
- `NoveltyPredictabilitySelectionStrategy` combines novelty and masked-pixel accuracy ranks. Random/pretrained ResNet18 observers share architecture and head initialization, train online on all distinct displayed images after the choice, and run on CPU by default or MPS explicitly. Initialization and sampling remain on CPU. First-grid comprehension is unavailable. The existing RNG supplies model/update seeds; evaluation metadata carries replay/training records and model hashes. These are not resumable model checkpoints.
- `experiments/compare_selection.py` runs novelty, predictability-random and predictability-imagenet through the existing runner, with fresh strategies and seeds 7/8/9 by default. Its manifest includes all scheduled runs and failures; reports use final endpoints and exclude unavailable/repeated prediction errors from fresh-image MSE. `--report-only` rebuilds reports without training. Larger runs may take tens of minutes; outputs must be new directories.
- Random selection is uniform over all nine candidates, including the parent, and performs no inference. ImageNet selection scores each candidate by its maximum class probability and chooses the first maximum, except that with probability epsilon it chooses uniformly instead. Classes may change; greedy ties retain the parent after initialization. Epsilon is in [0, 1], with 0 greedy and 1 always random but still evaluated.
- Both paths use the same breeding and rendering code. When comparing selection strategies, match seeds, image sizes, mutation settings and decision budgets; report classifier cost separately. Human backtracking, resets or parameter adjustments introduce additional differences. Matching seeds does not keep choice sets identical after selections diverge.
- Use NEAT-Python's genomes and mutation operators, not its full population/speciation/crossover algorithm. There is currently no crossover or MAP-Elites archive.
- With `S` decisions the runner presents `9*S` candidates and generates `9 + 8*(S-1)` genomes. ImageNet selection evaluates all nine each turn, including exploratory turns and retained parents (`9*S` evaluations). Novelty also records `9*S` measurement rows (the first nine are unavailable-reference placeholders), with no model inference. Pure random selection performs zero image evaluations.
- The frozen classifier is Torchvision ResNet18 with explicit `IMAGENET1K_V1` weights, CPU by default. Its softmax scores are relative across 1,000 classes. Official preprocessing includes a centre crop, so the classifier sees a cropped version of the displayed image.

Experiment 2 adds `OffspringValueSelectionStrategy`, using the same complete
strategy interface. Its scalar predictor starts from scratch, trains on completed
selected-parent transitions, and predicts mean child value from parent pixels and
scoring context. The separate comprehension observer also starts from scratch.
`FrozenOffspringValue` uses the selection-time observer, mean image and nine rank
references to score the eight actual children; exclude the retained parent and
count duplicate children individually. Ten valid targets precede forecast-driven
selection; gamma 0 trains/logs but exactly preserves Experiment 1 choices. Derive
predictor seeds without extra selection-RNG draws. Preserve every transition,
including repeated parents, and the original pre-outcome forecast and error.

Require nine RGB images in chronological order and the retained selected parent
at position 0. Pixel equality cannot prove ancestry; tests/notebook check saved
genome parent links. There are max(S-2,0) valid targets; the first transition is
ineligible and the final parent unobserved. Do not generate extra offspring or
invent rejected-parent labels. CPU/MPS replay is tested within one environment.
`experiments/compare_offspring_value.py` runs both gamma configurations over the
same seeds/budgets. `--report-only` rebuilds the all-outcome report without Torch
or training. Phase assignment uses completed target count at forecast time, even
for gamma zero; never use forecast-used status to put all control errors in warm-up.
The report recomputes errors from original forecasts and verifies delayed records,
past-only baselines, counts and actual ancestry. Keep missing/failed runs visible.

The Experiment 2 six-run MPS pilot is complete (219 tests passed). All gamma-zero
runs exactly match Experiment 1; active runs changed 19/11/22 choices and increased
pixel diversity, but none beat the running-mean baseline over all post-warm-up
outcomes. Most choice changes resolved current-value ties. Preserve these limits
when discussing results. README.md and `runs/experiment2-pilot-mps/REPORT.md` hold
the findings; only the final user review remains in TODO_EXPERIMENT_2.md.

## Code structure

Paths below are relative to the repository root. Core modules live in `src/automated_picbreeder/`.

| Path | Responsibility |
| --- | --- |
| `cppn.py`, `cppn.cfg` | NEAT configuration, genome initialization/mutation, coordinate inputs, rendering and genome serialization |
| `breeding.py` | `BreedingSession`: shared candidates, selection, ancestry, mutation, backtracking and reset history |
| `notebook.py` | `CPPNPlayground`: ipywidgets interface around the shared session |
| `evaluation.py` | Generic `ImageEvaluator` protocol and image-by-measurement `Evaluation` |
| `imagenet.py` | Optional frozen Torchvision classifier adapter and preprocessing provenance |
| `image_predictability.py` | Optional CPU/MPS masked ResNet18 observer, initialization, scoring, training and inference-only snapshots |
| `offspring_prediction.py` | Optional CPU/MPS scalar predictor, full-resolution image/context inputs and transition updates |
| `offspring_value.py` | Frozen selection-time offspring value target and fixed-reference ranks |
| `offspring_comparison.py` | Experiment 2 matched batch, original-forecast errors over time, all endpoints and costs |
| `selection_comparison.py`, `selection_diagnostics.py` | Fixed-condition batch orchestration, complete outcome reports, held-out-image and common-grid checks |
| `selection_strategies.py` | `SelectionStrategy`, `SelectionDecision`, random, novelty, novelty-predictability, offspring-value and ImageNet selection |
| `experiment.py` | Automated loop, settings, independent selection RNG, counters and checkpoints |
| `experiment_reporting.py` | Contact sheets and progress, using recorded decisions and optional scores |
| `persistence.py` | `SessionWriter`: common assembly and saving of run data for both interfaces |
| `experiments/run_selection.py` | Thin selection-strategy CLI; experiment entry points belong in `experiments/` |
| `notebooks/01_cppn_selection.ipynb` | Human breeding, genome inspection and weight sweeps |
| `notebooks/02_image_evaluation.ipynb` | Classifier inspection, selection-strategy comparison and a short run |
| `notebooks/05_novelty_predictability.ipynb` | Pixel novelty distances/reference means and a short run |
| `notebooks/06_selection_comparison.ipynb` | Comparison reports, trajectories and optional diagnostics |
| `notebooks/07_offspring_value.ipynb` | Target inspection, short online prediction runs, ancestry and exact replay checks |
| `tests/` | Rendering, mutation, UI callbacks, evaluator contracts, exports and human/automated parity |

## Representation and colour

Inputs are `(x, y, radius)`; outputs are mutable hue, saturation and brightness (HSB/HSV), not RGB directly. The only supported mapping is `[CPPNRendering] output_mapping = picbreeder_hsb_v1` in `cppn.cfg`:

```text
H = (h + 1) % 1
S = clip(s, 0, 1)
B = clip(abs(b), 0, 1)
RGB = colorsys.hsv_to_rgb(H, S, B)
byte = int(255 * clip(channel, 0, 1) + 0.5)
```

This matches Picbreeder-VLM's output-to-pixel mapping. Our activation-function definitions and reproduction rules differ from that repository. Output nodes are not forced to sigmoid. Do not silently change the mapping or add legacy grayscale/sigmoid-HSB compatibility; the user explicitly prefers one simple current implementation. `CPPNConfig.save()` preserves the rendering setting that plain NEAT configuration saving would omit.

## Shared interfaces and records

Keep candidate generation, selection strategies and persistence separate. The runner takes a single `selection_strategy` argument, with `choose(images, *, rng) -> SelectionDecision` and JSON-compatible `describe()`. The selection strategy owns any evaluation, scoring and final choice. Do not introduce separate scorer/selector configuration or classifier-specific breeding logic. Reuse `ImageNetEvaluator` as an internal helper or for standalone inspection. Evaluation rows follow candidate order; columns have stable positional identities because display names may repeat.

Both interfaces export version-2 `session.json` through `SessionWriter`, with all genomes, ancestry, events, configuration, rendering metadata and every generated PNG in `images/`. Include rejected alternatives and discarded branches. Each selection event has optional evaluation values/names/metadata and `decision` containing mode and optional preference scores. Human choices have null evaluation and decision. Pure random choices have null evaluation and scores, with mode `random`. ImageNet choices retain full measurements and scores, with mode `greedy` or `random`, even when a random draw happens to pick the greedy winner. Scores belong to decisions, since repeated evaluations of a parent may differ.

Store strategy configuration in `metadata.selection_strategy` and resolved seeds in `metadata.settings`. Existing version-2 HSB sessions with older metadata and no decision field remain readable by the interpretation tools; do not rewrite them. The old experiment API and command have been replaced without compatibility wrappers.

Automated runs additionally save grids and checkpoints. These are audit snapshots, not resumable evolution sessions. Use new output directories. `runs/`, `.cache/` and `.venv/` are local ignored artifacts; do not commit generated runs or model weights.

## Development and verification

Python 3.13+, managed with `uv`; dependencies are locked in `uv.lock`, with NEAT-Python pinned to 2.0.0. From the repository root:

```sh
uv sync --locked --extra imagenet
uv run jupyter lab notebooks/01_cppn_selection.ipynb
uv run python experiments/run_selection.py --selection-strategy random --steps 3 --seed 7
uv run --extra imagenet python experiments/run_selection.py --selection-strategy imagenet --epsilon 0.1 --steps 3 --seed 7
uv run --extra imagenet pytest
```

The ImageNet extra is optional for human, pure random and novelty selection. Include `--extra imagenet` in uv commands that need Torch. Classifier unit tests use controlled models and do not download weights; a real classifier smoke run may download the checkpoint into `.cache/imagenet`. CLI help lists all options; epsilon is only valid for ImageNet selection; device applies to ImageNet, novelty-predictability and offspring-value. Predictor options apply only to offspring-value; it uses a scratch observer. Observer devices are cpu/mps; comparison batches accept --device mps on Apple Silicon.

For changes affecting breeding, rendering or saving, check deterministic replay between human and automated paths, unchanged parents, output mapping, complete records and saved-image reproduction. Preserve the user's notebook seeds and exploratory settings unless the requested change requires otherwise. Check actual notebook/CLI settings rather than assuming they match.

The same seed and action sequence reproduce candidates in a fixed environment. The runner creates a fresh selection RNG per run, separate from breeding, with a stably derived seed or explicit `selection_seed` override. Selection strategies must use that supplied RNG, not global or persistent RNG state. Extra selection draws must not change mutation draws. Back restores a grid but does not rewind the breeding RNG; reset draws new roots from the continuing stream. RNG isolation is designed for single-threaded use. Restart notebook kernels after changing imported modules; rerunning imports alone does not refresh existing playground instances.

## Working with the user

Prefer direct explanations, inspectable changes and small experiments. Prioritize accuracy over reassuring claims. Keep implementation simple; avoid speculative frameworks and compatibility layers. Explain scientific consequences when changing representation or mutation. Distinguish implemented behaviour, proposed experiments and empirical findings. Preserve useful rough edges and user experiments rather than tidying unrelated work.
