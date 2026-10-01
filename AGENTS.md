# Automated Picbreeder: agent guide

## Purpose and current scope

This is a small research and learning project for evolving compositional pattern-producing networks (CPPNs). The user experiments in stages, inspecting human-selected and automatically selected images before extending the system.

The longer-term question is whether automatically evolved CPPNs develop unified factored representations (UFR): meaningful properties that can be varied independently, and repeated structure that depends on shared computation. The working breeding tools are implemented; the quantitative representation assessment and frozen experimental protocol are not yet complete. High classifier confidence is not evidence of naturalness, creativity or UFR.

Read [docs/experiment-brief.md](docs/experiment-brief.md) for the scientific objective and [README.md](README.md) for usage and implementation details. The current experiment is the single-parent loop below.

## Current experiment

- Begin with nine random CPPNs. Select one parent, retain it unchanged in the first position, and generate eight independently mutated offspring. Repeat.
- Human selection happens in the notebook. Automated runs accept one `SelectionStrategy`: random, pixel novelty, novelty-imagenet, novelty-predictability, offspring-value, offspring-value-imagenet, ImageNet or VLM.
- `VLMSelectionStrategy` uses the optional `vlm` extra and official OpenRouter SDK. Send nine separate full-resolution PNGs labelled 0–8 in one stateless request; default prompt is exactly `choose the most interesting image to you`. Require an explicit model and a strict selected-index/reason JSON response. Preserve candidate order, null scores/evaluation, and the supplied RNG state. Read `OPENROUTER_API_KEY` from the ignored repository `.env` (environment takes precedence); keep `.env.example` blank and trackable. Save decision metadata, usage/cost/attempts, and terminal failure diagnostics without credentials. Remote choices are not deterministically replayable. See `docs/strategies/vlm.md`.
- Novelty selects uniformly on the first grid, then maximizes full-resolution pixel MSE from `novelty_reference`: `previous-grid-mean` (default) or `previous-parent` (the preceding choice's selected image). In previous-grid-mean mode, all previous candidates contribute equally, including duplicates. It records raw distance and percentile-rank scores, requires no Torch, and retains state: construct a fresh instance for each run/inspection.
- `NoveltyImageNetSelectionStrategy` combines the same novelty ranks with frozen maximum ImageNet class-confidence ranks, using `comprehension_weight=0.5` by default. The first grid has neutral novelty; positive weights choose by confidence immediately, while weight zero preserves novelty's random first choice. No training or warm-up. Weight one matches greedy ImageNet choices. Classify all nine images every decision, including at weight zero. Store two measurements plus both ranks and full classifier output/provenance in `evaluation.metadata.classifier_evaluation`. Create a fresh strategy per run.
- `NoveltyPredictabilitySelectionStrategy` combines novelty and masked-pixel accuracy ranks. Random/pretrained ResNet18 observers share architecture and head initialization, train online on all distinct displayed images after the choice, and run on CPU by default or MPS explicitly. Initialization and sampling remain on CPU. Observers use a fixed 96x96 input/output with sixteen 24x24 masks. Other rendered image sizes are resized to 96x96; there is no observer-resolution option. First-grid comprehension is unavailable. The existing RNG supplies model/update seeds; evaluation metadata carries replay/training records and model hashes. These are not resumable model checkpoints.
- Random selection is uniform over all nine candidates, including the parent, and performs no inference. ImageNet selection scores each candidate by its maximum class probability and chooses the first maximum, except that with probability epsilon it chooses uniformly instead. Classes may change; greedy ties retain the parent after initialization. Epsilon is in [0, 1], with 0 greedy and 1 always random but still evaluated.
- Both paths use the same breeding and rendering code. When comparing selection strategies, match seeds, image sizes, mutation settings and decision budgets; report classifier cost separately. Human backtracking, resets or parameter adjustments introduce additional differences. Matching seeds does not keep choice sets identical after selections diverge.
- Use NEAT-Python's genomes and mutation operators, not its full population/speciation/crossover algorithm. There is currently no crossover or MAP-Elites archive.
- With `S` decisions the runner presents `9*S` candidates and generates `9 + 8*(S-1)` genomes. ImageNet selection evaluates all nine each turn, including exploratory turns and retained parents (`9*S` evaluations). Novelty also records `9*S` measurement rows (the first nine are unavailable-reference placeholders), with no model inference. Pure random selection performs zero image evaluations.
- The frozen classifier defaults to Torchvision ResNet18 with explicit `IMAGENET1K_V1` weights on CPU. The CLI also accepts other ImageNet-1K models and explicit checkpoint versions. Its softmax scores are relative across 1,000 classes. The default checkpoint's preprocessing includes a centre crop, so the classifier sees a cropped version of the displayed image.

`OffspringValueSelectionStrategy` predicts offspring value, using the same complete
strategy interface. Its scalar predictor starts from scratch, trains on completed
selected-parent transitions, and predicts max child value by default (`offspring_aggregation="mean"` is optional) from parent pixels and
scoring context. The separate comprehension observer also starts from scratch.
`FrozenOffspringValue` uses the selection-time observer, active novelty reference image and nine rank
references to score the eight actual children; exclude the retained parent and
count duplicate children individually. Both observer strategies default to ten
comprehension warm-up decisions: first random, then novelty-only selection, with
observer training throughout. Set `comprehension_warmup_steps=0` to disable.
Only parents selected after comprehension warm-up (and with a reference available)
are eligible for offspring targets; exclude earlier parents even when their children
arrive after warm-up. Ten eligible targets then precede forecast-driven selection
by default (first forecast use at decision 21); gamma 0 trains/logs but exactly preserves matching
novelty-predictability scratch-observer choices. Derive
predictor seeds without extra selection-RNG draws. Preserve every eligible transition,
including repeated parents, and the original pre-outcome forecast and error.

Require nine RGB images in chronological order and the retained selected parent
at position 0. Pixel equality cannot prove ancestry; tests/notebook check saved
genome parent links. For comprehension warm-up W, there are
max(S-max(W,1)-1,0) eligible targets; the first grid and warm-up parents are
ineligible and the final parent unobserved. Do not generate extra offspring or
invent rejected-parent labels. CPU/MPS replay is tested within one environment.

`OffspringValueImageNetSelectionStrategy` shares the offspring loop, predictor and
fixed-reference target calculations, using novelty-imagenet current values.
Only its offspring predictor trains; no patch observer or comprehension warm-up.
Keep the selection-time active novelty reference image and nine novelty/confidence references
for scoring the eight actual children. Reuse confidence from the next grid's one
classifier pass; all nine images are classified once per decision, including at
gamma zero. Gamma zero must preserve novelty-imagenet choices and selection RNG
use. Derive predictor seeds from a hash of the initial supplied RNG state without
drawing from it. Predictor context has 21 scalars (the patch version retains its
23, including observer training statistics). First transition ineligible, final
parent unobserved: max(S-2,0) targets. Ten targets precede forecast-driven selection
by default, so decision 12 can first use forecasts. CPU/MPS predictor only.

All five novelty strategies accept `novelty_reference="previous-grid-mean"` or
`"previous-parent"`; store the actual final selected image for the next decision,
including forecast-driven choices. The first grid has no reference in either mode.
Both offspring strategies accept `offspring_aggregation="max"` (new-run default)
or `"mean"`. Apply that aggregation to eight actual child values under the frozen
selection-time references; do not label maxima as means. Max forecasts the expected
best value of an eight-child brood. Existing saved mean runs retain their meaning
and are not rewritten. Derived `display_novelty` always uses the previous displayed
grid mean; it differs from strategy `pixel_novelty` in previous-parent mode.

## Code structure

Paths below are relative to the repository root. Core modules live in `src/automated_picbreeder/`.

| Path | Responsibility |
| --- | --- |
| `cppn.py`, `cppn.cfg` | NEAT configuration, genome initialization/mutation, coordinate inputs, rendering and genome serialization |
| `breeding.py` | `BreedingSession`: shared candidates, selection, ancestry, mutation, backtracking and reset history |
| `notebook.py` | `CPPNPlayground`: ipywidgets interface around the shared session |
| `evaluation.py` | Generic `ImageEvaluator` protocol and image-by-measurement `Evaluation` |
| `imagenet.py` | Optional frozen Torchvision classifier adapter and preprocessing provenance |
| `vlm.py` | Optional OpenRouter SDK adapter, PNG requests, strict response validation, credentials and request audit |
| `image_predictability.py` | Optional CPU/MPS masked ResNet18 observer, initialization, scoring, training and inference-only snapshots |
| `offspring_prediction.py` | Optional CPU/MPS scalar predictor, full-resolution image/context inputs and transition updates |
| `offspring_value.py` | Shared fixed-reference targets for frozen patch observers or already measured ImageNet confidence |
| `selection_strategies.py` | `SelectionStrategy`, `SelectionDecision`, random, novelty, novelty-imagenet, novelty-predictability, offspring-value, offspring-value-imagenet, ImageNet and VLM selection |
| `experiment.py` | Automated loop, settings, independent selection RNG, counters and checkpoints |
| `experiment_batch.py` | Sequential independent seeded runs, failure/status records and batch aggregation |
| `experiment_metrics.py` | Saved-session diagnostics, image references, per-generation CSV/JSON and equal-run summaries |
| `session_history.py` | Chronological display visits/selection events and actual final genome ancestry |
| `posthoc_evaluation.py` | Optional frozen ImageNet evaluation of saved images, separate from selection records and costs |
| `experiment_viewer.py`, `viewer_cache.py`, `viewer/` | Offline HTML export, relative image links and interactive charts/galleries |
| `experiment_reporting.py` | Contact sheets and progress, using recorded decisions and optional scores |
| `persistence.py` | `SessionWriter`: common assembly and saving of run data for both interfaces |
| `selection_cli.py` | Shared CLI option groups, strategy subcommands and builders; add a configure function and register it in `COMMANDS` |
| `experiments/run_selection.py` | Thin selection-strategy CLI entry point; experiment entry points belong in `experiments/` |
| `experiments/view_results.py` | Build a viewer for saved runs, batches or collections, including human sessions |
| `notebooks/01_cppn_selection.ipynb` | Human breeding, genome inspection and weight sweeps |
| `notebooks/02_image_evaluation.ipynb` | Classifier inspection, selection-strategy comparison and a short run |
| `notebooks/05_novelty_predictability.ipynb` | Pixel novelty distances/reference means and a short run |
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

`--runs N` on the existing CLI runs a batch with seed `base_seed + index`; explicit
selection seeds increment too, otherwise they derive from each run seed. Construct
fresh strategies/models for each run. Single and batch runs save version-1 metrics
JSON/CSV and stage timings without changing selection or running extra inference.
Batch manifests retain failures, and aggregates include only completed runs with
equal run weight. Unavailable values are null, generation indices are zero based,
and offspring outcomes align to the parent selection generation. See
[docs/batch-experiments.md](docs/batch-experiments.md). Human notebook saves now
export the same metrics and selected grids without inference. Each explicit
selection click gets a trajectory row; `display_history` separately counts grid
visits so reselections do not inflate candidate presentations. Back/reset visits
and discarded branches remain counted. `final_ancestry` follows actual genome
parent links, and the final image is the session's current selection, which can
differ from the last click. Human interaction timing is separate from automated
runtime. Optional `evaluate_session_imagenet` writes post-run measurements and
costs separately; it never relabels human choices as classifier decisions.
Every metrics export also generates `index.html`; batches maintain an overview.
`experiments/view_results.py runs/` builds an offline collection viewer without
inference or source-record changes. Each page embeds JavaScript and data, images use
relative paths, and no server/CDN is required. Collection overviews link to
companion per-run HTML pages. Viewer diagnostics and image inventories are cached
in `.viewer-cache/` with source/image stat and metric-code fingerprints; original
records stay unchanged. `--refresh` forces extraction and `--jobs` bounds independent
run workers (up to four by default). Save paths pass existing reports/aggregates
to rendering. Keep missing values as gaps, report
failure status, and preserve chronological clicks, display visits and actual
ancestry as separate views. The viewer is read-only; it cannot influence breeding.

## Strategy documentation

Keep README.md short and link to one guide per strategy in `docs/strategies/`.
When changing a strategy, constructor, CLI option or default, update its guide and
the linked shared option tables (`docs/strategies/options.md`, `docs/run-options.md`)
in the same change. Document every applicable option, default, constraint, Python
mapping and scientific limitation. Terminal examples must run long enough for the
advertised mechanism to affect selection; label shorter runs as smoke tests or
explain explicit warm-up overrides. Check examples against the parser and keep
`tests/test_strategy_docs.py` passing. Preserve useful notebook, export and
interpretation instructions in `docs/usage.md`, and update the experiment brief
and metric definitions when semantics change. Document current behaviour and
scientific limitations; omit completed implementation plans and deferred features.

## Development and verification

Python 3.13+, managed with `uv`; dependencies are locked in `uv.lock`, with NEAT-Python pinned to 2.0.0. From the repository root:

```sh
uv sync --locked --extra imagenet
uv run jupyter lab notebooks/01_cppn_selection.ipynb
uv run python experiments/run_selection.py random --steps 3 --seed 7
uv run --extra imagenet python experiments/run_selection.py imagenet --epsilon 0.1 --steps 3 --seed 7
uv run --extra imagenet pytest
uv run --extra imagenet --extra vlm pytest
```

The ImageNet extra is optional for human, pure random and novelty selection. Include `--extra imagenet` in uv commands that need Torch. Classifier unit tests use controlled models and do not download weights; a real classifier smoke run may download the checkpoint into `.cache/imagenet`. Run all automated experiments through `experiments/run_selection.py`. Put the strategy subcommand before all options. Top-level help lists strategies; `STRATEGY --help` lists only applicable options; epsilon is only valid for ImageNet selection; device applies to strategies using a model. `--imagenet-model`, `--imagenet-weights` and `--imagenet-batch-size` configure the frozen evaluator for imagenet, novelty-imagenet and offspring-value-imagenet; `--cache-dir` sets model weight storage. `--novelty-reference` applies to all five novelty strategies; `--offspring-aggregation` applies only to the two offspring strategies. `--comprehension-weight` applies to all combined novelty strategies. Predictor options apply to offspring-value and offspring-value-imagenet. Patch observers use fixed 96x96 inputs with sixteen 24x24 masks, including notebooks 05 and 07. Observer training/warm-up options apply only to novelty-predictability and offspring-value; the latter requires a scratch observer. Observer/predictor devices are cpu/mps; use --device mps on Apple Silicon. For offspring-value-imagenet, CLI device configures both classifier and predictor; Python strategy device configures the predictor, with classifier device supplied through its evaluator.

For changes affecting breeding, rendering or saving, check deterministic replay between human and automated paths, unchanged parents, output mapping, complete records and saved-image reproduction. Preserve the user's notebook seeds and exploratory settings unless the requested change requires otherwise. Check actual notebook/CLI settings rather than assuming they match.

The same seed and action sequence reproduce candidates in a fixed environment. The runner creates a fresh selection RNG per run, separate from breeding, with a stably derived seed or explicit `selection_seed` override. Selection strategies must use that supplied RNG, not global or persistent RNG state. Extra selection draws must not change mutation draws. Back restores a grid but does not rewind the breeding RNG; reset draws new roots from the continuing stream. RNG isolation is designed for single-threaded use. Restart notebook kernels after changing imported modules; rerunning imports alone does not refresh existing playground instances.

## Working with the user

Prefer direct explanations, inspectable changes and small experiments. Prioritize accuracy over reassuring claims. Keep implementation simple; avoid speculative frameworks and compatibility layers. Explain scientific consequences when changing representation or mutation. Distinguish implemented behaviour, proposed experiments and empirical findings. Preserve useful rough edges and user experiments rather than tidying unrelated work.
