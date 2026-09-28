# Experiment 2: predicting offspring value

Status: implementation and all six Phase 3 pilot runs are complete. The online
predictor and strategy are verified on CPU/MPS; reporting and artifact audits pass.
Results are delivered for final user review. See README.md for findings and commands.
Progress is tracked in TODO_EXPERIMENT_2.md. Source: [THOUGHTS.md](THOUGHTS.md) and the planning conversation.
The completed Experiment 1 plan, checklist and results remain in PLAN.md, TODO.md
and WRAP_UP.md.

## Objective and scope

Test whether predicting the mean value of a parent's next eight offspring helps
selection beyond the current-image value from Experiment 1. Predict a scalar
directly. Do not predict offspring images or generate trial offspring to choose
their parent. Train only after the ordinary breeding loop produces actual offspring.

Confirmed during planning:

- Start the offspring-value predictor from scratch in every run. Defer historical
  pretraining and the earlier proposed frozen-pretrained-predictor comparison.
- Judge offspring against the original nine-image comparison at parent selection
  time. Freeze the observer, novelty reference and rank scale for that target.
- Preserve the existing nine-candidate loop and complete selection strategies.

Starting defaults adopted with plan approval:

- Predictor inputs are images and scoring context; defer CPPN structure/weights.
  Plan approval adopts this first test; identical images can still have different
  mutation behaviour.
- Use the randomly initialized Experiment 1 comprehension observer in both arms.
  Keep its architecture and online training unchanged. The earlier pilot provides
  no reason to double this batch with another observer-initialization comparison.
- Compare current-image selection with and without using the learned prediction.
  Numerical defaults below are pilot choices to inspect before scaling.

This tests expected immediate offspring value, not offspring diversity, long-term
evolvability, learning progress, human interestingness or CPPN representation quality.
VLM selection, goal switching and historical training remain outside this batch.

## What the predictor learns

Let G_t be the current nine-image grid. The existing strategy scores it using:

- M_t: the mean image of G_(t-1), used for pixel novelty;
- O_t: the comprehension observer before training on G_t;
- A_t and B_t: the nine raw novelty and comprehension measurements for G_t.

Keep all four fixed when judging this parent's offspring. In particular, their
novelty uses the same M_t that judged the parent, not the newly computed mean of
G_t. Their comprehension uses O_t, not the observer after its next training update.

```text
value_t(image) = (1 - w) * reference_rank(novelty(image, M_t), A_t)
                + w * reference_rank(comprehension(image, O_t), B_t)

target_t = mean(value_t(child) for the eight actual children of the selected parent)

selection_score_t(parent) = value_t(parent) + gamma * predicted_target_t(parent)
```

The retained parent is never part of target_t. Identical offspring still count as
separate mutation outcomes; do not deduplicate the eight children. The target does
not include the child's own predicted offspring value: there is no recursion or
bootstrapped multi-generation return in this experiment.

The target is an average over one realised brood, a noisy observation of the
expected mean the predictor is trying to learn. It can differ from the scores used
to select from G_(t+1), whose observer and novelty reference have since advanced.

### Fixed-reference ranks

Do not rerank children against their siblings, or insert them into the reference
set. Every child is compared separately against the same original nine values.
Use the following proposed extension of the current average-tie percentile ranks:

```text
L = number of reference values strictly below x
E = number of reference values equal to x
reference_rank(x, reference) = clip((L + (E - 1) / 2) / 8, 0, 1)
```

On the original nine values this exactly matches Experiment 1, including ties.
Values below or above all references score 0 or 1. With all references equal,
an equal value scores 0.5. Rank resolution and saturation remain limitations:
inspect target variance and how often scores hit the bounds.

Why this matters: nine within-grid percentile ranks always average 0.5, including
with ties. The combined scores therefore also sum to 4.5. Excluding the retained
parent gives `(4.5 - parent_score) / 8`, which mostly measures the parent's relative
position rather than the quality of a brood. Fixed-reference scoring avoids this.

## Predictor and online learning

Keep the offspring predictor separate from the comprehension observer. Predictor
updates must not change the observer that defines the target.

Proposed small randomly initialized model:

- Input: candidate RGB image and M_t concatenated into six channels, in [0, 1],
  at the rendered resolution. Do not add a second 32x32 downsampling bottleneck.
- Three padded 3x3 convolutions, stride 2, with 16, 32 and 64 output channels and
  ReLU activations; global average pooling.
- Concatenate 23 context scalars: raw candidate novelty and comprehension, its
  current value, the sorted nine values in each rank reference, and log1p of the
  observer update count and number of distinct replay images before this decision.
- Linear layers from 87 to 64 to 1, ReLU between them, sigmoid output in [0, 1].
  No BatchNorm, dropout, pretrained weights or shared observer parameters.

This context only summarizes the observer's knowledge; it does not encode its
entire state. Likewise, images do not identify the underlying CPPN. Prediction
error therefore mixes model limitations, missing inputs and mutation randomness.
Hold mutation strength/topology fixed in the first pilot; do not claim transfer
to other mutation settings without conditioning and testing for that change.

Each completed valid transition contributes one example: the selected parent's
selection-time inputs and its realised mean offspring value. Store every transition,
including repeated parent images; their contexts and mutation outcomes can differ.
Retain the original inputs and labels. Do not recompute old labels with a newer
observer or quietly convert previous parents into timeless image-value examples.

Proposed training: Adam, learning rate 0.001, 10 updates of batch size 16 after each
new valid target. Sample transitions uniformly with replacement from all completed
transitions in this run. There is no historical dataset or cross-run model state.
Record these budgets separately from the comprehension observer's existing budget.

Use current-image selection alone until 10 valid targets have arrived. Continue
logging the predictor's forecasts and errors during this initial period, but flag
that they did not influence selection. Thereafter use gamma=1 by default. The
control uses gamma=0 throughout; validate gamma as finite and non-negative.
Do not rerank forecasts across candidates: use their predicted scalar values.

## Strategy lifecycle and feedback

Add one complete `OffspringValueSelectionStrategy` through the existing
`choose(images, *, rng) -> SelectionDecision` and `describe()` interface.
The current runner already presents the selected parent at position 0 followed
by eight actual offspring on the next call. A pending selection inside the strategy
can associate that grid with the earlier forecast; no new runner callback is needed
for the proposed image-input experiment.

Require nine RGB candidates and a fresh strategy per chronological run. Check that
the first image on the next call matches the pending selected-parent image. This
detects some misuse, but pixel equality does not prove genotype identity. Saved-run
validation must additionally check the recorded parent/offspring genome links.
Backtracking, resets and arbitrary common-grid playback are not supported mid-run.

Order within each `choose`:

1. Validate the incoming grid and any pending parent association.
2. If a valid prediction is pending, score positions 1–8 with its frozen value
   function. Record the original forecast, realised target and error before any
   predictor update. Admit that completed transition and train the predictor.
3. Score the current grid with the live comprehension observer and current novelty
   reference. Forecast each candidate's offspring value with the same predictor
   weights. Newly observed feedback is available for this decision.
4. Fix the choice using current value plus the forecast term if enabled. Preserve
   Experiment 1's uniform first choice and first-maximum rule for subsequent ties.
5. Save the chosen parent's inputs, forecast, and frozen scoring context for the
   next call. Copy observer weights and BatchNorm buffers before current-grid
   training. Use a separate inference-only snapshot; never temporarily overwrite
   the live observer or let target scoring update its BatchNorm state.
6. Train the live comprehension observer on all distinct displayed images,
   including rejected candidates, exactly as Experiment 1 does. Advance the
   novelty reference. Return copied decision records and training metadata.

Only one frozen observer snapshot is needed at a time. Once its target is measured,
retain the scalar label and predictor inputs and release that snapshot. Freeze the
reference arrays and mean image too; later updates must not mutate them.

The initial grid has no informed value function, so its parent-to-child transition
does not provide a training label. The final selected parent has no observed brood
within the run budget. Do not invent its target or breed another generation to
complete it. S decisions therefore yield max(S-2, 0) valid completed targets: 98
for a 100-decision run. The existing 9*S presentations and 9+8*(S-1) genomes remain.

## Control, reproducibility and records

Use two configurations of the new strategy:

| Condition | Choice rule | Predictor training |
| --- | --- | --- |
| Current-image control | gamma=0 | Learns and logs forecasts, but cannot affect selection |
| Predicted-offspring selection | gamma=1 after the initial learning period | Same initialization and update budget |

The control makes it possible to measure prediction under current-image selection
with the same instrumentation. Verify that gamma=0 reproduces the existing
`NoveltyPredictabilitySelectionStrategy` choices, observer states and candidate
trajectory for matched settings and seeds.

Preserve the existing selection-RNG draw sequence for the comprehension observer.
Derive predictor initialization and per-generation update seeds by a recorded,
versioned SHA-256 namespace from the observer's run initialization seed and generation.
Do not consume extra selection draws, use unrelated persistent RNGs, or change
global RNG state. Initialize on CPU and sample with temporary CPU generators;
support CPU and MPS computation, recording the chosen device.

Keep decision scores as the actual selection preference vector. Use stable
evaluation columns for raw novelty, comprehension, current value and predicted
offspring value, with explicit availability/use flags and finite placeholders.
Metadata should additionally record:

- All candidate forecasts, gamma, learning-period status and transition count.
- Origin generation/selected position/image hash; original forecast and predictor
  state hash; frozen observer/reference hashes and rank references.
- On arrival: origin link, the eight raw child measurements and fixed-reference
  scores, their mean target, and the pre-update forecast error.
- Predictor inputs/context, replay size, seeds, sampled-transition hashes, losses,
  update counts and model hashes before/after training.
- Comprehension training records, target-scoring cost, predictor inference/training
  cost and snapshot cost separately. Extra model work is not extra breeding.

Store delayed outcomes on the receiving decision, linked to their origin; do not
mutate earlier events. Reports identify the first ineligible and final unobserved
transitions explicitly. Saved images/events reconstruct predictor inputs; avoid
embedding model weights or image arrays in JSON. Fresh replay must reproduce the
observed targets, forecasts, choices and state hashes within the tested environment.
These records are not resumable checkpoints, and cross-device equality is not promised.

## Code touchpoints

| Location | Change |
| --- | --- |
| `selection_strategies.py` | Complete new strategy; minimal internal sharing of current-value measurement and observer updates |
| `offspring_value.py`, new `offspring_prediction.py` | Fixed-reference ranks and frozen target; lazily imported scalar predictor (transition replay owned by strategy) |
| `image_predictability.py` | Narrow inference-snapshot helper if needed; preserve existing observer behaviour |
| `experiments/run_selection.py` | Thin dispatch for offspring-value selection and its specific options |
| New `experiments/compare_offspring_value.py` | Two-condition batch through the existing runner |
| `selection_comparison.py` or a focused companion | Explicit condition lists and forecast/outcome reporting; preserve Experiment 1 reports |
| New `notebooks/07_offspring_value.ipynb` | Target inspection, learning diagnostics and complete comparison results |
| `tests/`, `README.md`, `AGENTS.md`, `THOUGHTS.md` | Contracts, replay checks, commands, lifecycle and experiment status |

Keep breeding, rendering, the runner interface and session schema unchanged.
Use existing metadata and artifact conventions. No configurable scorer/selector
pipeline, generic feedback framework, migrations or compatibility wrappers are
needed. Existing run artifacts and Experiment 1 notebooks/settings remain intact.

## Delivery phases and review gates

### Phase 1 — make the target inspectable

**Red:** test reference-rank equivalence on original candidates, ties/bounds and
non-finite rejection. Show that reranking children collapses mean-rank information,
whereas a fixed comparison can distinguish better and worse broods. Test frozen
observer/reference independence, parent exclusion and duplicate-child weighting.

**Green:** implement target helpers and the notebook's target inspection. Use short
recorded or freshly generated factual transitions, displaying the original grid,
its novelty mean, the selected parent, actual offspring, raw measurements and target.
The observation must belong to the recorded parent; an alternative suggested
selection does not inherit that parent's offspring. Existing runs are diagnostic
fixtures only, not predictor pretraining data.

**Exit:** target calculations match hand-checkable fixtures and remain unchanged
after the live observer learns. Inspect saturation and non-constant target variation.
If the target is uninformative, resolve that before adding a predictor.

**Pause for user review of the target and notebook before Phase 2.**

### Phase 2 — online predictor as a complete selection strategy

**Red:** test feedback association, score-before-update error records, fresh-run
isolation, initial learning period, transition replay, first/last transition handling,
and absence of labels for rejected parents. A small controlled fixture must show
that the scalar model can learn and that updates change its weights. Verify gamma=0
parity, RNG isolation, immutable records and saved-run replay.

**Green:** implement the predictor, new strategy, CLI and records. Run short actual
breeding sequences on CPU and MPS. Inspect predictions, realised labels and losses
before launching the larger comparison. Preserve the comprehension observer's
existing training and masking behaviour.

**Exit:** a fresh replay reproduces forecasts, targets and choices; no prediction
uses its own future label. Standard artifacts and ancestry are complete. The
control matches Experiment 1 and predictor draws cannot perturb breeding or
observer draws. Run the full suite and execute the inspection notebook. Time both
conditions and record costs before accepting the proposed run budget.

**Pause for user code and prediction-inspection review before Phase 3.**

### Phase 3 — matched comparison and interpretation

Proposed pilot: two conditions, seeds 7/8/9, 100 decisions each: six new runs.
Use MPS, size 96, mutation strength 0.2 and topology enabled. Keep comprehension
weight 0.5 and the existing observer's 20 Adam updates/batch size 16/lr 0.001.
Use fresh strategies, predictors and output directories. The previously observed
near-black seed-9 result remains a useful failure case; do not omit it. Defer a
gamma sweep, architecture sweep and longer runs until these results justify them.

**Red:** test report assembly against fixtures with missing/unobserved targets,
initial-period forecasts, failures and repeated images. Verify that all runs and
final endpoints are included, and errors are calculated from the original forecast.

**Green:** report prediction error before each update, separately during and after
the initial learning period. Compare with two inexpensive forecasts recorded at
the same origin time: the mean of targets already observed (0.5 when none exist),
and the selected parent's current value. Show MAE/MSE, target variance, boundary
saturation and errors over time. Low error on an almost-constant target is not a
successful forecast if a constant baseline does as well.

Show all six final images and full trajectories, selected repeat/parent-retention
rates, pixel diversity, raw measurements and computational costs. Report how often
the forecast changes the chosen parent relative to current-value selection on
that same grid. Retain flat/near-black and noisy outcomes. Do not compare raw target
levels across changing scoring contexts as absolute interestingness.

Only selected parents reveal outcomes. Neither forecast accuracy on those parents
nor a low regression loss establishes correct rankings for rejected alternatives.
After policy choices diverge, the two conditions see different images. Report this
selection bias; do not invent counterfactual labels or add unrequested offspring probes.

**Exit:** every completed run has 900 presentations, 801 genomes and 98 valid targets;
all standard artifacts and costs are accounted for. Prediction failure is a valid
experimental result. Report separately whether forecasts beat simple baselines and
whether using them changes exploration; neither implies improved interestingness.

Audit delivery against this plan, THOUGHTS.md, the accepted Experiment 2 checklist
and conversation. Surface unintended drift. Ask whether a separate
WRAP_UP_EXPERIMENT_2.md is wanted, preserving the Experiment 1 summary.
**Pause for final user code/results review.**

## Research relationship and limits

[Evolvability Search (Mengistu, Lehman and Clune, 2016)](https://researcher.itu.dk/en/publications/evolvability-search-directly-selecting-for-evolvability-in-order-/)
selects for behavioural diversity of immediate offspring. This experiment instead
learns a forecast of their mean value from ordinary selected-parent outcomes. It
neither reproduces that method nor directly measures offspring optionality.

The practical motivation overlaps with
[surrogate-assisted evolutionary computation (Jin, 2011)](https://www.sciencedirect.com/science/article/pii/S2210650211000198):
use a learned approximation to inform evolutionary selection. Here the prediction
target is a future brood's contextual value, rather than the candidate's own fitness.
The architecture, target convention and protocol above are project-specific proposals.

Experiment 1 already showed that current-image predictability can favour nearly
black images, and that 32x32 observation can miss visible texture. Forecasting that
same value need not fix either problem and may reinforce them. The first question
is whether useful next-generation forecasting can be learned from fewer than 100
selected-parent outcomes; broader claims require separate experiments.
