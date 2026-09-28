# Experiment 1: novelty plus current-image predictability

Status: accepted on 28 September 2026; Phases 1–2 implemented and verified.
Phase 3 comparison workflow and nine-run MPS pilot are complete and verified.
Ready for final code/results review. See TODO.md and WRAP_UP.md for checks and findings.
Source: [THOUGHTS.md](THOUGHTS.md), with the decisions below confirmed during
planning on 28 September 2026. Implementation progress is tracked in TODO.md.

## Objective and agreed scope

Test whether adding current-image predictability to novelty changes what CPPN
evolution discovers. Deliver complete selection strategies through the existing
`choose(images, *, rng) -> SelectionDecision` and `describe()` interface, using
the existing runner, breeding loop and session format.

The user confirmed:

- Comprehension means predicting withheld parts of the current image using an
  observer trained on earlier images. It does not mean predicting mutations.
- All displayed images, including rejected candidates, are eligible for subsequent
  training. Start fresh in every run, without historical CPPN pretraining.
- Compare randomly initialized and pretrained versions of the same ResNet18
  observer, with the same prediction head, target and online training. This later
  clarification extends the original scratch-only choice. Pretrained weights are
  an experimental condition, not a different prediction task.
- Implement experiments as complete selection strategies, consistent with random
  and ImageNet selection. Keep evaluation, scoring and choice inside each strategy.
- Measure novelty as pixel distance from the mean image of the previous generation.
  Reserve embedding-based novelty for a later experiment.

The first comparison is novelty alone, novelty plus scratch-observer predictability,
and novelty plus pretrained-observer predictability. Random selection is already
implemented and is not part of this experimental comparison.
Learning progress, historical CPPN pretraining, offspring-value prediction, VLMs and
goal adaptation are outside this batch. Predictability is a limited proxy for
comprehension; this does not reproduce compression progress or establish UFR.

The settings below are proposed pilot defaults, not user-confirmed numerical
choices or a frozen scientific protocol. Change them explicitly at a phase review
if inspection gives a reason; record changes before comparing runs.

## Strategy design

Add `NoveltySelectionStrategy` and `NoveltyPredictabilitySelectionStrategy`, each
implementing the existing protocol. They may share internal pixel-distance,
previous-grid state and numerical helpers. Give the combined strategy an explicit
`observer_initialization="random" | "imagenet"` option: these are two configurations
of the same complete selection strategy. Do not introduce a configurable scorer/selector
pipeline, a second breeding runner, or changes to the protocol for future experiments.

Each strategy instance owns its state across successive grids. Construct a fresh instance
for every run and every independent inspection replay. Calling `choose` advances
that state, so a notebook preview must not reuse its instance for a new run.
Document this lifecycle in the protocol and examples; do not add reset/resume hooks.

### Novelty

Start with direct pixel differences. Convert the displayed RGB images to floating
point in [0, 1] before subtracting; use their full rendered resolution, without
cropping, resizing or feature extraction. Require a consistent image shape within
each strategy instance. Both strategies use the same calculation:

```text
previous_mean = mean(previous_generation_images, axis=0)
novelty(image) = mean((image - previous_mean) ** 2)
```

The first mean averages corresponding RGB pixels across the previous generation's
nine displayed images, producing one mean image. The second mean averages squared
differences over pixel positions and RGB channels, giving a score in [0, 1].
Novelty-only selection chooses the candidate with the highest score. The combined
strategy uses this same novelty score alongside comprehension.

- Compare every current candidate against the same previous-generation mean.
- Include all nine previous candidates equally, including their retained parent
  and any duplicates. Do not deduplicate before computing the mean.
- After fixing the choice, replace the reference with the current grid's mean
  for the next decision. Older generations do not contribute to this reference.
- An image equal to the previous mean has zero novelty. A repeated image or
  retained parent can have positive novelty if it differs from that mean.
- With no previous generation, give all candidates zero novelty and choose
  uniformly for the first decision.

This measures departure from the previous generation's average appearance. The
mean may resemble none of its members, and the score does not penalize revisiting
older images. Those are properties of this chosen local comparison, not bugs.
Novelty needs only the previous mean image and NumPy, with no model or weights.
The observer's replay dataset remains separate and still includes all earlier
distinct displayed images.

Keep nearest-neighbour cosine distance in frozen ResNet18 `IMAGENET1K_V1` pooled
features in reserve, with L2 normalization and recorded preprocessing. A later
experiment could compare that approach against pixel distance; do not implement
it in this batch.

### Observer and comprehension

Use the same ResNet18-based predictor on 32 x 32 RGB images in [0, 1], resized
without cropping. For scoring, divide each image into sixteen 8 x 8 tiles. Make
sixteen input variants, hiding one tile in each. Supply masked RGB plus a binary
visibility channel; never supply the hidden RGB values through another input path.
Use the same masks for every candidate and decision.

Use the full ResNet18 backbone through global average pooling, replacing the
classifier with a linear head from 512 features to `3 * 32 * 32` outputs, followed
by sigmoid and reshaping. Both conditions predict hidden RGB pixels, not a learned
scalar value. Compute the scalar comprehension score from those predictions.
This simple reconstruction head is a pilot choice; it is identical in both arms.

Normalize visible RGB using the ImageNet mean/std in both conditions, then replace
hidden normalized RGB with zero. Extend the first convolution to accept the mask
as a fourth channel. Initialize its additional-channel weights to zero in both
conditions. Keep original unnormalized RGB only as the loss target. For the
pretrained condition, load `IMAGENET1K_V1` backbone weights and BatchNorm state;
for the scratch condition, retain Torchvision's random initialization. Give both
the same newly initialized reconstruction head for a matched observer seed. All
backbone and head parameters train online in both arms. Use evaluation mode during
scoring so BatchNorm cannot update on the current grid; use training mode only
during the subsequent updates. Record architecture, initialization and preprocessing
in `describe()`.

Full ResNet18 is the first architecture; a sweep over truncated stages is deferred.
ResNet18 is used only for the observer in this batch. Novelty is computed directly
from pixels and cannot change as a consequence of observer training.

Compute mean squared error only over the hidden pixels, averaged across masks and
RGB channels. The raw comprehension measure is `1 - masked_mse`. All candidates
use the same observer weights before any training on that grid. This measures
accuracy now, not improvement caused by training on the candidate.

After fixing the decision, admit every new distinct displayed image to a replay
dataset. Train with Adam, learning rate 0.001, for 20 updates of batch size 16 per
decision, sampling images uniformly with replacement and one tile uniformly per
training example. All earlier distinct images remain eligible; repeats do not add
extra copies. Training budgets are fixed even when the grid contains repeats.
The first update therefore uses the first grid after its uninformed choice.

The retained parent may already have been trained on. Record whether each candidate
was previously seen, and separate new-image errors from repeat errors in analysis.
Low error on retained images is not evidence of generalisation. Spatial masking
does not make a repeatedly encountered image an unseen test example.

### Combining the scores

The two raw measurements can have very different spreads. For the initial pilot,
convert novelty and comprehension separately to ascending within-grid percentile
ranks, using average ranks for ties: `(average_rank - 1) / (n - 1)` for `n > 1`.
Use 0.5 for a singleton. All equal values consequently have equal rank scores.

```text
score = (1 - comprehension_weight) * novelty_rank
        + comprehension_weight * comprehension_rank
```

Default `comprehension_weight = 0.5`; validate it in [0, 1]. Novelty-only selection
uses novelty rank. Weight zero in the combined strategy must make the same choices
as novelty alone after identical history, apart from the shared uninformed-first-
decision rule. Use the first maximum on later ties, as ImageNet selection does.
Do not add epsilon exploration in this batch.

For the first decision, set both score components to the same neutral value across
the grid and select uniformly using the supplied RNG. Record comprehension as
unavailable, with a finite placeholder and an explicit availability flag, rather
than allowing random initial predictions to determine selection. The observer
trains after the choice. Exclude unavailable measurements from plots and summaries.

Ranks simplify the first comparison but discard magnitude and can amplify tiny
differences. Preserve raw measurements and inspect this failure mode before runs.
Scores are relative to a grid, history and observer; they are not comparable
absolute interestingness values across generations or runs.

### Exact order within `choose`

1. Validate image shapes and values; identify repeats; prepare normalized pixels.
2. Score novelty against the previous generation's mean. With a trained observer, score all
   candidates with the same pre-update weights and fixed masks.
3. Normalize the two components and fix the choice. First decision is uniform;
   subsequent decisions are greedy.
4. Store the current grid's mean for the next decision. Update seen-image hashes
   and, for the combined strategy, add new distinct images to its replay data.
   Deduplicate replay entries by original pixel bytes, shape and dtype.
5. Train the observer for the fixed budget. These updates affect the next decision,
   never the scores just used. Finish all updates before returning.
6. Return the fixed choice, its pre-update measurements/scores, and update metadata.

This requires no offspring feedback or genotype access. Keep training cost visible:
the existing evaluated-image counter counts scored candidates, not masked forward
passes or optimizer updates.

## Reproducibility and records

Use the supplied selection RNG for choices and to obtain recorded initialization
and per-update seeds. Seed temporary Torch generators from those values for model
initialization and training sampling; do not retain an unrelated random stream or
change global Python, NumPy or Torch RNG state. Generate the first uniform choice
before drawing observer seeds so all conditions can start from the same
selected root for a matched selection seed. CPU is the default observer execution
target. During Phase 3 the user requested
Apple Silicon MPS support; expose an explicit device choice while retaining CPU
initialization and sampling. Record the device and test short same-device replay;
do not assume equality across devices. The matched pilot uses MPS throughout,
with the earlier interrupted CPU batch retained separately.

Keep `SelectionDecision.scores` as the final preference vector. Use stable
`Evaluation` columns for raw novelty and comprehension, and JSON metadata for:

- Raw masked errors, component ranks and availability flags for every candidate.
- Ordered image hashes, repeat flags, reference-generation index and replay sizes
  before and after update. The previous grid's saved images reconstruct its mean.
- Observer initialization seed, model-state hashes before/after training, update
  seed, update count, sampled-example count, loss summaries and elapsed scoring/
  training time. Record optimizer configuration and exact sampling rules.
- Pixel-distance definition, observer preprocessing/configuration and versions;
  pretrained observer checkpoint hash when applicable.

`describe()` must describe configuration before the first `choose`, because the
runner records it at initialization. Put subsequently resolved seeds and state in
decision metadata. Ensure recorded arrays and metadata are copied rather than
mutated by later updates.

Saved source, ordered candidate images, configuration, seeds and update records
must support replay from a fresh observer in the same environment. Verify scores
and model hashes in a short replay test. Do not store model weights inside JSON or
add resumable sessions in this batch; audit snapshots retain their existing meaning.
If deterministic replay cannot be achieved, resolve and document that limitation
before accepting the phase, rather than claiming the observer state is recoverable.

Preserve existing genomes, rejected images, ancestry, grids, checkpoints and final
`selected.png`. Do not add a separate best-image archive. Existing saved sessions
and random/ImageNet behavior remain unaffected; no schema migration is needed.

## Code touchpoints

| Location | Intended change |
| --- | --- |
| `selection_strategies.py` | New complete strategies; stateful lifecycle documentation; lightweight imports |
| `selection_strategies.py` internal helpers | Shared pixel-distance calculation and previous-grid mean; no feature-extractor module needed |
| New `image_predictability.py` | Masking, matched ResNet18 observers, masked loss and explicit online updates |
| `experiment_reporting.py` | Show both component measurements for small evaluation matrices; retain the existing class summary for ImageNet |
| `experiments/run_selection.py` | Thin dispatch for `novelty` and `novelty-predictability`; weight and observer-initialization options validated only for the latter |
| New `notebooks/05_novelty_predictability.ipynb` | Component inspection, training diagnostics, common-grid replay and short comparisons |
| `tests/` | Strategy, observer, CLI and saved-run/replay checks |
| `README.md`, `AGENTS.md`, `THOUGHTS.md` | Usage, lifecycle and confirmed experiment choices; distinguish implemented methods from later proposals |

Keep `breeding.py`, the renderer, the runner's strategy interface and session schema
unchanged. Existing metadata carries the additional records. Reuse the existing
`imagenet` optional dependency stack for the ResNet18 observers and explain that
installation requirement; novelty-only, random and human paths must import and
run without Torch.
Avoid modifying existing notebook experiments and their settings.

## Delivery phases and review gates

### Phase 1 — novelty strategy through the existing runner

**Red:** add known-pixel tests for mean-image construction, squared distance,
equal weighting including duplicates, a shared reference across the current grid,
first choice, ties, immutable records and RNG isolation. Cover integer-overflow
avoidance, mismatched shapes and replacement of the reference after each decision;
older generations must not affect it.
Add CLI dispatch and minimal saved-run contract tests without model downloads.

**Green:** implement the previous-grid mean and `NoveltySelectionStrategy`, add
CLI dispatch, and show candidates alongside the reference mean image and their
distances in the new notebook.

**Exit:** novelty runs through `run_experiment` and saves the normal artifacts;
controlled score tests pass; distances are finite and repeatable; an image matching
the reference mean has zero novelty, while a repeated image need not. Existing
random/ImageNet tests pass. Novelty-only and random selection run without Torch.
Inspect reference means and sensitivity to
colour changes, translations and noise.

**Pause for user code review before Phase 2.**

### Phase 2 — online predictability strategy and audit records

**Red:** test that hidden pixels cannot enter observer inputs and only masked
pixels contribute to loss. Use controlled observers to verify all candidates are
scored before training, first-decision handling, equal masks, score weighting,
all-candidate admission, deduplication, fixed updates and fresh-instance isolation.
Add a bounded real-model test that training changes weights and reduces error on a
simple learnable fixture; do not assert that arbitrary images are interesting.

Test matched architecture/head initialization and target masks across both arms,
correct pretrained-weight loading using local controlled weights, and that neither
current-grid scoring nor observer training mutates the stored reference mean.

**Green:** implement both observer initializations and the combined strategy, seed discipline,
component metadata, CLI dispatch and reporting. Extend the notebook to show masks,
predictions, raw errors, rankings and update costs. Add saved-run replay coverage.

**Exit:** weight zero matches novelty choices on identical histories; scoring a
candidate cannot benefit from the current update; rejected images enter replay;
retained images receive repeat flags; metadata survives round-trip saving. A short
CPU replay reproduces decisions, scores and observer state within the declared
environment. Selection/training draws cannot perturb breeding draws. Run the
appropriate full suite with `uv run --extra imagenet pytest`.

**Pause for user code review and score inspection before Phase 3.**

### Phase 3 — small comparisons and interpretation

First inspect flat colours, gradients, repeated patterns, independent noise and
actual CPPN images. Train diagnostic observers on a separate synthetic training
set and evaluate fresh examples from each family. Show predictions and errors
alongside simple visible-region-mean filling as a diagnostic baseline. This checks
what the observer learns; it does not prescribe a universal ordering of images.

Replay a short common sequence of saved grids into fresh strategies to compare
choices under identical exposure. Each strategy receives every grid even when its
choice differs from the recorded trajectory; label this as score inspection, not
an evolutionary outcome. Do not use a single empty-history grid as the comparison.

Then run a pilot of 3 breeding seeds (7, 8, 9), 100 decisions per run, for novelty
and novelty-plus-predictability with each observer initialization: 9 runs
total. Use the same recorded settings
and derived selection seeds, size 96, mutation strength 0.2 and topology enabled.
Use fresh strategy instances and new output directories. Keep default combined
weight 0.5; defer a larger weight sweep until these runs justify it. First time a
short smoke run of both observers to confirm the budget is practical before
launching the batch. Full ResNet18 training costs more than the originally proposed
small predictor. If the CPU budget is impractical, reduce the common update/run
budget at this review gate and document it for both arms before comparisons.

**Red:** test comparison assembly against saved fixtures: all runs/seeds included,
final images selected without cherry-picking, correct decision and candidate counts,
and unavailable observer values excluded. Keep statistical analysis descriptive.

**Green:** complete the notebook comparison and usage documentation. Present all
final images using the same layout, with full trajectories available. Summarize
repeat rates, raw novelty and new-image prediction error where recorded, and
pixel-scoring/observer-inference/training cost separately. For cross-strategy image diversity,
compute the same post-run measure for all three conditions without changing selection.
Label pixel-distance summaries as measurements of the selected proxy; visual
inspection must assess whether those differences correspond to useful structure.

**Exit:** runs meet `9*S` candidate presentations and `9+8*(S-1)` generated genomes;
all standard artifacts are present; no seeds or unattractive outcomes are omitted.
Report whether the combined method changes discovery, repeats or smooth-image
preference, without requiring a positive result. Scores optimized by selection
are not independent evidence of success. The final image is the agreed reporting
endpoint, not proof that absolute interestingness improved monotonically.

Review delivered work against THOUGHTS.md, PLAN.md, the accepted TODO.md and the
conversation, and surface any unintended drift. Ask whether a WRAP_UP.md summary
is wanted. **Pause for final user code/results review.**

## Interpreting the initialization comparison

The user clarified that the added experiment should predict exactly the same
target with the same architecture, changing initialization. Compare scratch and
pretrained observers on the same chronological grids, training examples, masks,
head initialization and optimizer budget before comparing independent breeding
runs. Their shared-grid prediction errors test whether pretraining helps the
observer; their breeding outcomes test whether that difference helps selection.

Lower prediction error need not improve selection: it may alter relative rankings
or make nearly every image predictable. Show both accuracy and choices. Once
breeding choices diverge, the observers receive different images, so those runs
alone cannot isolate the effect of initialization on prediction accuracy.

## Research basis and practical limits

- [Context Encoders (Pathak et al., 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/Pathak_Context_Encoders_Feature_CVPR_2016_paper.html)
  provides a precedent for learning visual features by predicting missing image
  regions. Our online ResNet18 predictor and selection score are proposed adaptations,
  not a reproduction of its architecture or results.
- [Torchvision ResNet source](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html)
  identifies the pooled representation before the final classifier;
  [ResNet18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18)
  specifies the explicit weights and official preprocessing. These support the
  observer initialization comparison and the reserved embedding-novelty experiment.

Pixel prediction can reward smoothness and memorization. Downsampling can hide
fine structure and noise from the observer. Pixel novelty can reward colour shifts,
translations and noise without new meaningful structure. Rank weighting can elevate
tiny error differences. These are
specific things to inspect in Phase 3, not reasons to claim the proxies measure
human interestingness. Learning progress and evolvability remain separate tests.
