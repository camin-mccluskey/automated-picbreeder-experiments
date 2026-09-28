# Experiment 1 implementation

The three conditions are pixel novelty alone, novelty plus a randomly initialized
observer, and novelty plus a pretrained observer. Each is a complete selection
strategy in the existing breeding runner. Novelty maximizes pixel MSE from the
previous generation's mean image. Comprehension measures hidden-pixel prediction
accuracy on current candidates, before training on them. Rejected candidates also
enter the observer's replay dataset after the decision.

The two observers share ResNet18 architecture, masks, head initialization and
training budget. Only backbone initialization differs. Combined selection averages
within-grid novelty and comprehension ranks. These are proxies: this implementation
does not measure learning progress or predict future offspring value. Experiment 2
remains planned.

## Delivery and verification

- A batch command runs all three conditions with matched seed sets and budgets.
- Reports retain every scheduled run, including failures, and show final endpoints,
  complete trajectory records, repeat rates, prediction errors and costs.
- Notebook 06 supports report inspection, held-out synthetic checks and identical-grid
  comparisons without launching new runs by default.
- All 165 tests pass, including the real MPS observer test. The comparison notebook
  executes successfully. Short fresh replays match scores, sampled batches and model
  hashes separately on CPU and MPS in this environment.

The user requested MPS during Phase 3. Initialization and sampling remain on CPU;
model computation can use the Apple GPU explicitly. Twenty training updates took
about 0.4 seconds on MPS versus 5–6 seconds on the busy CPU. The interrupted CPU
batch remains in `runs/experiment1-pilot`; the completed comparison is stored
separately in `runs/experiment1-pilot-mps`. Budgets and scientific settings are
unchanged. Cross-device numerical equality is not assumed.

One reporting limitation remains: combined-strategy scoring time includes both
pixel comparison and observer inference, rather than timing them individually.
Training is timed separately. Novelty-only runs record total elapsed time without
a separate scoring timer. No breeding, rendering or session-schema changes were
needed for this experiment.

## Preliminary diagnostics

After 40 updates on 24 synthetic training images, both observers still performed
worse than visible-region-mean filling on all four held-out families: flat colours,
gradients, repeated patterns and noise. The random observer had lower error than
the pretrained observer on every family at this small budget. This tests early
learning, not convergence. On five identical CPPN grids, the combined strategies
differed from novelty alone on several later grids after the shared initial choice.

## Full pilot

All **9 runs × 100 decisions** completed using seeds 7, 8 and 9, 96×96 rendering,
mutation strength 0.2, topology enabled, weight 0.5, and 20 updates of batch size 16.
All 8,100 presentations, 7,209 genomes/images and 900 grids are recorded. Observer
runs used 12,000 updates and 192,000 training examples altogether. Total run time
was approximately 16 minutes on this machine.

- Both combined strategies had lower pairwise pixel diversity than novelty alone
  in every seed. This measures a narrower range of pixels, not worse interestingness.
- Seed 9 exposed a strong failure mode: both combined conditions ended near black.
  Selected-image repeat rates were 22% for scratch and 36% for pretrained, versus
  2% for novelty alone. Their pairwise pixel MSE was approximately 0.0004 and
  0.0003, versus 0.383 for novelty alone.
- Other combined endpoints contain high-frequency texture. Hidden-pixel accuracy
  at 32×32 does not guarantee comprehensibility at the 96×96 display resolution.
- Pretraining showed no early prediction advantage in the matched synthetic check.
  Independent breeding trajectories encounter different images, so their error
  differences do not isolate the effect of initialization.

Every final image and per-seed result is in the local
[pilot report](runs/experiment1-pilot-mps/REPORT.md), with all trajectories retained.
The [diagnostics](runs/phase3-diagnostics-mps/heldout/REPORT.md) and
[common-grid choices](runs/phase3-diagnostics-mps/common-grids/REPORT.md) are separate
checks. Generated run artifacts remain ignored by Git; this summary records the
main findings without treating these proxies as a validated interestingness scale.

See [README.md](README.md#run-the-experiment-1-comparison) for commands and
[notebook 06](notebooks/06_selection_comparison.ipynb) for inspection.
