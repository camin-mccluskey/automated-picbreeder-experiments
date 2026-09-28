# Experiment 2 implementation checklist

PLAN_EXPERIMENT_2.md accepted, including its proposed defaults. Work follows the
plan's phase review gates; the Experiment 1 checklist remains separate.

## Phase 1 — inspect the offspring target

- [x] Red: test fixed-reference ranks, ties/bounds, invalid inputs and varying brood targets.
- [x] Red: test frozen observer/reference independence, parent exclusion and duplicate weighting.
- [x] Green: implement target helpers and an inference-only observer snapshot.
- [x] Add notebook 07 showing factual parent/offspring transitions and target calculations.
- [x] Execute the notebook; inspect target variation, saturation and saved ancestry.
- [x] Run relevant tests and the full suite; update documentation and scope status.
- [x] User review of the target and notebook before Phase 2.

## Phase 2 — online prediction and selection

- [x] Red: test delayed feedback, original forecasts, initial learning period,
  transition replay, first/last transitions, RNG isolation and gamma-zero parity.
- [x] Green: implement the scalar predictor and complete offspring-value strategy.
- [x] Add CLI options, learning records, prediction diagnostics and saved replay.
- [x] Verify short CPU/MPS runs, source/artifact completeness and computational cost.
- [x] User code and prediction-inspection review before Phase 3 (user authorized remaining work).

## Phase 3 — comparison

- [x] Test and implement complete reports, forecast baselines and missing-target handling.
- [x] Run the six-run pilot after the Phase 2 review and budget check.
- [x] Report all endpoints/trajectories, prediction errors, exploration measures and costs.
- [x] Audit delivery against the plan, THOUGHTS.md, this checklist and conversation.
- [x] Ask whether WRAP_UP_EXPERIMENT_2.md is wanted — user declined; README and report suffice.
- [ ] Final user code/results review — implementation, pilot and results delivered; awaiting user review.

## Phase 1 verification

All 189 tests passed with Apple GPU access, including CPU/MPS snapshot isolation.
Notebook 07 executed successfully on MPS. Three six-decision runs produced 12
valid offspring targets (range 0.4453125–0.84765625, standard deviation 0.1104792).
Across their novelty/comprehension rank components, 28.65% reached 0 or 1; one
brood reached 81.25%, so boundary saturation remains an inspection concern.

Each diagnostic saved 49 genomes/images and 54 candidate presentations. Recorded
ancestry verifies that the eight target images belong to the selected parent.
All choices and post-update observer hashes match the corresponding first six
steps of the original Experiment 1 scratch-observer runs, for seeds 7/8/9.
Snapshots preserved the original candidate scores after live observer training.

The executed notebook, source copy, report and all run artifacts are in
[runs/offspring-target-inspection-20260928T220137-622671Z](runs/offspring-target-inspection-20260928T220137-622671Z).
The target is non-constant in this small inspection; this does not establish that
it can be predicted. The user passed the target/notebook review and authorized Phase 2. No breeding, renderer, runner or session-schema changes were made.

## Phase 2 verification and findings

The user authorized Phase 2 after confirming the aim: learn which parents produce
valuable offspring. Implemented `OffspringValueSelectionStrategy` and a separate
scratch scalar predictor, using the accepted architecture, scoring context,
transition replay and default budgets. The pure NumPy target helper remains in
`offspring_value.py`; the Torch model lives in `offspring_prediction.py` so imports
stay lightweight. This file split does not change the experiment.

All 211 tests passed with Apple GPU access. CPU and MPS checks verify delayed original-forecast errors, exact fresh replay,
ancestry and gamma-zero parity with Experiment 1 (choices, observer hashes,
selection RNG and generated images). A controlled regression fixture learns its
target on both devices. The strategy also passes controlled tests for warm-up,
forecast-driven choice changes, repeated-parent transitions, immutable records,
invalid chronology and first/final missing targets. No runner, breeding, rendering
or session-schema changes were needed for this phase.

Notebook 07 was executed on MPS with three target-inspection runs and two new
fourteen-decision online runs at the default budgets. Both online runs saved 126
presentations, 113 genomes and twelve valid targets. They took 18.27 and 17.49
seconds, respectively, and both replayed exactly, including forecasts, target
values and model hashes. Gamma 1 was used for three choices but changed none of
them, so both conditions followed the same trajectory in this short inspection.

Prediction MSE was 0.027128; the running-mean baseline achieved 0.016150 and the
current-parent-value baseline 0.018840. Prediction MAE was 0.096949. These twelve
observations do not demonstrate useful predictive accuracy. The target standard
deviation was 0.118007. No outcomes were omitted and no rejected parent received
an invented target.

Results, all forecasts, both final images, the executed notebook and source copy:
[runs/offspring-target-inspection-20260928T222120-932319Z](runs/offspring-target-inspection-20260928T222120-932319Z).
The approximate two-minute per-100-decision extrapolation from these short runs
suggests the proposed Phase 3 budget is practical, but rendering, growing replay,
and record-saving costs may change with run length. At that Phase 2 checkpoint, the six-run pilot had not started. The user subsequently
passed the review and authorized Phase 3; completion is recorded below.

A separate CPU CLI smoke run used four decisions, size 16, one observer update
(batch 2), two predictor updates (batch 2), and a one-target warm-up. It completed
with 33 genomes and two targets in `runs/offspring-phase2-cpu-smoke`. These reduced
settings are a CLI/device check and do not replace the accepted pilot defaults.

## Phase 3 implementation

The user approved finishing all remaining work. Added the focused
`offspring_comparison.py` reporter/batch and `experiments/compare_offspring_value.py`
CLI. Reused the existing runner and Experiment 1 outcome measurements, preserving
both strategy implementations and their default budgets. Reports retain all
scheduled runs, failures, final images and complete selected trajectories. They
verify delayed forecasts, past-only baseline inputs, eight-child target means,
ancestry and expected counts. Errors are recomputed from original forecasts.

Reports separate warm-up from later forecasts using the number of targets available
at forecast time in both arms, and include ten-outcome windows plus first/last
20-outcome summaries. Training loss is separately recorded and is not used as
forecast accuracy. The full suite passed: 219 tests with Apple GPU access.
The six-run MPS pilot completed in `runs/experiment2-pilot-mps` with the accepted
100-decision, three-seed protocol. The user declined a separate wrap-up document.

## Phase 3 completion and scope audit

All six runs completed: each has 900 presentations, 801 genomes/PNGs, 100 grid
images, ten audit checkpoints and 98 valid targets. Verified every saved PNG,
final-selected-image identity, source-snapshot hash and parent/child link. All
three controls exactly match their corresponding Experiment 1 runs across 100
choices, observer hashes and 801 image files. The total recorded run time was
939.89 seconds (15.7 minutes). No runs or outcomes were omitted.

The forecast term was enabled for 89 choices per active run and changed 19, 11
and 22 choices for seeds 7/8/9. Of those 52 changes, 43 resolved current-value ties.
Pixel diversity rose from 0.09213 to 0.40952 (seed 7), 0.22096 to 0.30149 (seed 8),
and 0.00040 to 0.24891 (seed 9). These are proxy differences, not a validated
improvement in interestingness or evidence of correct counterfactual rankings.

Across all post-warm-up outcomes, the active predictors' MSEs were 0.02064,
0.01630 and 0.01789, versus running-mean baseline MSEs 0.01724, 0.01268 and 0.01391.
Thus none of the active runs beat that baseline over the complete later period.
Error decreased from the first 20 to the last 20 outcomes in five of six runs;
for the active runs, seed 7 beat the baseline in the final window, seed 8 worsened,
and seed 9 approximately tied it. The reports also show changing target variance.
The seed-9 control remained near-black; the active endpoint was visibly noisy.
Both are included alongside the other four endpoints and all selected trajectories.

The comparison report can be rebuilt without Torch or new evolution. Executed the
new comparison-reading section of notebook 07 and preserved its output and full
notebook source with the pilot. Earlier target/online-inspection sections were
already executed during Phase 2. `README.md`, `AGENTS.md`, `THOUGHTS.md` and the
accepted plan now describe the completed implementation, commands and findings.

Delivery matches the accepted scope: existing strategy protocol, unchanged
breeding/rendering/runner/schema, scratch online predictor, frozen selection-time
child targets, actual selected-parent feedback only, and the two-condition pilot.
No budgets, seeds or hypotheses were changed after seeing results. The focused
comparison module is the planned companion to Experiment 1's reporting; the
separate Torch predictor module preserves lightweight imports. Learning progress,
historical pretraining, offspring diversity, recursive returns and VLMs remain
deferred. The user declined WRAP_UP_EXPERIMENT_2.md. Only final user review remains.

Artifacts: [full report](runs/experiment2-pilot-mps/REPORT.md),
[all metrics](runs/experiment2-pilot-mps/report.json),
[artifact audit](runs/experiment2-pilot-mps/artifact-audit.json),
[pilot measurements](runs/experiment2-pilot-mps/pilot-findings.json), and
[executed comparison notebook](runs/experiment2-pilot-mps/comparison-inspection.ipynb).
