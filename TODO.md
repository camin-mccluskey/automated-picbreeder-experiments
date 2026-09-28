# Selection experiments: implementation checklist

PLAN.md accepted on 28 September 2026. Work follows its phase review gates.

## Phase 1 — pixel novelty

- [x] Red: test previous-grid means, distance/ranks, repeats, first choice, ties,
  shape validation, detached records and RNG isolation.
- [x] Red: test CLI dispatch, Torch-free execution and saved-run replay.
- [x] Green: implement `NoveltySelectionStrategy` through the existing interface
  and add thin CLI dispatch; keep breeding and persistence unchanged.
- [x] Add a notebook showing reference means, distances and a short CPPN run.
- [x] Update usage and experiment notes to reflect agreed scope and actual status.
- [x] Verify targeted tests, existing suite and executed notebook; inspect outputs.
- [x] Pause for user code review before Phase 2 — user authorized continuation.

Verification: 53 targeted tests and 133 full-suite tests passed in the installed
environment (`uv run --no-sync --extra imagenet pytest -q`, with a writable UV
cache). Notebook 05 executed successfully; its four-decision run generated 33
genomes and saved-image replay reproduced all choices and raw distances. The
synthetic checks expose repeated-image and translation sensitivity as expected.
Phase 1 matches the accepted scope; no breeding/runner/schema changes were needed.

## Phase 2 — online predictability

- [x] Red: test masks/loss, score-before-training, replay admission/deduplication,
  fixed budgets, matched initializations, RNG isolation and saved replay.
- [x] Green: implement matched random/pretrained ResNet18 observers and combined
  strategy, metadata and CLI configuration.
- [x] Extend reporting and notebook with predictions, errors, ranks and costs.
- [x] Verify weight-zero equivalence, observer learning, reproducibility and full suite.
- [x] Pause for user code and score-inspection review before Phase 3 — user approved.

Verification: 153 tests passed. Notebook 05 executed with the real cached pretrained
checkpoint and both observer initializations; its prediction mosaics and component
contact sheets were inspected. Two-decision CLI smoke runs for each initialization
used the default 20 updates/batch size 16 and saved all 17 genomes. Fresh replay
matched choices, scores, losses, sampled-batch hashes and model hashes for both
arms. Unit tests also verify hidden-pixel isolation, unchanged BatchNorm during
scoring, matched heads, fixed budgets and unchanged breeding RNG behavior.

The notebook's illustrative observer section explicitly uses a smaller 2-update,
batch-size-4 budget for inspection. The strategy defaults and planned pilot remain
unchanged. No claim about relative interestingness follows from these checks.

## Phase 3 — comparisons

- [x] Red: test complete comparison assembly and unavailable-value handling.
- [x] Green: add structured-image diagnostics and common-grid comparison.
- [x] Time both observers, then run the agreed pilot (3 conditions x 3 seeds,
  100 decisions) or record a reviewed common budget adjustment.
- [x] Present all final images and trajectory diagnostics, including failures.
- [x] Review implementation against THOUGHTS.md, PLAN.md, TODO.md and conversation;
  surface unintended drift.
- [x] Ask whether an optional WRAP_UP.md summary is wanted — user requested it.
- [x] Produce the requested short WRAP_UP.md with decisions, findings and limitations.
- [ ] Final user code/results review — implementation and verification complete, ready for review.

Verification: all 165 tests passed with GPU access. The comparison notebook executes
with run-launch flags false. The MPS pilot completed all 9 runs at the original
100-decision budget: 8,100 candidate presentations, 7,209 saved genomes/images,
900 grids, 12,000 observer updates and 192,000 training examples. All final-image
files match the last selections; all conditions share the same initial choice for
each matched seed. Standard artifacts, checkpoints and source snapshots are present.

The user-requested MPS addition kept CPU initialization and sampling. Short fresh
MPS replays matched scores, batches and model hashes for both initializations.
The earlier CPU batch was interrupted cleanly and retained separately; its outcomes
are not mixed into the MPS comparison. No training/run budget was reduced.

Scope audit: existing complete strategies and runner are preserved; current-image
prediction only; no offspring prediction, learning-progress measure, embedding
novelty, VLM or new random-selection comparison. Cost instrumentation separates
combined scoring from training, but not pixel scoring from observer inference;
novelty-only records total run time. This reporting limit is explicit in WRAP_UP.md.
Results are descriptive and include the near-black convergence of both combined
conditions for seed 9. No positive interestingness or UFR result is claimed.
