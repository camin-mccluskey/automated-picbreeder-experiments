# Selection strategy follow-up

Source: the user's 30 September 2026 strategy-audit follow-up and confirmation
that new offspring runs should default to max. The user requested this plan and
then delegated execution. Review checkpoints below are implementation checks,
not additional approval holds. No separate THOUGHTS.md or TODO.md is required.

## Decisions and scope

1. Shorten README into an entry point linking to one Markdown guide per strategy.
   Each guide explains the rule, limitations, runnable terminal examples, all
   applicable options/defaults and when its distinctive mechanism becomes active.
   Shared run options may live in a linked common guide. Add an AGENTS.md rule
   requiring documentation updates with strategy or CLI changes.
2. Investigate whether the masked-image observer can use the full 96x96 image.
   Distinguish the frozen ImageNet classifier from the trainable reconstruction
   observer. Check architecture, checkpoint constraints, mask design and measured
   compute/memory costs. Deliver a source-grounded report and recommendation.
   **Follow-up decision:** after reviewing the measured cost, the user approved
   using 96x96 with sixteen masks. The user subsequently declined an optional
   32x32 baseline: use one fixed 96x96 observer with sixteen 24x24 tiles. Do not
   add an observer-resolution constructor or CLI option. Historical 32x32 remains
   in the benchmark report only; notebook observers use the new fixed size too.
3. Add `offspring_aggregation="max" | "mean"` to both offspring strategies and
   `--offspring-aggregation {max,mean}` to their CLI commands. **Max is the default**
   for new runs, confirmed by the user. Aggregate the eight actual child values
   under the frozen selection-time value function; exclude the retained parent.
   Max trains a forecast of the expected best value in an eight-child brood, not
   the maximum forecast across current candidates or an individual-child forecast.
4. Add `novelty_reference="previous-grid-mean" | "previous-parent"` to every
   novelty-based strategy and the corresponding `--novelty-reference` option.
   Keep `previous-grid-mean` as default. Previous parent means the image selected
   on the preceding choose call, retained in the next grid by the runner. It has
   zero raw novelty against itself. The initial grid has no reference in either
   mode; existing random/confidence first-choice rules remain.
5. Defer comparison with all previously seen images. Do not add a history archive,
   history-distance definition, CLI placeholder, or extra historical storage.

Existing saved sessions remain unchanged and readable under their original
meaning. New metadata records both choices explicitly, with generic target and
reference names where necessary. Update current helper callers and notebooks;
do not add legacy API wrappers, migrate saved runs or reinterpret old mean targets
as maxima. Keep existing actual-mean diagnostics explicitly named as means if
retained. Existing read-only historical-session support remains in scope.

## Scientific and implementation invariants

- One complete SelectionStrategy per run, one retained parent and eight actual
  children; no probing rejected parents, extra offspring or added classifier pass.
- Both aggregations apply to the same fixed-reference child scores. Keep duplicate
  children, original forecasts, delayed errors and every eligible transition.
- Preserve observer warm-up, predictor warm-up, RNG separation, replay and gamma-0
  parity for matching novelty-reference settings. Mean mode preserves the old
  target calculation. Defaults change only where explicitly stated above.
- Freeze the **active selection-time novelty reference image**, whether it is a
  grid mean or previous parent, for offspring targets and predictor inputs. Do not
  substitute the newly selected parent or next grid's reference during feedback.
- Preserve the 21/23-scalar contexts and current-plus-gamma-times-forecast rule;
  each fresh predictor learns only its configured aggregation within that run.
- Novelty ranks, quality definitions, tie handling, HSB rendering and mutation
  remain unchanged. Patch scores can change with observer resolution. Max value
  does not by itself measure diversity or long-term potential.
- Derived `display_novelty` remains previous-display-grid distance across strategies;
  strategy `pixel_novelty` follows the configured reference. Document the difference.
- Preserve unrelated working-tree changes: notebook 01 and deleted main.py.

## Phase 1: documentation structure

Owner: documentation sub-agent.

- Create `docs/strategies/{random,novelty,imagenet,novelty-imagenet,
  novelty-predictability,offspring-value,offspring-value-imagenet}.md`.
- Put common run settings and shared operations in a linked guide; preserve useful
  existing notebook, export, interpretation and reproducibility instructions when
  reducing README. Update links from other docs.
- Include real defaults, valid ranges, device/model constraints, applicable
  Python options and CLI help commands. Show both new configurable choices.
- Use examples that reach active comprehension (decision 11), patch offspring
  forecasts (decision 21), and ImageNet offspring forecasts (decision 12), or
  explicitly disable warm-up and explain the consequence. Label smoke tests.
- Update AGENTS.md, experiment brief and batch-metric wording for new semantics.
- Check examples against the actual parser without loading models; check local
  links and option coverage. A small drift check should catch newly undocumented
  commands/options, without duplicating the implementation in tests.

Exit: concise README links to seven complete guides; all examples parse; guides
match the implemented defaults and no longer advertise mechanisms they never use.

## Phase 2: configurable selection semantics

Owners: novelty sub-agent and offspring sub-agent; root integrates shared files.

### Novelty reference

- First add tests with grids where previous parent and previous-grid mean give
  different distances/choices. Verify first-grid rules, duplicate weighting,
  detached saved reference, pure novelty and confidence endpoint behaviour.
- Carry the actual final selected position into reference updates, including
  offspring-driven selections. Never remember the current-value baseline choice
  when a forecast changes the winner.
- Thread the option through all five novelty-based constructors and describe/
  decision metadata. Use neutral reference-image names in new internal plumbing.
- Test fixed-reference offspring targets and predictor inputs in both modes,
  chronological replay, gamma-0 parity and new-option rejection on unrelated CLIs.

### Offspring aggregation

- First add deterministic child-score fixtures that distinguish max from mean,
  including duplicates and an excluded high-valued retained parent. Invalid values
  must fail before expensive model construction.
- Implement aggregation in both frozen target helpers, training-label selection,
  strategy/predictor descriptions and recorded target context. Record the selected
  aggregation and a neutral target value; never label a maximum as a mean.
- Add the CLI option only to the two offspring commands. Update existing mean
  tests/inspection callers explicitly where their original purpose is mean value.
- Verify warm-up counts, pre-outcome errors, mean-mode targets, max defaults,
  gamma-0 parity, saved-run replay and generic metric/viewer interpretation.

Exit: both options work independently and together; default novelty behaviour is
preserved; new offspring runs default to max; mean remains reproducible explicitly.

## Phase 3: observer-resolution investigation and approved implementation

Owner: root initially; delegate to a research sub-agent when a slot is free.

- Inspect installed Torchvision ResNet18 and primary documentation. Confirm what
  adaptive pooling permits, what pretrained weights require, and which 32x32
  constants belong to our reconstruction head, masks and replay preprocessing.
- Compare two explicit 96x96 mask designs: 16 tiles of 24x24 (same fraction hidden
  and number of inference masks), versus 144 tiles of 8x8 (same absolute tile size,
  nine times as many masked evaluations). Do not conflate their costs/objectives.
- Use an isolated benchmark with real forward/backward passes at 32/96,
  matched batch sizes and device, bounded repeats, no checkpoint downloads and
  synchronization where needed. Record architecture parameter counts, timing,
  memory estimates/measurements and hardware/library versions. Keep unmeasured
  extrapolation distinct from measurements. Synthetic benchmarks are not a claim
  about selection quality or full-run duration.
- Save findings and primary links in `docs/observer-resolution.md`; link it from
  the patch-strategy guides. Remove the one-off benchmark script after the
  investigation, as subsequently requested by the user; retain its findings.
- Implement the user's approved fixed 96x96 observer with sixteen 24x24 masks.
  Update resize, head, mask geometry, scoring, reconstruction, replay and provenance;
  preserve CPU initialization/sampling. No optional baseline or new resolution flag.
- Verify finite training/scoring, hidden-pixel mask coverage and loss, snapshot
  agreement, seeded scratch/pretrained heads and saved-run replay at 96x96.
  Notebook 07 explicitly retains mean targets; notebook observers use 96x96.

Exit: measured feasibility/cost report and tested fixed 96x96 implementation,
with sixteen masks and no superseded observer-resolution option.

## Phase 4: integration and review

Owner: root.

- Coordinate shared-file ownership: novelty owns selection_strategies.py during
  its implementation; offspring owns offspring_value.py/offspring_prediction.py.
  Root subsequently integrates aggregation into selection_strategies.py, both
  options into selection_cli.py, and metrics/notebook changes not delegated.
- Run focused tests during each change, then the full suite once integration is
  complete. Exercise both aggregations and novelty references in small real
  CPU runs through `experiments/run_selection.py`, using temporary output paths.
  Use controlled classifiers for tests; avoid unneeded checkpoint downloads.
- Check saved strategy metadata, target values, reference provenance, unchanged
  parent links, fresh replay and old saved-session readers. Verify documentation
  examples, links, option coverage and diff whitespace.
- Review delivered work against this plan and the conversation. Record completed
  work, validation and intentional deviations here; report unresolved issues.
- Deliver links to the plan, guides and resolution report. A separate WRAP_UP.md
  is optional and will not be created without a request.

Exit: coherent implementation/docs, passing relevant checks, research report,
and no unrelated modifications or generated experiments committed.

## Progress

- Complete: README links seven strategy guides with runnable examples and full
  applicable option coverage; shared guides cover run/model settings. AGENTS.md
  requires keeping strategy documentation synchronized with implementation.
- Complete: both offspring strategies support max (new default) and mean targets.
  All five novelty strategies support previous-grid-mean (unchanged default) and
  previous-parent references. Frozen target references, delayed feedback, warm-up
  rules and gamma-zero parity are covered by tests. All-history novelty is deferred.
- Complete: the observer uses fixed 96x96 images and sixteen 24x24 masks, with
  matching head, loss, reconstruction, replay, snapshots and provenance. The
  investigation and measured costs are in `docs/observer-resolution.md`.
- Complete: documentation and notebook source/output sweep removed stale 32x32
  observer guidance. Notebook 05 uses 96x96 examples and clears affected stale
  outputs; notebook 07 explicitly retains its mean-target experiment. Historical
  32x32 comparisons remain labelled as such in the investigation report.
- User-directed scope changes: no optional 32x32 baseline or resolution setting;
  the one-off benchmark script was deleted after investigation, and its links and
  commands removed. Findings remain in the report.
- Validation: full suite **507 passed, 6 skipped** (MPS unavailable). After the
  final documentation cleanup and mock reconstruction-size update, the relevant
  documentation/predictability tests passed again: **55 passed**.
- Validation: four real CPU CLI smoke runs covered both offspring strategies,
  max/previous-parent and mean/previous-grid-mean. Each completed four decisions
  and two eligible targets with forecasts active. Saved aggregate targets,
  reference metadata, 96x96 images and observer mask provenance were checked.
  Tests additionally cover the full aggregation/reference combination matrix.
- Independent integration review found no actionable correctness issues.
  Documentation links/options/examples, notebook syntax and `git diff --check`
  passed. Pre-existing notebook 01 edits and deletion of `main.py` were preserved.
