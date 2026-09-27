# Interchangeable selection experiments

## Brief and agreed scope

The conversation is the source brief; the user chose it instead of a THOUGHTS.md.
Make Python/notebook experiments easy to configure, with a thin CLI using the
same components. Start with uniform random, greedy and epsilon-greedy selection.
Replace the existing experiment API and `experiments/imagenet_selection.py`;
update repository callers instead of adding compatibility wrappers. Existing
saved sessions must remain readable in the interpretation notebooks, and no
existing runs or artifacts may be deleted or rewritten.

Preserve the nine-candidate, single-parent breeding loop, rendering, mutation,
parent retention, complete image/genome records and human/automated replay.
Keep ImageNet preprocessing, explicit weights, all 1,000 class measurements,
batching, frozen inference and provenance. Do not add novelty history, VLMs,
archives, crossover, dynamic mutation or a plugin/configuration framework.

The checkout already contains unrelated edits and new interpretation/reference
tools. Preserve them, including notebook seeds and exploratory settings. Merge
documentation changes into the existing README and notebooks without replacing
their other work.

## Design

Separate `images -> measurements -> preference scores -> selection`.

- Retain `ImageEvaluator.evaluate(images) -> Evaluation`. Measurements remain a
  generic matrix, with ordered columns, names and metadata.
- Add a small scoring interface with a description and one finite score per
  candidate. Implement maximum measurement, reproducing today's maximum class
  confidence. Larger preference scores are better. Preserve raw measurements;
  reducing them for these selectors does not change the evaluator contract.
- Selectors receive candidate count, optional preference scores and a run-owned
  random generator. They return a decision with position and selection mode.
  Keep selectors free of persistent RNG state so reuse across notebook runs
  does not silently continue a previous random sequence.
- Greedy chooses the first maximum, retaining current tie behaviour. Uniform
  random samples all nine positions, including the parent. Epsilon-greedy uses
  that random rule with probability epsilon and greedy otherwise. Record the
  branch actually taken, even when exploration happens to pick the best score.
  Validate epsilon in [0, 1]; handle endpoints explicitly.
- The runner owns separate breeding and selection randomness. Derive a stable
  selection seed from the run seed by default, allow an explicit override, and
  record the resolved value. Selection draws must not affect mutation draws.
- Allow evaluation and scoring to be absent for random runs. Also allow random
  selection with evaluation for observation; measurements must not influence
  its choices. Require a scoring component and evaluator for scored policies.
  Evaluate all candidates on each scored decision, including exploratory turns.
- Save raw evaluation, derived scores, scoring identity and decision mode using
  the shared session writer. Add optional decision information to selection
  events, null for human choices. Preserve the version-2 fields used by current
  saved-network readers; additive metadata requires no migration or fallback.
- Separate candidate presentations, evaluated images and unique genomes in
  summaries. For S decisions these are 9*S presentations, either 0 or 9*S image
  evaluations for the initial supported configurations, and 9+8*(S-1) genomes.
- Make contact sheets/progress use preference scores and decision mode. Keep
  ImageNet class labels as measurement information, not as a universal account
  of why a candidate was selected. Unscored runs must render useful grids.
- Reject invalid component combinations before creating a run directory or
  loading/downloading a model. Use fresh output directories as today.

Proposed Python interface (names may be refined without changing responsibilities):

```python
run_experiment(
    output_dir="runs/example",
    evaluator=ImageNetEvaluator(...),
    scoring=MaximumMeasurement(),
    selector=EpsilonGreedy(epsilon=0.1),
    settings=ExperimentSettings(seed=7, ...),
)

run_experiment(
    output_dir="runs/random-example",
    selector=UniformRandom(),
    settings=ExperimentSettings(seed=7, ...),
)
```

## Phase 1: Scoring and selection components

Touch `evaluation.py` only where contracts need clarification; add `scoring.py`
and the new policies/decision type in `selection.py`. The old runner remains
working during this phase; remove its superseded selector in phase 2.

Red: add meaningful unit tests for score shape/finiteness, first-maximum ties,
uniform selection, controlled epsilon branches, epsilon endpoints, invalid
configuration and input immutability. Use controlled random draws for branch
tests rather than flaky statistical thresholds.

Green: implement the small components and their descriptions. Keep inference
out of scoring and selection. No inheritance hierarchy or strategy registry.

Exit: new component tests and the existing evaluator/ImageNet/runner tests pass.
Pause for user code review before phase 2.

## Phase 2: Generic runner, records and CLI

Replace the experiment API and wire the new components into the shared loop.
Extract contact-sheet/report formatting from loop control if doing so clarifies
the code. Update `BreedingSession.select` and `SessionWriter` only as needed for
shared decision records; preserve failure snapshots and full candidate history.
Update existing tests/callers and remove `MaximumClassConfidence`.

Replace `experiments/imagenet_selection.py` with `experiments/run_selection.py`.
Use a thin argparse layer: `--selector random|greedy|epsilon-greedy`, optional
`--epsilon`, `--evaluator none|imagenet`, and the existing experiment settings
plus optional `--selection-seed`. Default evaluator to none for random and
ImageNet for scored selectors. Support explicit ImageNet observation for random.
Reject epsilon on unrelated selectors and none for scored selection. Import
and construct ImageNet only when requested. Align CLI and Python defaults.

Red: extend integration tests to establish greedy trajectory equivalence,
reproducible random/epsilon-greedy runs, independent mutation RNG, complete
records, truthful counters, unscored failure snapshots and saved-image replay.
Check that the same recorded choices replay through the human session, with
only evaluation/decision metadata differing. Test CLI dispatch and invalid
combinations without downloads, including a random run with Torch unavailable.
Check existing saved-session loading and newly saved-session loading through
the interpretation code; existing files must not be mutated.

Green: implement and run the relevant tests. Exercise a short random CLI run in
a temporary directory and inspect its saved grid. Use controlled classifier
fixtures for scored integration checks; no weight download is required.

Exit: all three strategies work through Python and CLI, saved sessions remain
readable, and shared-loop/replay checks pass. Pause for user code review.

## Phase 3: Notebook ergonomics, documentation and final verification

Update notebook 02 to evaluate a fixed grid, compare the three selectors on
those same candidates, then show how to configure a small run. Retain classifier
crop inspection and the simple alternative-evaluator example. Remove obsolete
MAP-Elites promises from notebooks 01/02 without changing exploratory settings.

Update README, AGENTS.md, experiment brief and executable help to describe the
new API/CLI, optional ImageNet dependency, epsilon semantics, tie handling,
separate randomness, run records and counters. Describe random/epsilon-greedy
as available experiments, not empirical results or a frozen scientific protocol.

Document and verify these canonical commands:

```sh
uv run python experiments/run_selection.py --selector random --steps 100 --seed 7
uv run --extra imagenet python experiments/run_selection.py --selector greedy --steps 100 --seed 7
uv run --extra imagenet python experiments/run_selection.py --selector epsilon-greedy --epsilon 0.1 --steps 100 --seed 7
```

Red/green: run the full available test suite after integration and documentation
examples are complete. Validate changed notebook code with small offline inputs
or controlled evaluators; do not overwrite the user's outputs/settings by
executing expensive or unrelated cells. Check CLI help and documentation for
stale command/API names. Manually inspect representative random and scored
contact sheets because text labels and selected borders are user-facing output.

Exit: documentation matches the shipped interface; examples, replay, readers and
tests pass. Compare delivered work against this plan, the conversation brief
(in place of THOUGHTS.md), the accepted TODO.md and later decisions; report any
unintended drift. Pause for final user code review. Ask whether a short
WRAP_UP.md is wanted; generate it only if requested.

## Planning gate

Discovery is complete. The pre-change evaluator/ImageNet/experiment test subset
passed: 27 tests. Review this plan before drafting TODO.md and beginning the
phased implementation, as required by feature-thoughts-planner.
