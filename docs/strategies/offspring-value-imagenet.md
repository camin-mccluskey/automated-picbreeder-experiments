# Offspring value with ImageNet confidence

[All strategies](../../README.md#automated-selection-experiments)

Extend [novelty-imagenet](novelty-imagenet.md) with a scratch predictor of next-brood
value. Current value combines novelty and maximum class-confidence ranks. After
predictor warm-up, choose by `current_value + gamma * predicted_offspring_value`.
By default the target is the **maximum** value among the next eight children;
`--offspring-aggregation mean` predicts their average instead.
Only the offspring predictor trains; the classifier remains frozen.

```sh
uv run --extra imagenet python experiments/run_selection.py offspring-value-imagenet --offspring-aggregation max --steps 14 --seed 7
uv run --extra imagenet python experiments/run_selection.py offspring-value-imagenet --offspring-aggregation mean --novelty-reference previous-parent --steps 14 --seed 7
uv run --extra imagenet python experiments/run_selection.py offspring-value-imagenet --gamma 0 --steps 14 --seed 7
uv run python experiments/run_selection.py offspring-value-imagenet --help
```

The first choice already uses confidence at positive comprehension weights, but
has no novelty reference and cannot supply a target. Decision 2 selects the first
eligible parent; decision 3 receives its target. With ten warm-up targets,
**decision 12** can first use forecasts. Fourteen decisions give twelve targets
and three forecast-enabled choices (except the gamma-zero control). There is no
patch observer or comprehension warm-up. For S decisions, there are
`max(S-2,0)` targets; the final selected parent remains unobserved.

[All run options](../run-options.md) apply. Applicable strategy options:

- [Novelty reference](options.md#novelty-reference): `--novelty-reference`.
- [Combined value](options.md#combined-value): `--comprehension-weight`.
- [Frozen classifier](options.md#frozen-imagenet-classifier): model, explicit
  checkpoint and inference batch size.
- [Offspring predictor](options.md#offspring-predictor): aggregation (`max` default
  or `mean`), gamma, completed-target warm-up, updates, batch size, learning rate.
- [Runtime](options.md#runtime): CPU/MPS device and weight cache. CLI device sets
  both models; CUDA is unavailable here because the predictor supports CPU/MPS.

Python: `OffspringValueImageNetSelectionStrategy(comprehension_weight=0.5,
novelty_reference="previous-grid-mean", offspring_aggregation="max", evaluator=None,
gamma=1.0, warmup_targets=10, predictor_training_steps=10,
predictor_batch_size=16, predictor_learning_rate=0.001, device="cpu")`.
Strategy `device` controls the predictor; a supplied `ImageNetEvaluator` owns the
classifier's model, weights, batching, cache and device. The CLI configures both.
Observer options (including `--comprehension-warmup-steps`) and epsilon are rejected.

Freeze the active selection-time novelty reference and nine novelty/confidence
measurements for each eligible selected parent. Score its eight actual children
under those references when they arrive; exclude the retained parent and keep
duplicates, then aggregate max or mean. Confidence comes from the next grid's
single classifier pass, so all nine images are classified exactly once per decision,
including gamma-zero and comprehension-weight-zero runs. No extra offspring or
classifier inference is added for targets. [Shared target semantics](options.md#offspring-predictor)
explains why the reference stays fixed in either novelty mode.

Gamma zero trains/logs while exactly preserving matching novelty-imagenet choices
and selection RNG use. Predictor seeds derive from the initial supplied RNG state
without drawing from it. The predictor uses full-resolution parent/reference images
and 21 context scalars; the patch version uses 23 including observer statistics.
Every eligible transition remains in replay, including repeated parents. Original
forecasts, delayed errors, child values, references, full classifier output,
provenance, hashes and training records are saved.

Use a fresh strategy and chronological nine-image grids with the retained parent
first. Genome parent links establish ancestry; pixel equality cannot. Max forecasts
the expected best child of the next brood, not long-term potential or diversity.
Only selected parents reveal outcomes. Class confidence remains an unvalidated
preference proxy; neither improved prediction loss nor high confidence proves UFR.
