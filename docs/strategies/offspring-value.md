# Offspring value with patch predictability

[All strategies](../../README.md#automated-selection-experiments)

Extend [novelty-predictability](novelty-predictability.md) with a scratch predictor
of the selected parent's next eight children. By default its training target is
the **maximum child value**; `--offspring-aggregation mean` uses the average.
After warm-up, select by `current_value + gamma * predicted_offspring_value`.
The separate comprehension observer is also scratch-initialized and uses a fixed
96x96 input with sixteen 24x24 masks.

```sh
uv run --extra imagenet python experiments/run_selection.py offspring-value --offspring-aggregation max --steps 23 --seed 7
uv run --extra imagenet python experiments/run_selection.py offspring-value --offspring-aggregation mean --novelty-reference previous-parent --steps 23 --seed 7
uv run --extra imagenet python experiments/run_selection.py offspring-value --gamma 0 --steps 23 --seed 7
uv run python experiments/run_selection.py offspring-value --help
```

Defaults give ten comprehension warm-up decisions followed by ten eligible
completed targets. Decision 11 selects the first eligible parent; decision 12
receives its target; **decision 21** can first use forecasts. The examples reach
23 decisions, yielding twelve targets and three forecast-enabled choices (except
the gamma-zero control). A fourteen-decision run with default warm-ups does not
exercise forecast-driven selection. With `--comprehension-warmup-steps 0`, the
first forecast-enabled choice moves to decision 12 at default predictor warm-up.

[All run options](../run-options.md) apply. Applicable strategy options:

- [Novelty reference](options.md#novelty-reference): `--novelty-reference`.
- [Combined value](options.md#combined-value): `--comprehension-weight`.
- [Observer](options.md#comprehension-observer): comprehension warm-up, initialization
  (`random` only), update count, batch size and learning rate.
- [Offspring predictor](options.md#offspring-predictor): aggregation (`max` default
  or `mean`), gamma, completed-target warm-up, updates, batch size, learning rate.
- [Runtime](options.md#runtime): CPU/MPS device and weight cache. Scratch models
  need Torch but download no weights.

Python: `OffspringValueSelectionStrategy(comprehension_weight=0.5,
novelty_reference="previous-grid-mean", offspring_aggregation="max",
comprehension_warmup_steps=10, gamma=1.0, warmup_targets=10,
predictor_training_steps=10, predictor_batch_size=16,
predictor_learning_rate=0.001, observer_initialization="random",
training_steps=20, batch_size=16, learning_rate=0.001, cache_dir=None, device="cpu")`.
`batch_size` configures the observer, not the predictor. ImageNet classifier and
epsilon options do not apply. Use `--device mps` explicitly on Apple Silicon.

Score all eight actual children under the **frozen selection-time** observer,
active novelty reference and original nine novelty/comprehension measurements,
then take their max or mean. Exclude the retained parent; keep duplicate children.
A previous-parent reference is the parent selected on the preceding choice, not
the parent whose children are now being scored. Each frozen target preserves that
actual reference until feedback arrives. See [target semantics](options.md#offspring-predictor).

Only parents selected after comprehension warm-up and with a reference qualify.
Earlier parents stay ineligible even if children arrive after warm-up ends.
For S decisions and comprehension warm-up W, target count is
`max(S-max(W,1)-1,0)`. The final parent is unobserved; rejected parents have no labels.
Every eligible transition enters replay, including repeated parents. Errors use
the original forecast, not a new prediction after training on its outcome.

Gamma zero still trains and logs the predictor but exactly preserves matching
scratch novelty-predictability choices and selection RNG use. Predictor seeds do
not consume additional selection-RNG draws. Use fresh strategies, chronological
nine-image grids and retained parents in position zero; arbitrary playback and
backtracking are unsupported. Saved genome parent links establish ancestry;
equal pixels alone cannot. Records preserve targets, references, original forecasts,
errors, seeds, hashes, sampled replay and training timing.

Max predicts the expected best immediate child in a brood, not diversity or
long-term potential. Mean instead rewards consistently valuable children. A
pixel-only predictor cannot distinguish identical-looking CPPNs with different
mutation behaviour. Errors on selected parents cannot establish that rejected
alternatives were correctly ranked. Patch predictability also retains its
simplicity/familiarity bias. The 96x96 observer avoids downsampling default CPPN
renders, but increases compute/memory and uses larger hidden tiles; see the
[resolution investigation and measured costs](../observer-resolution.md).
Notebook 07 uses the same 96x96 observer and explicitly retains mean aggregation
for its original target experiment. It inspects targets, ancestry and replay;
its shortened warm-up settings also differ from CLI defaults.
