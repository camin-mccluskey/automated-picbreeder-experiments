# Random selection

[All strategies](../../README.md#automated-selection-experiments)

Choose uniformly among all nine candidate positions, including the retained
parent and duplicate images. There is no inference, measurement or preference
score. This is the selection control, not a diversity-maximising strategy.

```sh
uv run python experiments/run_selection.py random --steps 100 --seed 7
uv run python experiments/run_selection.py random --help
```

[All run options](../run-options.md) apply. There are no strategy-specific options.
Python: `RandomSelectionStrategy()` takes no options. Each choice uses the
run-supplied selection RNG and records mode `random`; evaluation and scores are
null. See [Python usage](../usage.md#configure-experiments-in-python-or-notebooks).
