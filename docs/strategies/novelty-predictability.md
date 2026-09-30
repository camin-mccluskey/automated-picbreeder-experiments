# Novelty plus masked-image predictability

[All strategies](../../README.md#automated-selection-experiments)

Combine novelty rank with `1 - masked_mse` rank from an online ResNet18 observer.
Measure comprehension before training on the current grid. The observer learns
to reconstruct hidden 24x24 tiles in the full default 96x96 images, using 16
masks per image. Observer resolution is fixed at 96x96; novelty uses
full-resolution pixels.
After each choice it trains on replay of all distinct displayed images, including
rejected candidates. A retained image may already be familiar to the observer.

```sh
uv run --extra imagenet python experiments/run_selection.py novelty-predictability --observer-initialization random --steps 15 --seed 7
uv run --extra imagenet python experiments/run_selection.py novelty-predictability --observer-initialization imagenet --steps 15 --seed 7
uv run --extra imagenet python experiments/run_selection.py novelty-predictability --novelty-reference previous-parent --comprehension-warmup-steps 0 --steps 5 --seed 7
uv run python experiments/run_selection.py novelty-predictability --help
```

With default warm-up, the first choice is uniform and the next nine use novelty
alone, with observer training throughout. Comprehension first influences selection
at **decision 11**. Fifteen decisions exercise that mechanism. Setting warm-up to
zero enables comprehension from decision 2; the first grid remains unavailable.
A three-decision default run would only be a training smoke test.

[All run options](../run-options.md) apply. Applicable strategy options and complete defaults:

- [Novelty reference](options.md#novelty-reference): `--novelty-reference`.
- [Combined value](options.md#combined-value): `--comprehension-weight`.
- [Comprehension observer](options.md#comprehension-observer): warm-up,
  initialization, training updates, batch size and learning rate.
- [Runtime](options.md#runtime): CPU/MPS device and checkpoint cache.

Python: `NoveltyPredictabilitySelectionStrategy(comprehension_weight=0.5,
novelty_reference="previous-grid-mean", comprehension_warmup_steps=10,
observer_initialization="random", training_steps=20, batch_size=16,
learning_rate=0.001, cache_dir=None, device="cpu")`. Both observer initializations
train the entire backbone and use matched new heads; only `imagenet` downloads
pretrained weights. No frozen classification options or epsilon apply.

Use a fresh strategy per run. Seeds, hashes (including BatchNorm buffers), sampled
replay examples, losses, timings, warm-up state and pre-update errors are saved.
Training loss is distinct from comprehension. Exact replay requires a matching
environment and full chronological history, not saved model checkpoints.

Prediction accuracy can reward flat or previously trained-on images. It does not
measure learning progress or UFR. The fixed 96x96 observer avoids downsampling
96x96 CPPN renders, at increased compute/memory cost; its larger hidden tiles also
change the prediction task. See the [resolution investigation and measured
costs](../observer-resolution.md). Notebook 05 inspects observer initialization,
references and predictions using the same 96x96 observer.
