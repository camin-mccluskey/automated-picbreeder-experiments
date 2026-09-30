# ImageNet selection

[All strategies](../../README.md#automated-selection-experiments)

Score each candidate by its maximum probability across the 1,000 ImageNet classes.
Choose the first maximum, except with probability epsilon choose uniformly from
all nine positions. The winning class can change. Greedy ties retain the parent
after initialization. With deterministic scores, greedy selected confidence cannot
decrease; exploration can select a lower-confidence image.

```sh
uv run --extra imagenet python experiments/run_selection.py imagenet --steps 100 --seed 7
uv run --extra imagenet python experiments/run_selection.py imagenet --epsilon 0.1 --steps 100 --seed 7
uv run python experiments/run_selection.py imagenet --help
```

[All run options](../run-options.md) apply. Strategy options:

- `--epsilon`: default `0`, finite in [0, 1]. Zero is greedy; one always chooses
  randomly while still evaluating. Random draws include the parent and can select
  the greedy winner; records retain the operation actually used.
- [Frozen classifier options](options.md#frozen-imagenet-classifier): model,
  explicit weights and inference batch size.
- [Runtime options](options.md#runtime): device and weight cache.

Python: `ImageNetSelectionStrategy(epsilon=0.0, evaluator=None)`; pass an
`ImageNetEvaluator` to configure the classifier. Default: frozen ResNet18 /
IMAGENET1K_V1 on CPU. All nine images are classified at every decision, including
exploration. Pure [random selection](random.md) avoids that cost.

Confidence is an unvalidated preference proxy and the default preprocessing crops
image edges. No classifier learning or warm-up occurs. Full class measurements,
maximum-confidence scores and model provenance are saved per choice; repeated
parents are evaluated again. Notebook 02 displays the actual crop and labels.
