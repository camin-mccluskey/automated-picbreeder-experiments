# Novelty plus ImageNet confidence

[All strategies](../../README.md#automated-selection-experiments)

Combine novelty and maximum class-confidence percentile ranks:
`current_value = (1-weight)*novelty_rank + weight*confidence_rank`.
Choose the first maximum. Default weight is 0.5. The first grid has neutral novelty:
positive weights use confidence immediately; weight zero chooses uniformly.

```sh
uv run --extra imagenet python experiments/run_selection.py novelty-imagenet --comprehension-weight 0.5 --steps 100 --seed 7
uv run --extra imagenet python experiments/run_selection.py novelty-imagenet --novelty-reference previous-parent --comprehension-weight 0.25 --steps 100 --seed 7
uv run python experiments/run_selection.py novelty-imagenet --help
```

[All run options](../run-options.md) apply. Applicable strategy options:

- [Novelty reference](options.md#novelty-reference): `--novelty-reference`, default
  `previous-grid-mean`, alternative `previous-parent`.
- [Combined value](options.md#combined-value): `--comprehension-weight`, default 0.5,
  finite in [0, 1].
- [Frozen classifier](options.md#frozen-imagenet-classifier) and
  [runtime](options.md#runtime): model, weights, inference batch size, device, cache.

Python: `NoveltyImageNetSelectionStrategy(comprehension_weight=0.5,
novelty_reference="previous-grid-mean", evaluator=None)`.
A supplied `ImageNetEvaluator` configures the classifier. Use a fresh strategy.

Weight zero preserves matching novelty choices and RNG use; weight one matches
greedy ImageNet choices. Every setting classifies all nine candidates per decision,
including weight zero. There is no learning, epsilon exploration or warm-up.
Records retain raw novelty and confidence, both ranks, full class vectors and
checkpoint/preprocessing provenance in `evaluation.metadata.classifier_evaluation`.

Confidence is not validated comprehension. Novelty is local and uses the whole
rendered image; the classifier sees its checkpoint's transform, including the
default crop. Opposing ranks can cancel at equal weight and cause parent retention.
