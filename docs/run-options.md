# Run options and reproducibility

[Strategy guides](../README.md#automated-selection-experiments) · [Usage](usage.md)

All eight strategies accept these options, after the strategy subcommand.
`--help` (or `-h`) lists only options applicable to that command. Abbreviated or
unrelated option names are rejected. Commands run from the repository root.

| CLI option | Default | Meaning and constraint |
| --- | --- | --- |
| `--steps` | `100` | Number of decisions, including initial choice; integer >= 1 |
| `--runs` | unset | Independent batch of N runs; integer >= 1; unset makes a standalone run |
| `--seed` | `7` | Integer breeding seed; batch run i uses seed + i |
| `--selection-seed` | derived | Integer selection seed; explicit batch seed increments by i |
| `--size` | `96` | Rendered image side length; integer >= 2; patch observers always use 96x96, resizing other rendered sizes |
| `--mutation-strength` | `0.2` | Finite nonnegative weight/bias mutation strength |
| `--no-topology` | off | Freeze structure, connection enabled states and activation functions |
| `--checkpoint-every` | `10` | Decisions between audit snapshots; integer >= 1 |
| `--output` | timestamped directory in `runs/` | Must be a new directory; existing outputs are never overwritten |

Python uses `ExperimentSettings(steps=100, seed=7, selection_seed=None, size=96,
mutation_strength=0.2, topology=True, checkpoint_every=10)`. Pass `output_dir` and
one `selection_strategy` to `run_experiment`; see [Python usage](usage.md#configure-experiments-in-python-or-notebooks).
`--runs` uses a fresh strategy/model per run; the [batch guide](batch-experiments.md)
explains `run_batch`, failures, seed scheduling and equal-run aggregation.

```sh
uv run python experiments/run_selection.py novelty --runs 10 --seed 7 --steps 100 --output runs/novelty-batch
```

Breeding and selection have separate RNGs. Each run starts a fresh selection RNG;
stateful strategies must also be constructed afresh. Matching seeds does not keep
candidate grids identical after choices diverge. Match settings, environment,
checkpoint, device and batching for replay; cross-device equality is not promised.
Restart notebook kernels after imported modules change.
VLM selection does not consume the selection RNG or send an API seed. Remote
choices are not guaranteed reproducible, even at temperature zero; exact model,
prompt, settings and returned choices are recorded.

With S decisions, every automated strategy presents `9*S` candidates and generates
`9 + 8*(S-1)` genomes. Frozen classifiers evaluate all nine images on every grid,
including retained parents and exploratory turns. Novelty records `9*S` rows
(including initial unavailable-reference placeholders), but needs no inference.
Pure random selection evaluates no images. Patch observers additionally process
16 masked inputs per candidate and train online; report those costs separately.
`evaluated_images` counts candidate rows, not model calls or training examples.
VLM selection produces no numeric evaluation rows: `evaluated_images` is zero.
Its separate `api_requests` and `api_images_submitted` counters include retries;
without retries, S decisions make S requests containing 9*S images. API usage,
reported USD costs and elapsed time are saved separately; unknown values stay
null. See [VLM selection](strategies/vlm.md) for failure accounting.

A 100-decision run therefore presents 900 candidates and generates 801 genomes.
Saved checkpoints are audit records, not resumable sessions. All generated images,
including rejected alternatives, are preserved; see [saved sessions](usage.md#saved-sessions).
The network loader also reads older version-2 HSB sessions. Metrics and viewers
require current strategy and decision metadata; see [format details](usage.md#saved-sessions).
