# Shared strategy options

[Strategy guides](../../README.md#automated-selection-experiments) · [Run options](../run-options.md)

Only the groups linked by a strategy guide apply to that strategy. Defaults below
are effective defaults: the CLI forwards unspecified values to the constructor.

## Novelty reference

All five novelty-based strategies accept `--novelty-reference` (Python:
`novelty_reference`):

| Value | Definition |
| --- | --- |
| `previous-grid-mean` (default) | Full-resolution RGB mean of all nine images on the preceding grid; duplicates count equally |
| `previous-parent` | Full-resolution image actually selected on the preceding decision; the retained parent has zero raw distance |

Raw `pixel_novelty` is mean squared distance over pixels and RGB channels in
[0, 1]. The initial grid has no reference. Novelty ranks are percentile ranks
within the current grid; ties receive their average rank, with neutral rank 0.5
when all candidates tie. Greedy score ties choose the first candidate, normally
the retained parent. Previously-seen flags do not penalize selection.

The common derived metric `display_novelty` always compares with the previous
displayed grid's mean, even when selection uses `previous-parent`.
It matches strategy `pixel_novelty` only in the default mode on normal automated
runs. See [metric definitions](../batch-experiments.md#shared-measurements).

## Combined value

`--comprehension-weight` (Python: `comprehension_weight`) defaults to `0.5`, finite
in [0, 1]. It applies to novelty-imagenet, novelty-predictability and both offspring
strategies. Current value is `(1-weight)*novelty_rank + weight*quality_rank`.
Quality is maximum class confidence or masked-image accuracy, depending on the
strategy. Patch strategies temporarily set this weight to zero during warm-up.
Ranks are relative to a grid, not absolute evidence of progress. Opposing rank
orders can cancel completely at weight 0.5 and leave the parent selected by a tie.

## Frozen ImageNet classifier

Applies to imagenet, novelty-imagenet and offspring-value-imagenet.

| CLI option | Default | Meaning and constraint |
| --- | --- | --- |
| `--imagenet-model` | `resnet18` | Torchvision ImageNet-1K model |
| `--imagenet-weights` | `IMAGENET1K_V1` | Explicit checkpoint supported by the model; `DEFAULT` is rejected |
| `--imagenet-batch-size` | `16` | Inference batch size; integer >= 1 |

Python: pass `evaluator=ImageNetEvaluator(model_name="resnet18",
weights="IMAGENET1K_V1", batch_size=16, device="cpu", cache_dir=...)` to the strategy.
Without an evaluator, the strategy constructs the default classifier. One loaded
frozen evaluator may be shared across otherwise fresh strategies.

The default transform resizes to 256, centre-crops to 224 and applies ImageNet
normalization; image edges are cropped. This differs from the patch observer's
fixed 96x96 input. All 1,000 softmax scores and preprocessing/checkpoint provenance are
saved. Confidence is a relative class score, not validated recognisability,
naturalness, creativity or comprehension. No classifier parameters train.

## Comprehension observer

Applies only to novelty-predictability and offspring-value.

| CLI option | Default | Meaning and constraint |
| --- | --- | --- |
| `--comprehension-warmup-steps` | `10` | Decisions before comprehension affects selection/target eligibility; integer >= 0; 0 disables warm-up |
| `--observer-initialization` | `random` | `random` or `imagenet` for novelty-predictability; only `random` for offspring-value |
| `--training-steps` | `20` | Observer Adam updates after each decision; integer >= 1 |
| `--observer-batch-size` | `16` | Observer training batch size; integer >= 2 for BatchNorm |
| `--learning-rate` | `0.001` | Observer Adam learning rate; finite and > 0 |

Python uses the hyphen-to-underscore names, except `--observer-batch-size` maps
to `batch_size`. Both initializations train the full ResNet18 backbone plus a new
reconstruction head with matched initialization. A fourth input channel encodes
visibility and starts with zero weights. Scoring uses evaluation mode, fixed
BatchNorm statistics and the model before the current grid's training.
Replay contains all distinct displayed images, including rejects, in first-seen
order. New images enter replay after selection. Initialization and training-example
sampling use CPU even when model computation uses MPS.

The observer uses a fixed 96x96 input and reconstruction output, retaining the
full image at the default rendering size. Sixteen separate 24x24 tiles are hidden
in turn for scoring: a 4x4 grid, with 1/16 of the image hidden each time. `--size`
controls CPPN rendering independently: other rendered sizes are resized to 96x96.
There is no observer-resolution option.

Comprehension is `1 - masked_mse`. The first grid has no comprehension estimate.
This measures prediction accuracy, not learning progress. Flat or familiar images
may be easy to reconstruct. The observer avoids downsampling default renders,
but does not establish useful understanding or UFR. See the [resolution
investigation and measured costs](../observer-resolution.md) for the compute,
memory and prediction-task consequences of this design. Observer input size,
mask geometry and model hashes are recorded.

## Offspring predictor

Applies only to offspring-value and offspring-value-imagenet.

| CLI option | Default | Meaning and constraint |
| --- | --- | --- |
| `--offspring-aggregation` | `max` | `max` or `mean` of eight realised child values supplies each training target |
| `--gamma` | `1` | Forecast contribution weight; finite and >= 0; 0 trains/logs but preserves the current-value control |
| `--warmup-targets` | `10` | Completed eligible targets before forecasts affect selection; integer >= 1 |
| `--predictor-training-steps` | `10` | Predictor Adam updates per completed target; integer >= 1 |
| `--predictor-batch-size` | `16` | Transition training batch size; integer >= 1 |
| `--predictor-learning-rate` | `0.001` | Predictor Adam learning rate; finite and > 0 |

Python uses the same names with underscores. After warm-up, choose the first
maximum of `current_value + gamma * predicted_offspring_value`; forecasts are
not reranked. Each fresh predictor learns the configured aggregation within its
run. `max` forecasts the expected best value in an eight-child brood. It does not
mean selecting the largest forecast alone or forecasting an individual child.
`mean` forecasts average child value.
Target metadata stores the chosen value as `offspring_value`; separate
`mean_offspring_value` and `max_offspring_value` fields retain their literal meanings.

Freeze the **selection-time active novelty reference image**, observer (patch
version), and original nine novelty/quality measurements until the selected
parent's children arrive. Score each child against those fixed references, then
aggregate their values. Exclude the retained parent; count all eight children,
including duplicates. The reference may be a previous-grid mean or a previous
parent; it is not replaced by the newly selected parent's image during feedback.

Only actual selected-parent transitions supply labels. Preserve every eligible
transition, original pre-outcome forecast and delayed error, including repeated
parents. The final selected parent has no observed children. There is no probing
of rejected parents, extra offspring generation or added classifier pass.

The scratch scalar predictor takes full-resolution parent and reference images,
with 23 context scalars for patch value or 21 for ImageNet value. It cannot
separate pixel-identical CPPNs with different mutation behaviour. Max targets
reward the best immediate continuation, not diversity or long-term potential;
mean targets reward consistently valuable offspring. Neither establishes UFR.

## Runtime

All local model strategies accept these options; random, novelty and VLM do not.

| CLI option | Default | Meaning and constraint |
| --- | --- | --- |
| `--device` | `cpu` | `cpu` or `mps` for observer/offspring strategies; frozen classifier-only strategies also accept devices such as `cuda` |
| `--cache-dir` | repository `.cache/imagenet` | Storage for downloaded weights; scratch models download nothing |

An unavailable requested device fails without automatic fallback. In Python,
observer strategies accept `device` and `cache_dir`; frozen evaluators own those
settings. For offspring-value-imagenet, strategy `device` controls the predictor
and the supplied evaluator controls its classifier; the CLI sets both together.
Python `cache_dir=None` uses Torch Hub's checkpoints directory; the CLI explicitly
chooses the repository cache.

Saved seeds, hashes, replay samples, timing and training records support fresh
chronological replay in a fixed environment; these are not resumable model
checkpoints. Local model commands need `uv run --extra imagenet`, including scratch
strategies that use Torch but do not download ImageNet weights.

## OpenRouter VLM

Only `vlm` accepts these options. Requires `uv run --extra vlm` and
`OPENROUTER_API_KEY` in the repository `.env` or environment.

| CLI option | Default | Python mapping and constraint |
| --- | --- | --- |
| `--vlm-model` | unset; required to run | `model`: nonempty explicit OpenRouter model ID supporting nine images and structured output |
| `--vlm-prompt` | `choose the most interesting image to you` | `prompt`: nonempty text, saved exactly |
| `--temperature` | `0.0` | `temperature`: finite in [0, 2]; no deterministic replay guarantee |
| `--max-completion-tokens` | `1024` | `max_completion_tokens`: integer >= 1; total output budget, including reasoning where applicable |
| `--timeout` | `120.0` | `timeout`: finite positive seconds per SDK attempt |
| `--max-retries` | `5` | `max_retries`: integer >= 0; extra attempts for transient errors only; pacing, jitter and server-directed waits are automatic |
| `--env-file` | repository `.env` | `env_file=None`: default path; explicit path accepted; environment variable takes precedence |

Python-only `client=None` allows injecting an SDK client; the caller owns its
lifetime and credentials. The strategy sends no history or selection RNG seed.
No local model trains. Read [VLM selection](vlm.md) for request formatting,
validation, failure records, retry accounting and scientific limitations.
