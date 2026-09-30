# VLM selection

[All strategies](../../README.md#automated-selection-experiments)

Send all nine candidates in one request to an OpenRouter vision-language model
using the official `openrouter` Python SDK. The exact default selection
instruction is:

> choose the most interesting image to you

The strategy does not define interestingness or add recognisability, realism,
novelty or complexity criteria. Separate protocol text explains the numbered
images and asks for one index and a brief reason. Each call is independent:
there is no previous conversation, archive, training or warm-up.

## Setup and use

Install the optional extra; Torch and ImageNet weights are unnecessary:

```sh
uv sync --locked --extra vlm
```

Fill in `OPENROUTER_API_KEY` in the repository-root `.env`. That file is ignored
by Git. `.env.example` is a versioned template containing `OPENROUTER_API_KEY=`
with no value. On a fresh checkout, copy `.env.example` to `.env` first. An
existing environment variable takes precedence, including an explicitly empty
one. An empty or missing key fails before a standalone run creates output.
The key is not included in strategy descriptions, requests' saved metadata or
error messages. Python and the CLI both default to the repository `.env`, even
when launched from `notebooks/`; `--env-file` selects another file.

Choose a model explicitly. There is no default model or automatic model
fallback. The model/provider must accept nine input images and JSON Schema
structured output. The example uses an OpenRouter model ID; check current model
availability before starting a long run.

```sh
uv run --extra vlm python experiments/run_selection.py vlm --vlm-model openai/gpt-4.1-mini --steps 3 --seed 7
uv run --extra vlm python experiments/run_selection.py vlm --help
```

Three decisions are a smoke test; the VLM chooses from the first decision.
Use `--steps 100` for a longer run or add `--runs N` for independent runs.
Every request uses the paid API account associated with the key. The example
[GPT-4.1 Mini model](https://openrouter.ai/openai/gpt-4.1-mini) supports image
inputs and JSON Schema output; this is an example, not a required model.

[All run options](../run-options.md) and [OpenRouter options](options.md#openrouter-vlm)
apply. No local device, cache, classifier, observer or predictor options apply.

Constructor defaults (supply `model` to construct a usable strategy):

```python
VLMSelectionStrategy(
    model=None,
    prompt="choose the most interesting image to you",
    temperature=0.0,
    max_completion_tokens=1024,
    timeout=120.0,
    max_retries=2,
    env_file=None,
    client=None,
)
```

```python
from automated_picbreeder.selection_strategies import VLMSelectionStrategy

strategy = VLMSelectionStrategy(model="openai/gpt-4.1-mini")
```

See [Python experiment usage](../usage.md#configure-experiments-in-python-or-notebooks).
`client` is an optional injected OpenRouter SDK client for testing or custom
transport configuration. Its caller owns its lifetime and authentication; when
it is supplied, the strategy does not read credentials. Otherwise it creates
and closes an SDK client for each selection. The strategy never consumes the
supplied selection RNG and does not send an API seed.

## Request and choice

The strategy accepts exactly nine RGB uint8 arrays. It encodes each as a separate
full-resolution PNG without local cropping, resizing, labelling the pixels or
combining them into a collage. Text labels `Image 0` through `Image 8` precede
the corresponding attachments. The order matches the displayed candidates;
after initialization the retained parent is at index 0. Duplicates are sent
individually. The provider may apply its own image preprocessing.

One non-streaming `client.chat.send` request carries all nine images. It uses
`response_format.type=json_schema`, `strict=true`, `additionalProperties=false`
and `provider.require_parameters=true`. The response must contain exactly:

```json
{"selected_index": 4, "reason": "A brief explanation of the choice."}
```

The index must be an integer in 0–8; booleans, strings and fractional values are
rejected. The reason must be nonempty. Refusals, truncated responses, malformed
JSON and invalid choices fail without selecting anything or substituting a
random choice. A response is accepted only with `finish_reason=stop`.

Transient transport errors and HTTP 408, 429, 500, 502, 503 and 504 retry the same
request up to `max_retries` times, with delays of 1, 2, 4 and then at most 8
seconds. SDK automatic retries are disabled so every attempt is counted.
Authentication, payment, unsupported-parameter and invalid-output errors do not
retry. OpenRouter may route between providers of the requested model; returned
routing metadata is saved when available.

## Records and interpretation

Decisions record mode `vlm`, with null numeric evaluation and preference scores.
`decision.metadata.vlm` contains the exact prompt/protocol/schema, model and
generation settings, SDK version, ordered PNG hashes, selected index, reason,
and SDK response for each attempt, including usage, finish reason and response
ID. Candidate IDs and saved PNGs in the selection event map indices back to
genomes. Credentials and base64 request bodies are not saved.

`decision.metadata.inference` records API request attempts, images submitted,
prompt/completion tokens, reported USD cost and elapsed time including retries.
These appear in metrics JSON/CSV, the run summary and batch metric aggregates.
Unknown usage or cost remains null, including when a retry has unreported usage;
known responses still remain individually inspectable. `evaluated_images` is
zero because no numeric evaluation rows were produced; it is not an API cost
counter. Without retries, S decisions make S API calls and submit 9*S images.

A failed request saves `selection_failure.json` with the generation, candidate
IDs and diagnostic metadata. The current grid was already saved before the
request. Completed-decision metrics exclude that failed decision and its costs;
consult its failure record too. An interrupted request may have unknown cost.

Interestingness is the chosen model's response to this prompt, not a validated
measurement or evidence of UFR. Its reason is a reported explanation, not proof
of its internal decision process. Fixed ordering can introduce position bias,
including a preference for the retained parent. Temperature zero and matching
seeds do not guarantee identical remote choices: models, routing and inference
can change. Saved choices still support inspecting and reproducing the breeding
trajectory in a fixed local environment without calling the API again.

The separate-image presentation follows the current
[Picbreeder-VLM selection implementation](https://github.com/smearle/picbreeder-vlm/blob/76c2563787c1789f2b810be2f49970ef819d20af/picbreeder_vlm/vlm/chat.py#L278-L420).
This strategy implements only a single parent choice, not its multi-agent archive
system. See OpenRouter's [SDK](https://openrouter.ai/docs/client-sdks/python/overview),
[image inputs](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
and [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
documentation for provider requirements.
