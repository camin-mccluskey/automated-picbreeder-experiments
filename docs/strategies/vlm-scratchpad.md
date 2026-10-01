# VLM selection with a scratchpad

[All strategies](../../README.md#automated-selection-experiments)

Ask an OpenRouter vision-language model to select one of nine current images,
carrying forward one rewritable text note. The exact default task prompt is:

> find something interesting

The note starts empty. Every successful response contains the selected index,
a brief reason and a complete replacement note. Only the latest note is sent on
the next decision; no previous messages or historical image attachments are sent.
Image 0 is the previous selection retained unchanged after the initial grid.
The other eight images are its independently mutated children. The initial grid
contains nine random images. The fixed protocol explains this breeding loop.
It explicitly tells the model that whichever index it selects now becomes
Image 0 on the next call. The replacement note must refer to that selection as
the retained parent (Image 0 in the next grid) and describe other candidates by
visual features rather than grid numbers. Candidate numbers are not persistent
identities.

The note is free-form. The model is instructed to keep it concise, record useful
observations and goals, and revise or abandon goals as interesting features emerge.
There is no separate summarisation call, append-only journal, tool loop, training
or warm-up. There is no hard note-length limit; the shared completion-token budget
bounds the entire response. Interestingness is not given additional criteria.

## Setup and use

Use the optional `vlm` extra and an explicit OpenRouter model supporting nine
images and strict JSON Schema output. Follow the existing [VLM setup](vlm.md#setup-and-use)
for the ignored repository `.env` and `OPENROUTER_API_KEY`. Environment values take
precedence; credentials are never saved in records. Torch is unnecessary.

```sh
uv run --extra vlm python experiments/run_selection.py vlm-scratchpad --vlm-model openai/gpt-4.1-mini --steps 3 --seed 7
uv run --extra vlm python experiments/run_selection.py vlm-scratchpad --help
```

Three decisions are a smoke test that exercises note creation and two subsequent
uses. Longer runs use `--steps 100`; `--runs N` constructs a fresh strategy with an
empty note for each run. API requests use the account associated with the key.
The model ID is an example; check availability before a long run.

All [run options](../run-options.md) and [OpenRouter options](options.md#openrouter-vlm)
apply. The `--vlm-prompt` default is `find something interesting`. There are no
scratchpad-specific options or local device, classifier, observer or predictor
options.

Constructor defaults (supply `model` to construct a usable strategy):

```python
VLMScratchpadSelectionStrategy(
    model=None,
    prompt="find something interesting",
    temperature=0.0,
    max_completion_tokens=1024,
    timeout=120.0,
    max_retries=5,
    env_file=None,
    client=None,
)
```

```python
from automated_picbreeder.selection_strategies import VLMScratchpadSelectionStrategy

strategy = VLMScratchpadSelectionStrategy(model="openai/gpt-4.1-mini")
```

Use a fresh instance for each independent run or inspection. Each `choose` call
advances the note on success. `client` is an optional injected SDK client whose
caller owns its lifetime and authentication, as for the original VLM strategy.
See [Python experiment usage](../usage.md#configure-experiments-in-python-or-notebooks).

## Request and choice

The shared adapter sends exactly nine separate full-resolution RGB uint8 PNGs,
labelled `Image 0` through `Image 8`, preserving order and duplicates. It sends
one system message containing the fixed protocol and one user message containing
the task prompt, `Current scratchpad:` followed by the exact note, and the nine
images. The model returns exactly:

```json
{
  "selected_index": 4,
  "reason": "The branching shape is clearer here.",
  "scratchpad": "Explore branching forms while preserving the central split."
}
```

The index must be an integer in 0–8, and both text fields must be nonempty strings.
Additional fields, refusals, truncated responses and malformed output fail without
selecting anything. Only a complete, validated response replaces the note. A
transport or validation failure leaves the prior note intact. Transient retries
send the identical request and note. Pacing, retry limits, credential handling and
failure diagnostics use the existing [VLM transport](vlm.md#request-and-choice).
The strategy consumes no selection RNG draws and sends no API seed.

## Records and interpretation

Decisions use mode `vlm-scratchpad`, with null evaluation and preference scores.
The existing `decision.metadata.vlm` audit stores `scratchpad_before` (the note
sent) and `scratchpad` (the accepted replacement), alongside the choice, reason,
exact protocol/schema, image hashes and response attempts. Each version remains
in the saved session and per-generation metrics even after the live note is
replaced. Configuration records describe the memory rule without including the
mutable note. A failure audit includes `scratchpad_before`; invalid returned text
remains inspectable in the recorded response. Saved sessions are audit records,
not resumable checkpoints.

All generated images are still saved locally for inspection and breeding replay.
Only current candidates are attached to API requests. Without retries, S decisions
make S API calls and submit 9*S images. The note adds input/output tokens but no
extra calls; existing API usage, cost and timing counters include them. Numeric
`evaluated_images` remains zero. See [VLM records](vlm.md#records-and-interpretation)
for missing usage, failures and nondeterministic remote choices.

The model remembers its descriptions of earlier images, not the images themselves.
Omitted details are lost, and mistaken observations or unproductive goals can
persist. A note or stated goal is not proof of the model's internal decision
process, human interestingness or UFR. Comparing defaults with the original VLM
strategy changes the task prompt and breeding instructions as well as memory;
it does not isolate the effect of the note alone.
