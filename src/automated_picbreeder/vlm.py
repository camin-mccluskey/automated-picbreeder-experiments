"""OpenRouter transport and strict parsing for direct VLM selection.

Optional SDK imports live inside the adapter. No breeding logic or numeric
image scores belong here, and credentials never enter configuration or records.
"""

import base64
from contextlib import nullcontext
from copy import deepcopy
from datetime import timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from importlib.metadata import version
import json
import math
from numbers import Real
import os
from pathlib import Path
from random import SystemRandom
import time

from .cppn import png_bytes


PROTOCOL = (
    "You are selecting one of nine candidate images. Each image is preceded by "
    "its label, Image 0 through Image 8. Return a JSON object with selected_index "
    "(an integer from 0 to 8) and reason (a brief explanation of your choice)."
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_selection", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "selected_index": {"type": "integer", "enum": list(range(9))},
                "reason": {"type": "string"},
            },
            "required": ["selected_index", "reason"], "additionalProperties": False,
        },
    },
}

SCRATCHPAD_PROTOCOL = (
    "You are selecting one of nine candidate images. Each image is preceded by "
    "its label, Image 0 through Image 8. Your selected image will be retained "
    "unchanged and used as the parent of eight independently mutated children. "
    "The first grid contains nine random images; on subsequent grids, Image 0 "
    "is your previous selection and Images 1 through 8 are its children. "
    "Whatever index you select now, that selected image will be Image 0 in "
    "the next generation shown to the next VLM call. For example, if you select "
    "Image 4 now, the next call will see that same image as Image 0, alongside "
    "eight new children labelled Image 1 through Image 8. Image numbers are "
    "positions in the current grid, not persistent identities. "
    "Update your scratchpad with observations and goals useful for future "
    "selections. You may revise or abandon goals as interesting features emerge. "
    "Write the replacement note for the next call: refer to your selected image "
    "as 'the retained parent (Image 0 in the next grid)' and describe its visual "
    "features. Describe other images by their visual features, not their current "
    "numbers; those candidates will not be carried forward. "
    "This scratchpad is your only memory between decisions and starts empty. "
    "Keep it concise and free-form; return the complete replacement note. "
    "Return a JSON object with selected_index (an integer from 0 to 8), reason "
    "(a brief explanation of your choice), and scratchpad (a nonempty string)."
)
SCRATCHPAD_RESPONSE_FORMAT = deepcopy(RESPONSE_FORMAT)
SCRATCHPAD_RESPONSE_FORMAT["json_schema"]["name"] = "image_selection_with_scratchpad"
SCRATCHPAD_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]["scratchpad"] = {"type": "string"}
SCRATCHPAD_RESPONSE_FORMAT["json_schema"]["schema"]["required"].append("scratchpad")

# Transport timing must not advance either the selection or breeding RNG.
_TIMING_RANDOM = SystemRandom()
_BACKOFF_INITIAL = 2.0
_BACKOFF_MAX = 60.0
_MIN_REQUEST_INTERVAL = 3.0
_REQUEST_JITTER = 1.0


def _retry_after_seconds(exc):
    """Read only retry timing from SDK errors; never retain raw headers."""
    headers = getattr(exc, "headers", {})
    for name, scale in (("retry-after-ms", .001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except (TypeError, ValueError, OverflowError):
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = max(0.0, date.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def _parse_choice(response, *, scratchpad=False):
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Expected one complete VLM response (finish_reason=stop).")
    message = choices[0].get("message", {})
    if message.get("refusal"):
        raise ValueError("The VLM refused the selection request.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("The VLM returned no JSON text.")
    try:
        choice = json.loads(content)
    except ValueError:
        raise ValueError("The VLM returned invalid JSON.") from None
    fields = {"selected_index", "reason", "scratchpad"} if scratchpad else {"selected_index", "reason"}
    if not isinstance(choice, dict) or set(choice) != fields:
        raise ValueError(f"Expected exactly {', '.join(sorted(fields))} in the VLM response.")
    position = choice["selected_index"]
    if type(position) is not int or not 0 <= position < 9:
        raise ValueError("VLM selected_index must be an integer from 0 to 8.")
    if not isinstance(choice["reason"], str) or not choice["reason"].strip():
        raise ValueError("The VLM must supply a nonempty reason.")
    if scratchpad and (not isinstance(choice["scratchpad"], str) or not choice["scratchpad"].strip()):
        raise ValueError("The VLM must supply a nonempty scratchpad string.")
    return choice


class OpenRouterSelection:
    """Own request construction, credential loading, retries and response audit."""

    protocol = PROTOCOL
    response_format = RESPONSE_FORMAT
    uses_scratchpad = False

    def __init__(self, *, model, prompt, temperature, max_completion_tokens,
                 timeout, max_retries, env_file, client):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Set an explicit OpenRouter model with model= or --vlm-model.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The VLM prompt must be nonempty.")
        for name, value, minimum, maximum in (
            ("temperature", temperature, 0, 2), ("timeout", timeout, 0, math.inf),
        ):
            if name == "temperature" and value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not minimum <= value <= maximum
                    or (name == "timeout" and value <= 0)):
                raise ValueError(f"Invalid VLM {name}.")
        for name, value, minimum in (("max_completion_tokens", max_completion_tokens, 1),
                                     ("max_retries", max_retries, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.model, self.prompt = model, prompt
        self.temperature = None if temperature is None else float(temperature)
        self.timeout = float(timeout)
        self.max_completion_tokens, self.max_retries = max_completion_tokens, max_retries
        self._next_request_at = 0.0
        self.client = client
        self._api_key = None
        if client is None:
            try:
                from dotenv import dotenv_values
                from openrouter import OpenRouter  # noqa: F401: fail early without the extra
            except ImportError:
                raise ImportError("VLM selection requires uv sync --extra vlm.") from None
            path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[2] / ".env"
            self._api_key = os.environ.get("OPENROUTER_API_KEY")
            if self._api_key is None:
                self._api_key = dotenv_values(path).get("OPENROUTER_API_KEY")
            if not self._api_key or not self._api_key.strip():
                raise ValueError("Set OPENROUTER_API_KEY in the repository .env or environment.")

    def describe(self):
        return {
            "model": self.model, "prompt": self.prompt, "protocol": self.protocol,
            "temperature": self.temperature, "max_completion_tokens": self.max_completion_tokens,
            "timeout": self.timeout, "max_retries": self.max_retries,
            "retry_policy": {
                "owner": "adapter", "backoff_initial_seconds": _BACKOFF_INITIAL,
                "backoff_max_seconds": _BACKOFF_MAX, "backoff_jitter": "equal",
                "respect_retry_after": True, "pacing_scope": "strategy_instance",
                "min_request_interval_seconds": _MIN_REQUEST_INTERVAL,
                "request_jitter_seconds": _REQUEST_JITTER,
            },
            "response_format": deepcopy(self.response_format),
            "provider": {"require_parameters": True},
            "presentation": "nine separate full-resolution PNGs, labelled Image 0 through Image 8",
            "history_turns": 0, "sdk": "openrouter", "sdk_version": version("openrouter"),
        }

    def select(self, images, *, scratchpad=None):
        import httpx
        from openrouter import OpenRouter
        from .selection_strategies import SelectionError

        content = [{"type": "text", "text": self.prompt}]
        if self.uses_scratchpad:
            if not isinstance(scratchpad, str):
                raise ValueError("Supply the current scratchpad as a string.")
            content.append({"type": "text", "text": "Current scratchpad:\n" + scratchpad})
        hashes = []
        for i, pixels in enumerate(images):
            png = png_bytes(pixels)
            hashes.append(sha256(png).hexdigest())
            content.extend([
                {"type": "text", "text": f"Image {i}"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                }},
            ])
        request = dict(
            model=self.model,
            messages=[{"role": "system", "content": self.protocol}, {"role": "user", "content": content}],
            response_format=deepcopy(self.response_format), provider={"require_parameters": True},
            max_completion_tokens=self.max_completion_tokens,
            stream=False, retries=None, timeout_ms=max(1, int(self.timeout * 1000)),
            x_open_router_metadata="enabled",
        )
        if self.temperature is not None:
            request["temperature"] = self.temperature
        audit = {"configuration": self.describe(), "image_sha256": hashes, "attempts": []}
        if self.uses_scratchpad:
            audit["scratchpad_before"] = scratchpad
        started = time.perf_counter()
        context = nullcontext(self.client) if self.client is not None else OpenRouter(api_key=self._api_key)
        with context as client:
            for attempt in range(self.max_retries + 1):
                wait_seconds = max(0.0, self._next_request_at - time.monotonic())
                if wait_seconds:
                    time.sleep(wait_seconds)
                self._next_request_at = (time.monotonic() + _MIN_REQUEST_INTERVAL
                                         + _TIMING_RANDOM.uniform(0, _REQUEST_JITTER))
                attempt_start = time.perf_counter()
                record = {"attempt": attempt + 1, "wait_seconds": wait_seconds}
                audit["attempts"].append(record)
                try:
                    result = client.chat.send(**request)
                    response = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                    record["response"] = response
                except Exception as exc:
                    # Never persist exception text, headers or request bodies: SDK
                    # errors can contain credentials. Status/type suffice for audit.
                    status = getattr(exc, "status_code", None)
                    record.update(error_type=type(exc).__name__, status_code=status,
                                  seconds=time.perf_counter() - attempt_start)
                    transient = isinstance(exc, httpx.TransportError) or status in (408, 429, 500, 502, 503, 504)
                    retry_after = _retry_after_seconds(exc)
                    record.update(retryable=transient, retry_after_seconds=retry_after)
                    if transient:
                        ceiling = min(_BACKOFF_MAX, _BACKOFF_INITIAL * 2 ** min(attempt, 5))
                        delay = _TIMING_RANDOM.uniform(ceiling / 2, ceiling)
                        if retry_after is not None:
                            # Never cap the server's minimum or jitter below it.
                            delay = max(delay, retry_after + _TIMING_RANDOM.uniform(0, _REQUEST_JITTER))
                        self._next_request_at = max(self._next_request_at, time.monotonic() + delay)
                    if transient and attempt < self.max_retries:
                        record["retry_delay_seconds"] = max(0.0, self._next_request_at - time.monotonic())
                        continue
                    metadata = self._metadata(audit, started)
                    raise SelectionError(
                        f"OpenRouter selection failed ({type(exc).__name__}, HTTP {status}); no image selected.",
                        metadata=metadata,
                    ) from None
                record["seconds"] = time.perf_counter() - attempt_start
                try:
                    choice = _parse_choice(response, scratchpad=self.uses_scratchpad)
                except ValueError as exc:
                    record["validation_error"] = str(exc)
                    raise SelectionError(str(exc), metadata=self._metadata(audit, started)) from None
                audit.update(choice)
                return choice["selected_index"], self._metadata(audit, started)

    @staticmethod
    def _metadata(audit, started):
        attempts = audit["attempts"]
        usage = [(attempt.get("response", {}).get("usage") or {}) for attempt in attempts]
        # A missing usage field is unknown, including after a failed attempt.
        def total(key):
            values = [item.get(key) for item in usage]
            return sum(values) if all(v is not None for v in values) else None
        inference = {
            "api_requests": len(attempts), "api_images_submitted": 9 * len(attempts),
            "api_prompt_tokens": total("prompt_tokens"),
            "api_completion_tokens": total("completion_tokens"), "api_cost_usd": total("cost"),
            "api_seconds": time.perf_counter() - started,
        }
        return {"vlm": audit, "inference": inference}


class OpenRouterScratchpadSelection(OpenRouterSelection):
    """Use the shared transport with a replacement-note response contract."""

    protocol = SCRATCHPAD_PROTOCOL
    response_format = SCRATCHPAD_RESPONSE_FORMAT
    uses_scratchpad = True
