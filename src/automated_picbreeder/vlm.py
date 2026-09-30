"""OpenRouter transport and strict parsing for direct VLM selection.

Optional SDK imports live inside the adapter. No breeding logic or numeric
image scores belong here, and credentials never enter configuration or records.
"""

import base64
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import math
from numbers import Real
import os
from pathlib import Path
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


def _parse_choice(response):
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
    if not isinstance(choice, dict) or set(choice) != {"selected_index", "reason"}:
        raise ValueError("Expected exactly selected_index and reason in the VLM response.")
    position = choice["selected_index"]
    if type(position) is not int or not 0 <= position < 9:
        raise ValueError("VLM selected_index must be an integer from 0 to 8.")
    if not isinstance(choice["reason"], str) or not choice["reason"].strip():
        raise ValueError("The VLM must supply a nonempty reason.")
    return choice


class OpenRouterSelection:
    """Own request construction, credential loading, retries and response audit."""

    def __init__(self, *, model, prompt, temperature, max_completion_tokens,
                 timeout, max_retries, env_file, client):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Set an explicit OpenRouter model with model= or --vlm-model.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The VLM prompt must be nonempty.")
        for name, value, minimum, maximum in (
            ("temperature", temperature, 0, 2), ("timeout", timeout, 0, math.inf),
        ):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not minimum <= value <= maximum
                    or (name == "timeout" and value <= 0)):
                raise ValueError(f"Invalid VLM {name}.")
        for name, value, minimum in (("max_completion_tokens", max_completion_tokens, 1),
                                     ("max_retries", max_retries, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.model, self.prompt = model, prompt
        self.temperature, self.timeout = float(temperature), float(timeout)
        self.max_completion_tokens, self.max_retries = max_completion_tokens, max_retries
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
            "model": self.model, "prompt": self.prompt, "protocol": PROTOCOL,
            "temperature": self.temperature, "max_completion_tokens": self.max_completion_tokens,
            "timeout": self.timeout, "max_retries": self.max_retries,
            "response_format": deepcopy(RESPONSE_FORMAT),
            "provider": {"require_parameters": True},
            "presentation": "nine separate full-resolution PNGs, labelled Image 0 through Image 8",
            "history_turns": 0, "sdk": "openrouter", "sdk_version": version("openrouter"),
        }

    def select(self, images):
        import httpx
        from openrouter import OpenRouter
        from .selection_strategies import SelectionError

        content = [{"type": "text", "text": self.prompt}]
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
            messages=[{"role": "system", "content": PROTOCOL}, {"role": "user", "content": content}],
            response_format=deepcopy(RESPONSE_FORMAT), provider={"require_parameters": True},
            temperature=self.temperature, max_completion_tokens=self.max_completion_tokens,
            stream=False, retries=None, timeout_ms=max(1, int(self.timeout * 1000)),
            x_open_router_metadata="enabled",
        )
        audit = {"configuration": self.describe(), "image_sha256": hashes, "attempts": []}
        started = time.perf_counter()
        context = nullcontext(self.client) if self.client is not None else OpenRouter(api_key=self._api_key)
        with context as client:
            for attempt in range(self.max_retries + 1):
                attempt_start = time.perf_counter()
                record = {"attempt": attempt + 1}
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
                    if transient and attempt < self.max_retries:
                        time.sleep(2 ** min(attempt, 3))
                        continue
                    metadata = self._metadata(audit, started)
                    raise SelectionError(
                        f"OpenRouter selection failed ({type(exc).__name__}, HTTP {status}); no image selected.",
                        metadata=metadata,
                    ) from None
                record["seconds"] = time.perf_counter() - attempt_start
                try:
                    choice = _parse_choice(response)
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
