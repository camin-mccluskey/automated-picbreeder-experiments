"""Exercise the actual OpenRouter SDK over a mocked HTTP transport; no paid calls."""

import base64
from datetime import datetime, timezone
from email.utils import format_datetime
from io import BytesIO
import json
from random import Random
import random
import re
from types import SimpleNamespace
from urllib.parse import unquote

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("openrouter")
import httpx
from openrouter import OpenRouter

from automated_picbreeder.breeding import BreedingSession
from automated_picbreeder.cppn import render
from automated_picbreeder.experiment import ExperimentSettings, run_experiment
from automated_picbreeder.experiment_batch import run_batch
from automated_picbreeder.experiment_viewer import write_viewer
from automated_picbreeder.selection_cli import build_parser
from automated_picbreeder.selection_strategies import (
    SelectionError, VLMSelectionStrategy, VLMScratchpadSelectionStrategy,
)
from automated_picbreeder.vlm import PROTOCOL
from automated_picbreeder import vlm


def response(position=4, *, text=None, finish="stop", usage=True, scratchpad=None):
    choice = {"selected_index": position, "reason": "Its repeated curves interest me."}
    if scratchpad is not None:
        choice["scratchpad"] = scratchpad
    result = {
        "id": "generation-test", "created": 1, "model": "test/vision",
        "object": "chat.completion", "system_fingerprint": None,
        "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                     "message": {"role": "assistant", "content": text if text is not None else
                                 json.dumps(choice)}}],
    }
    if usage:
        result["usage"] = {"prompt_tokens": 90, "completion_tokens": 20, "total_tokens": 110, "cost": .002}
    return result


@pytest.fixture
def images():
    return [np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in range(9)]


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])
    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds
    monkeypatch.setattr(vlm, "time", SimpleNamespace(
        monotonic=lambda: clock.now, perf_counter=lambda: clock.now,
        time=lambda: 1_800_000_000 + clock.now, sleep=sleep))
    return clock


@pytest.fixture
def sdk():
    clients = []
    def factory(handler, **kwargs):
        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenRouter(api_key="test-secret", client=http, **kwargs)
        clients.append(http)
        return client
    yield factory
    for client in clients:
        client.close()


def decode_images(payload):
    return [np.asarray(Image.open(BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))))
            for part in payload["messages"][1]["content"] if part["type"] == "image_url"]


def test_sdk_request_has_nine_original_images_strict_schema_and_no_history(sdk, images):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(handler))
    rng = Random(7)
    state = rng.getstate()
    for _ in range(2):
        decision = strategy.choose(images, rng=rng)
        assert decision.position == 4 and decision.mode == "vlm"
        assert decision.scores is None and decision.evaluation is None
    assert rng.getstate() == state
    assert requests[0] == requests[1]
    payload = requests[0]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["content"] == PROTOCOL
    parts = payload["messages"][1]["content"]
    assert parts[0]["text"] == "choose the most interesting image to you"
    assert [p["text"] for p in parts[1:] if p["type"] == "text"] == [f"Image {i}" for i in range(9)]
    for actual, expected in zip(decode_images(payload), images):
        np.testing.assert_array_equal(actual, expected)
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] is True and schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["selected_index"]["enum"] == list(range(9))
    assert payload["provider"]["require_parameters"] is True
    assert "seed" not in payload
    assert decision.metadata["inference"]["api_requests"] == 1
    assert decision.metadata["inference"]["api_images_submitted"] == 9
    assert decision.metadata["inference"]["api_cost_usd"] == .002
    assert len(decision.metadata["vlm"]["image_sha256"]) == 9
    assert "test-secret" not in json.dumps(decision.metadata)


@pytest.mark.parametrize("strategy_type", [VLMSelectionStrategy, VLMScratchpadSelectionStrategy])
@pytest.mark.parametrize("options", [{}, {"temperature": None}, {"temperature": 0},
                                     {"temperature": .7, "max_completion_tokens": 512}])
def test_sdk_optional_temperature_and_completion_budget(sdk, images, strategy_type, options):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        note = "Continue exploring." if strategy_type is VLMScratchpadSelectionStrategy else None
        return httpx.Response(200, json=response(scratchpad=note))
    strategy = strategy_type(model="test/vision", client=sdk(handler), **options)
    decision = strategy.choose(images, rng=Random(7))
    payload = requests[0]
    temperature = options.get("temperature")
    if temperature is None:
        assert "temperature" not in payload
    else:
        assert payload["temperature"] == temperature
    budget = options.get("max_completion_tokens", 8192)
    assert payload["max_completion_tokens"] == budget
    configuration = decision.metadata["vlm"]["configuration"]
    assert configuration["temperature"] == temperature
    assert configuration["max_completion_tokens"] == budget


@pytest.mark.parametrize("text", [
    '{"selected_index":9,"reason":"x"}', '{"selected_index":-1,"reason":"x"}',
    '{"selected_index":true,"reason":"x"}', '{"selected_index":1.0,"reason":"x"}',
    '{"selected_index":"1","reason":"x"}', '{"selected_index":1,"reason":""}',
    '{"selected_index":1}', '{"selected_index":1,"reason":"x","score":10}',
    '```json\n{"selected_index":1,"reason":"x"}\n```', '[]', 'not json',
])
def test_invalid_choices_fail_without_retry_or_random_fallback(sdk, images, text):
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(200, json=response(text=text))
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(handler))
    with pytest.raises(SelectionError) as error:
        strategy.choose(images, rng=Random(0))
    assert len(attempts) == 1
    assert error.value.metadata["vlm"]["attempts"][0]["response"]["choices"][0]["message"]["content"] == text


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls"])
def test_incomplete_or_refused_completion_is_not_selected(sdk, images, finish):
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(
        lambda request: httpx.Response(200, json=response(finish=finish))))
    with pytest.raises(SelectionError):
        strategy.choose(images, rng=Random(0))


@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_transient_retry_is_counted_and_uses_identical_request(sdk, images, failure):
    requests = []
    def handler(request):
        requests.append(request.content)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("secret detail", request=request)
            return httpx.Response(failure, json={"error": {"message": "test-secret", "code": failure}})
        return httpx.Response(200, json=response())
    decision = VLMSelectionStrategy(model="test/vision", client=sdk(handler)).choose(images, rng=Random(0))
    assert len(requests) == 2 and requests[0] == requests[1]
    assert decision.metadata["inference"]["api_requests"] == 2
    assert decision.metadata["inference"]["api_images_submitted"] == 18
    assert decision.metadata["inference"]["api_cost_usd"] is None
    assert "test-secret" not in json.dumps(decision.metadata)


@pytest.mark.parametrize("status,expected", [(401, 1), (400, 1), (402, 1), (429, 6), (503, 6)])
def test_retry_limit_and_nontransient_failures(sdk, images, status, expected):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "test-secret", "code": status}})
    with pytest.raises(SelectionError) as error:
        VLMSelectionStrategy(model="test/vision", client=sdk(handler)).choose(images, rng=Random(0))
    assert len(requests) == expected
    assert "test-secret" not in str(error.value) + json.dumps(error.value.metadata)


@pytest.mark.parametrize("headers,minimum", [
    ({"Retry-After": "120"}, 120),
    ({"Retry-After": "1.5"}, 1.5),
    ({"Retry-After": format_datetime(datetime.fromtimestamp(1_800_000_120, timezone.utc), usegmt=True)}, 120),
    ({"retry-after-ms": "2500", "Retry-After": "120"}, 2.5),
    ({"retry-after-ms": "bad", "Retry-After": "120"}, 120),
])
@pytest.mark.parametrize("status", [429, 503])
def test_server_retry_wait_is_respected_and_audited(sdk, images, clock, monkeypatch, headers, minimum, status):
    monkeypatch.setattr(vlm, "_MIN_REQUEST_INTERVAL", 0)
    starts = []
    def handler(request):
        starts.append(clock.now)
        if len(starts) == 1:
            return httpx.Response(status, headers={**headers, "x-secret": "test-secret"},
                                  json={"error": {"message": "test-secret", "code": status}})
        return httpx.Response(200, json=response())
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(handler))
    result = strategy.choose(images, rng=Random(0))
    assert minimum <= starts[1] - starts[0] <= max(2, minimum + 1)
    attempts = result.metadata["vlm"]["attempts"]
    assert attempts[0]["retry_after_seconds"] == minimum
    assert attempts[0]["retry_delay_seconds"] == attempts[1]["wait_seconds"] == starts[1]
    assert result.metadata["inference"]["api_seconds"] == starts[1]
    assert "test-secret" not in json.dumps(result.metadata)


@pytest.mark.parametrize("header", [None, "bad", "-1", "NaN", "Infinity", "1e999"])
def test_missing_or_invalid_retry_after_uses_bounded_exponential_jitter(sdk, images, clock, monkeypatch, header):
    monkeypatch.setattr(vlm, "_MIN_REQUEST_INTERVAL", 0)
    monkeypatch.setattr(vlm, "_REQUEST_JITTER", 0)
    starts = []
    def handler(request):
        starts.append(clock.now)
        return httpx.Response(429, headers={} if header is None else {"Retry-After": header},
                              json={"error": {"message": "limited", "code": 429}})
    with pytest.raises(SelectionError) as error:
        VLMSelectionStrategy(model="test/vision", client=sdk(handler), max_retries=7).choose(images, rng=Random(0))
    assert len(starts) == 8 and len(clock.sleeps) == 7
    for actual, ceiling in zip(clock.sleeps, [2, 4, 8, 16, 32, 60, 60]):
        assert ceiling / 2 <= actual <= ceiling
    assert error.value.metadata["inference"]["api_requests"] == 8
    assert "retry_delay_seconds" not in error.value.metadata["vlm"]["attempts"][-1]


def test_successful_calls_are_paced_with_jitter_without_advancing_rngs(sdk, images, clock):
    starts = []
    def handler(request):
        starts.append(clock.now)
        return httpx.Response(200, json=response())
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(handler))
    rng = Random(17)
    selection_state, global_state = rng.getstate(), random.getstate()
    strategy.choose(images, rng=rng)
    clock.now += 1  # Local rendering counts toward the next start interval.
    strategy.choose(images, rng=rng)
    assert 3 <= starts[1] - starts[0] <= 4
    assert 2 <= clock.sleeps[0] <= 3
    clock.now += 10  # No extra sleep when the request interval has already elapsed.
    result = strategy.choose(images, rng=rng)
    assert len(clock.sleeps) == 1
    assert result.metadata["vlm"]["attempts"][0]["wait_seconds"] == 0
    assert rng.getstate() == selection_state and random.getstate() == global_state


def test_adapter_disables_injected_sdk_retries_and_zero_retries_never_sleeps(sdk, images, clock):
    from openrouter.utils import BackoffStrategy, RetryConfig
    starts = []
    def handler(request):
        starts.append(clock.now)
        return httpx.Response(503, headers={"Retry-After": "120"},
                              json={"error": {"message": "limited", "code": 503}})
    client = sdk(handler, retry_config=RetryConfig("backoff", BackoffStrategy(1, 1, 2, 10), True))
    with pytest.raises(SelectionError) as error:
        VLMSelectionStrategy(model="test/vision", client=client, max_retries=0).choose(images, rng=Random(0))
    assert len(starts) == 1 and clock.sleeps == []
    assert error.value.metadata["inference"]["api_requests"] == 1
    assert error.value.metadata["vlm"]["attempts"][0]["retry_after_seconds"] == 120


@pytest.mark.parametrize("strategy_type", [VLMSelectionStrategy, VLMScratchpadSelectionStrategy])
def test_run_preserves_breeding_parity_records_costs_and_offline_viewer(tmp_path, sdk, strategy_type):
    requests, positions = [], iter([6, 0, 8])
    def handler(request):
        requests.append(json.loads(request.content))
        note = f"Note {len(requests)}" if strategy_type is VLMScratchpadSelectionStrategy else None
        return httpx.Response(200, json=response(next(positions), scratchpad=note))
    strategy = strategy_type(model="test/vision", client=sdk(handler))
    output = tmp_path / "run"
    summary = run_experiment(selection_strategy=strategy, output_dir=output,
                             settings=ExperimentSettings(steps=3, size=8), progress=None)
    human = BreedingSession(seed=7)
    for generation, position in enumerate([6, 0, 8]):
        if generation:
            human.evolve(strength=.2, topology=True)
        for actual, key in zip(decode_images(requests[generation]), human.candidates):
            np.testing.assert_array_equal(actual, render(human.genomes[key], human.config, 8))
        human.select(position)
    session = json.loads((output / "session.json").read_text())
    selections = [event for event in session["events"] if event["action"] == "select"]
    assert [e["genome"] for e in selections] == [e["genome"] for e in human.events if e["action"] == "select"]
    assert summary["unique_candidates"] == 25 and summary["evaluated_images"] == 0
    assert summary["api_requests"] == 3 and summary["api_images_submitted"] == 27
    assert summary["api_cost_usd"] == pytest.approx(.006)
    assert all(e["decision"]["metadata"]["vlm"]["reason"] for e in selections)
    report = json.loads((output / "metrics.json").read_text())
    assert report["summary"]["classifier_images"] == 0
    assert report["summary"]["metrics"]["api_cost_usd"]["total"] == pytest.approx(.006)
    assert report["generations"][1]["decision_metadata"]["vlm"]["selected_index"] == 0
    assert (output / "index.html").exists()
    assert "api_requests" in (output / "metrics.csv").read_text()
    assert "test-secret" not in (output / "session.json").read_text()
    if strategy_type is VLMScratchpadSelectionStrategy:
        for index, event in enumerate(selections):
            audit = event["decision"]["metadata"]["vlm"]
            assert audit["scratchpad_before"] == (f"Note {index}" if index else "")
            assert audit["scratchpad"] == f"Note {index + 1}"
            assert event["decision"]["mode"] == "vlm-scratchpad"
        assert report["generations"][1]["decision_metadata"]["vlm"]["scratchpad"] == "Note 2"


@pytest.mark.parametrize("strategy_type", [VLMSelectionStrategy, VLMScratchpadSelectionStrategy])
def test_failure_saves_grid_and_response_audit_without_selecting(tmp_path, sdk, strategy_type):
    note = "Invalid selection must not save this note as accepted." if strategy_type is VLMScratchpadSelectionStrategy else None
    strategy = strategy_type(model="test/vision", client=sdk(
        lambda request: httpx.Response(200, json=response(position=99, scratchpad=note))))
    output = tmp_path / "run"
    with pytest.raises(SelectionError):
        run_experiment(selection_strategy=strategy, output_dir=output,
                       settings=ExperimentSettings(steps=1, size=8), progress=None)
    session = json.loads((output / "session.json").read_text())
    assert not any(e["action"] == "select" for e in session["events"])
    assert len(list((output / "images").glob("*.png"))) == 9
    failure = json.loads((output / "selection_failure.json").read_text())
    assert failure["generation"] == 0 and len(failure["candidate_ids"]) == 9
    assert failure["metadata"]["inference"]["api_requests"] == 1
    if strategy_type is VLMScratchpadSelectionStrategy:
        assert failure["metadata"]["vlm"]["scratchpad_before"] == ""
        assert "scratchpad" not in failure["metadata"]["vlm"]


def test_env_file_loading_environment_precedence_and_missing_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=file-secret\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    strategy = VLMSelectionStrategy(model="test/vision", env_file=env)
    assert strategy._selector._api_key == "file-secret"
    assert "file-secret" not in json.dumps(strategy.describe())
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-secret")
    assert VLMSelectionStrategy(model="test/vision", env_file=env)._selector._api_key == "environment-secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        VLMSelectionStrategy(model="test/vision", env_file=env)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    env.write_text("OPENROUTER_API_KEY=\n")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        VLMSelectionStrategy(model="test/vision", env_file=env)


def test_batch_aggregates_completed_api_costs_and_retains_failed_requests(tmp_path, sdk):
    runs = iter([True, False, True])
    def factory():
        succeeds = next(runs)
        return VLMSelectionStrategy(model="test/vision", client=sdk(
            lambda request: httpx.Response(200, json=response(position=4 if succeeds else 99))))
    output = tmp_path / "batch"
    manifest = run_batch(strategy_factory=factory, output_dir=output, runs=3,
                         settings=ExperimentSettings(steps=1, size=4), progress=None)
    assert manifest["completed_runs"] == 2 and manifest["failed_runs"] == 1
    complete = [entry for entry in manifest["runs"] if entry["status"] == "complete"]
    for entry in complete:
        report = json.loads((output / entry["directory"] / "metrics.json").read_text())
        assert report["summary"]["metrics"]["api_requests"]["total"] == 1
    failed = next(entry for entry in manifest["runs"] if entry["status"] == "failed")
    assert (output / failed["directory"] / "selection_failure.json").exists()
    def payload(path):
        return json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', path.read_text(), re.S)[1])

    overview = write_viewer(output)
    failed_view = next(run for run in payload(overview)["runs"] if run["status"] == "failed")
    page = overview.parent / unquote(failed_view["viewer_url"])
    link = payload(page)["runs"][0]["files"]["selection_failure.json"]
    assert (page.parent / unquote(link)).resolve() == (output / failed["directory"] / "selection_failure.json").resolve()


def test_missing_usage_remains_null_in_metrics_and_run_totals(tmp_path, sdk):
    replies = iter([response(), response(usage=False)])
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(
        lambda request: httpx.Response(200, json=next(replies))))
    output = tmp_path / "run"
    summary = run_experiment(selection_strategy=strategy, output_dir=output,
                             settings=ExperimentSettings(steps=2, size=4), progress=None)
    assert summary["api_cost_usd"] is None and summary["api_prompt_tokens"] is None
    report = json.loads((output / "metrics.json").read_text())
    assert report["summary"]["metrics"]["api_cost_usd"]["total"] is None
    assert report["summary"]["metrics"]["api_requests"]["total"] == 2


@pytest.mark.parametrize("kwargs", [{}, {"model": ""}, {"model": "test/vision", "prompt": " "},
    {"model": "test/vision", "max_retries": -1}, {"model": "test/vision", "timeout": 0},
    {"model": "test/vision", "temperature": float("nan")},
    {"model": "test/vision", "max_completion_tokens": True}])
@pytest.mark.parametrize("strategy_type", [VLMSelectionStrategy, VLMScratchpadSelectionStrategy])
def test_invalid_configuration_before_request(kwargs, strategy_type):
    with pytest.raises(ValueError):
        strategy_type(client=object(), **kwargs)


@pytest.mark.parametrize("strategy_type", [VLMSelectionStrategy, VLMScratchpadSelectionStrategy])
def test_invalid_grid_before_request(images, strategy_type):
    strategy = strategy_type(model="test/vision", client=object())
    for invalid in (images[:8], [image[:, :, 0] for image in images]):
        with pytest.raises(ValueError):
            strategy.choose(invalid, rng=Random(0))


@pytest.mark.parametrize("command", ["vlm", "vlm-scratchpad"])
def test_cli_vlm_generation_defaults(command, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    args = build_parser().parse_args([command, "--vlm-model", "test/vision"])
    description = args.build_strategy(args).describe()
    assert description["temperature"] is None
    assert description["max_completion_tokens"] == 8192


def test_cli_forwards_vlm_options_without_classifier_options(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    args = build_parser().parse_args(["vlm", "--vlm-model", "test/vision", "--vlm-prompt", "custom",
                                    "--temperature", ".7", "--max-completion-tokens", "512",
                                    "--timeout", "20", "--max-retries", "0", "--env-file", str(tmp_path / ".env")])
    description = args.build_strategy(args).describe()
    assert description["prompt"] == "custom" and description["temperature"] == .7
    assert description["max_completion_tokens"] == 512 and description["timeout"] == 20
    assert description["max_retries"] == 0
    for extra in (["--device", "cpu"], ["--imagenet-model", "resnet18"], ["--temperature", "nan"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["vlm", *extra])


def test_scratchpad_replaces_note_and_sends_only_current_images(sdk, images):
    requests, decisions = [], []
    notes = ["Explore branches.", "Now pursue rings.", "Keep the nested rings."]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(scratchpad=notes[len(requests) - 1]))
    strategy = VLMScratchpadSelectionStrategy(model="test/vision", client=sdk(handler))
    rng = Random(7)
    state, global_state = rng.getstate(), random.getstate()
    description = strategy.describe()
    for turn in range(3):
        current = [image + turn for image in images]
        decision = strategy.choose(current, rng=rng)
        decisions.append(decision)
        payload = requests[-1]
        assert len(payload["messages"]) == 2
        parts = payload["messages"][1]["content"]
        assert parts[0]["text"] == "find something interesting"
        assert parts[1]["text"] == "Current scratchpad:\n" + (notes[turn - 1] if turn else "")
        assert [p["text"] for p in parts[2:] if p["type"] == "text"] == [f"Image {i}" for i in range(9)]
        assert len(decode_images(payload)) == 9
        for actual, expected in zip(decode_images(payload), current):
            np.testing.assert_array_equal(actual, expected)
        assert decision.scores is None and decision.evaluation is None
        assert decision.mode == "vlm-scratchpad"
        assert decision.metadata["inference"]["api_images_submitted"] == 9
    assert rng.getstate() == state and random.getstate() == global_state
    assert strategy.describe() == description  # Configuration never contains mutable memory.
    assert description["selection_strategy"] == "vlm-scratchpad"
    assert description["history_turns"] == 0
    assert "Explore branches." not in json.dumps(requests[2])
    assert decisions[0].metadata["vlm"]["scratchpad_before"] == ""
    assert decisions[0].metadata["vlm"]["scratchpad"] == notes[0]
    schema = requests[0]["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) == {"selected_index", "reason", "scratchpad"}
    assert schema["schema"]["properties"]["scratchpad"]["type"] == "string"


@pytest.mark.parametrize("bad_reply", [
    response(), response(scratchpad=""), response(scratchpad="  "),
    response(scratchpad=12), response(scratchpad=[]),
    response(text='{"selected_index":1,"reason":"x","scratchpad":null}'),
    response(text='{"selected_index":1,"reason":"x","scratchpad":"note","extra":1}'),
    response(position=True, scratchpad="note"), response(position=9, scratchpad="note"),
    response(text='{"selected_index":1,"reason":"","scratchpad":"note"}'),
    response(finish="length", scratchpad="note"), response(text="not json"),
])
def test_invalid_scratchpad_response_preserves_previous_note(sdk, images, bad_reply):
    requests = []
    replies = iter([response(scratchpad="Keep this note."), bad_reply, response(scratchpad="New note.")])
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(replies))
    strategy = VLMScratchpadSelectionStrategy(model="test/vision", client=sdk(handler))
    strategy.choose(images, rng=Random(0))
    with pytest.raises(SelectionError) as error:
        strategy.choose(images, rng=Random(0))
    assert len(requests) == 2  # Invalid output is not retried.
    assert error.value.metadata["vlm"]["scratchpad_before"] == "Keep this note."
    result = strategy.choose(images, rng=Random(0))
    assert requests[1] == requests[2]
    assert result.metadata["vlm"]["scratchpad_before"] == "Keep this note."


def test_scratchpad_transport_failure_and_retry_preserve_request(sdk, images):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) in (2, 3, 4):
            return httpx.Response(503, json={"error": {"message": "test-secret", "code": 503}})
        return httpx.Response(200, json=response(scratchpad="First note." if len(requests) == 1 else "Next note."))
    strategy = VLMScratchpadSelectionStrategy(model="test/vision", client=sdk(handler), max_retries=1)
    strategy.choose(images, rng=Random(0))
    with pytest.raises(SelectionError) as error:
        strategy.choose(images, rng=Random(0))
    assert error.value.metadata["vlm"]["scratchpad_before"] == "First note."
    decision = strategy.choose(images, rng=Random(0))
    assert requests[1] == requests[2] == requests[3] == requests[4]
    assert decision.metadata["vlm"]["scratchpad_before"] == "First note."
    assert decision.metadata["inference"]["api_requests"] == 2
    assert "test-secret" not in json.dumps(error.value.metadata)


def test_scratchpad_batch_starts_each_run_empty(tmp_path, sdk):
    notes_sent = []
    def handler(request):
        payload = json.loads(request.content)
        notes_sent.append(payload["messages"][1]["content"][1]["text"])
        return httpx.Response(200, json=response(scratchpad="Continue exploring."))
    manifest = run_batch(
        strategy_factory=lambda: VLMScratchpadSelectionStrategy(model="test/vision", client=sdk(handler)),
        output_dir=tmp_path / "batch", runs=2,
        settings=ExperimentSettings(steps=2, size=4), progress=None,
    )
    assert manifest["completed_runs"] == 2
    assert notes_sent == ["Current scratchpad:\n", "Current scratchpad:\nContinue exploring."] * 2


def test_scratchpad_cli_defaults_and_options(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    parser = build_parser()
    args = parser.parse_args(["vlm-scratchpad", "--vlm-model", "test/vision"])
    strategy = args.build_strategy(args)
    assert isinstance(strategy, VLMScratchpadSelectionStrategy)
    assert strategy.describe()["prompt"] == "find something interesting"
    args = parser.parse_args(["vlm-scratchpad", "--vlm-model", "test/vision", "--vlm-prompt", "custom",
                             "--temperature", ".5", "--max-completion-tokens", "2048", "--max-retries", "0"])
    description = args.build_strategy(args).describe()
    assert description["prompt"] == "custom" and description["temperature"] == .5
    assert description["max_completion_tokens"] == 2048 and description["max_retries"] == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["vlm-scratchpad", "--device", "cpu"])
