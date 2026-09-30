"""Exercise the actual OpenRouter SDK over a mocked HTTP transport; no paid calls."""

import base64
from io import BytesIO
import json
from random import Random

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
from automated_picbreeder.selection_strategies import SelectionError, VLMSelectionStrategy
from automated_picbreeder.vlm import PROTOCOL


def response(position=4, *, text=None, finish="stop", usage=True):
    result = {
        "id": "generation-test", "created": 1, "model": "test/vision",
        "object": "chat.completion", "system_fingerprint": None,
        "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                     "message": {"role": "assistant", "content": text if text is not None else
                                 json.dumps({"selected_index": position, "reason": "Its repeated curves interest me."})}}],
    }
    if usage:
        result["usage"] = {"prompt_tokens": 90, "completion_tokens": 20, "total_tokens": 110, "cost": .002}
    return result


@pytest.fixture
def images():
    return [np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in range(9)]


@pytest.fixture
def sdk():
    clients = []
    def factory(handler):
        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenRouter(api_key="test-secret", client=http)
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
def test_transient_retry_is_counted_and_uses_identical_request(sdk, images, monkeypatch, failure):
    requests = []
    monkeypatch.setattr("automated_picbreeder.vlm.time.sleep", lambda seconds: None)
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


@pytest.mark.parametrize("status,expected", [(401, 1), (400, 1), (402, 1), (429, 3), (503, 3)])
def test_retry_limit_and_nontransient_failures(sdk, images, monkeypatch, status, expected):
    requests = []
    monkeypatch.setattr("automated_picbreeder.vlm.time.sleep", lambda seconds: None)
    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "test-secret", "code": status}})
    with pytest.raises(SelectionError) as error:
        VLMSelectionStrategy(model="test/vision", client=sdk(handler)).choose(images, rng=Random(0))
    assert len(requests) == expected
    assert "test-secret" not in str(error.value) + json.dumps(error.value.metadata)


def test_run_preserves_breeding_parity_records_costs_and_offline_viewer(tmp_path, sdk):
    requests, positions = [], iter([6, 0, 8])
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(next(positions)))
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(handler))
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


def test_failure_saves_grid_and_response_audit_without_selecting(tmp_path, sdk):
    strategy = VLMSelectionStrategy(model="test/vision", client=sdk(
        lambda request: httpx.Response(200, json=response(position=99))))
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
    assert "selection_failure.json" in write_viewer(output).read_text()


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
def test_invalid_configuration_before_request(kwargs):
    with pytest.raises(ValueError):
        VLMSelectionStrategy(client=object(), **kwargs)


def test_invalid_grid_before_request(images):
    strategy = VLMSelectionStrategy(model="test/vision", client=object())
    for invalid in (images[:8], [image[:, :, 0] for image in images]):
        with pytest.raises(ValueError):
            strategy.choose(invalid, rng=Random(0))


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
