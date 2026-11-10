"""The fork's gateways and Jev's routes (src/aihawk/fork/providers.py): OrcaRouter stays the default and unchanged, OpenRouter
works the same way through its own address and key, and Jev goes to TypeSafe directly when there is a TypeSafe key. No network:
scripted endpoints stand in."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError

import aihawk.fork
from aihawk import llm, shortlist
from aihawk.apply import server_parameters
from aihawk.config import ConfigError, load_config
from aihawk.database import Job
from aihawk.fork import providers
from aihawk.llm import Spend
from aihawk.session import Session
from aihawk.shortlist import question

ASKED = question("Senior Python work.", "Rome, Italy")
JOB = Job("g:a:1", "Acme", "Python Engineer", "Rome", "u", "a", "Build Python services.")
COMPLETION = {
    "id": "gen-123", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok": true}'}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
}


@pytest.fixture(autouse=True, scope="module")
def fork_on_for_this_module_only() -> Any:
    """The fork's features on for these tests, and off after them: upstream's own tests run on upstream's code alone."""
    aihawk.fork.activate()
    yield
    aihawk.fork.deactivate()


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ORCAROUTER_API_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "AIHAWK_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def settings(tmp_path: Path, text: str) -> None:
    (tmp_path / "aihawk.yaml").write_text(text, encoding="utf-8")


def mocked(client: AsyncOpenAI, handler: Any) -> AsyncOpenAI:
    """The same client, its settings kept, with a scripted endpoint behind it."""
    return client.copy(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def scripted(base: str, handler: Any, key: str = "k") -> AsyncOpenAI:
    return mocked(AsyncOpenAI(base_url=base, api_key=key, max_retries=0), handler)


# The default stays OrcaRouter, exactly as upstream has it.


def test_orcarouter_is_still_the_default_with_upstreams_own_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca")
    config = load_config(tmp_path)
    client = llm.make_client(config)
    upstream = llm.make_client.__wrapped__(config)
    assert config.key_file == tmp_path / "key" and config.api_key() == "sk-orca"
    assert (str(client.base_url), client.api_key, client.max_retries, client.timeout) == (
        str(upstream.base_url), upstream.api_key, upstream.max_retries, upstream.timeout)
    # No TypeSafe key: Jev goes through OrcaRouter under its full name.
    assert not providers.decision_direct(config) and providers.decision_model(config, "typesafe/jev-1.14") == "typesafe/jev-1.14"


def test_orcarouter_chat_reads_its_request_header_and_sends_no_routing_extra() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPLETION, headers={"X-Orca-Request-Id": "orca-1"})

    _, request_id = asyncio.run(llm.chat(scripted("https://api.orcarouter.ai/v1", handler), model="m", messages=[]))
    assert request_id == "orca-1" and "provider" not in json.loads(seen[0].content)


# OpenRouter: its own address and key, the same calls.


def test_openrouter_has_its_own_key_and_upstreams_retries_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings(tmp_path, "provider: openrouter\n")
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca")
    config = load_config(tmp_path)
    # The OrcaRouter key is never sent to OpenRouter.
    assert config.api_key() == "" and config.key_file == tmp_path / "openrouter-key"
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        llm.make_client(config)
    config.save_key("sk-or-typed")
    client = llm.make_client(config)
    assert str(client.base_url) == "https://openrouter.ai/api/v1/" and client.api_key == "sk-or-typed"
    assert (tmp_path / "openrouter-key").read_text(encoding="utf-8") == "sk-or-typed" and not (tmp_path / "key").exists()
    assert client.max_retries == 5 and client.timeout == llm.TIMEOUT
    assert client.default_headers["X-Title"] == "AIHawk"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert config.api_key() == "sk-or-env" and config.key_from_environment()


def test_the_variable_wins_over_the_settings_and_a_wrong_name_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIHAWK_PROVIDER", "OpenRouter")
    assert providers.gateway(load_config(tmp_path)) is providers.OPENROUTER
    monkeypatch.setenv("AIHAWK_PROVIDER", "nowhere")
    with pytest.raises(ConfigError):
        providers.gateway(load_config(tmp_path))
    monkeypatch.delenv("AIHAWK_PROVIDER")
    settings(tmp_path, "provider: nowhere\n")
    with pytest.raises(ConfigError):
        providers.gateway(load_config(tmp_path))


def test_openrouter_structured_calls_keep_the_schema_and_ask_for_providers_that_take_it() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPLETION)

    client = scripted("https://openrouter.ai/api/v1", handler)
    assert asyncio.run(llm.structured(client, "m", "sys", "user", "thing", {"type": "object"})) == {"ok": True}
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}
    # The generation's id, for its cost record, from the body.
    assert asyncio.run(llm.chat(client, model="m", messages=[]))[1] == "gen-123"


def test_openrouter_prices_and_charges_are_counted_the_same_way() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]})
        assert request.url.params["id"] == "gen-123"
        return httpx.Response(200, json={"data": {"total_cost": 0.5}})

    client = scripted("https://openrouter.ai/api/v1", handler)
    spend = Spend()
    asyncio.run(spend.load_prices(client))
    usage = type("Usage", (), {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": None})()
    assert spend.note("m", usage, "gen-123") == pytest.approx(1000e-6 + 20e-6)
    asyncio.run(spend.settle(client))
    assert spend.total == pytest.approx(0.5)


def test_errors_name_the_service_that_failed() -> None:
    for base, name in (("https://api.orcarouter.ai/v1", "OrcaRouter"), ("https://openrouter.ai/api/v1", "OpenRouter"), ("https://api.typesafe.ai/v1", "TypeSafe")):
        client = scripted(base, lambda request: httpx.Response(400, json={"error": "bad"}))
        with pytest.raises(Exception) as caught:
            asyncio.run(client.post("/systemone", body={}, cast_to=dict))
        assert llm.explain(caught.value).startswith(f"{name} rejected the request.")


# Jev: through the gateway, or on TypeSafe directly.


def jev(seen: list[httpx.Request], status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": "no"})
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {name: {"type": "noul", "noul": 0.75} for name in body["questions"]}})

    return handler


@pytest.mark.parametrize("base", ["https://api.orcarouter.ai/v1", "https://openrouter.ai/api/v1", "https://api.typesafe.ai/v1"])
def test_jev_asks_the_same_question_in_the_same_shape_on_every_route(base: str) -> None:
    seen: list[httpx.Request] = []
    assert asyncio.run(shortlist.fits(scripted(base, jev(seen)), "typesafe/jev-1.14", "cv", [JOB], ASKED, lambda _: None, lambda _: None)) == [0.75]
    (request,) = seen
    assert str(request.url) == base + "/systemone"
    body = json.loads(request.content)
    assert body["state"]["job_postings"] == {"p00": "Python Engineer at Acme. Location: Rome\n\nBuild Python services."}
    assert body["questions"]["p00_fit"]["type"] == "noul"


def session_on(tmp_path: Path, handler: Any) -> Session:
    """A session whose gateway client has a scripted endpoint behind it, as the session would make it."""
    made = Session(tmp_path)
    client = mocked(llm.make_client(made.config), handler)
    providers._configs[client] = made.config
    made._client = client
    return made


def test_with_a_typesafe_key_jev_goes_to_typesafe_and_the_rest_stays_on_openrouter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings(tmp_path, "provider: openrouter\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    seen: list[httpx.Request] = []
    made = session_on(tmp_path, jev(seen))
    try:
        values = asyncio.run(shortlist.fits(made.client, made.config.decision_model, "cv", [JOB], ASKED, lambda _: None, lambda _: None))
        assert values == [0.75]
        (request,) = seen
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == "Bearer ts-key" and "x-title" not in request.headers
        assert json.loads(request.content)["model"] == "jev-1.14"
        assert made.client.api_key == "sk-or"
    finally:
        made.close()


def test_decision_provider_gateway_keeps_jev_on_the_gateway_even_with_a_typesafe_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings(tmp_path, "provider: openrouter\ndecision_provider: gateway\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    seen: list[httpx.Request] = []
    made = session_on(tmp_path, jev(seen))
    try:
        asyncio.run(shortlist.fits(made.client, made.config.decision_model, "cv", [JOB], ASKED, lambda _: None, lambda _: None))
        assert str(seen[0].url) == "https://openrouter.ai/api/v1/systemone" and json.loads(seen[0].content)["model"] == "typesafe/jev-1.14"
    finally:
        made.close()


def test_decision_provider_typesafe_without_a_key_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings(tmp_path, "decision_provider: typesafe\n")
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca")
    made = session_on(tmp_path, jev([]))
    try:
        with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
            asyncio.run(shortlist.fits(made.client, made.config.decision_model, "cv", [JOB], ASKED, lambda _: None, lambda _: None))
    finally:
        made.close()


def test_a_refused_typesafe_key_is_forgotten_and_the_gateway_key_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca")
    providers.save_typesafe_key(load_config(tmp_path), "ts-typed")
    made = session_on(tmp_path, jev([], status=401))
    try:
        with pytest.raises(AuthenticationError):
            asyncio.run(shortlist.fits(made.client, made.config.decision_model, "cv", [JOB], ASKED, lambda _: None, lambda _: None))
        said = made.refuse_key()
        assert "TypeSafe refused the key" in said and not (tmp_path / "typesafe-key").exists()
        assert made.config.api_key() == "sk-orca" and not providers.decision_direct(made.config)
        # The next refusal is the gateway's again, in upstream's words.
        assert "OrcaRouter refused the key" in made.refuse_key()
    finally:
        made.close()


def test_a_refused_openrouter_key_is_named_with_openrouters_words(tmp_path: Path) -> None:
    settings(tmp_path, "provider: openrouter\n")
    made = Session(tmp_path)
    try:
        made.config.save_key("sk-or-wrong")
        said = made.refuse_key()
        assert said.startswith("OpenRouter refused the key") and "openrouter.ai/keys" in said and "orcarouter" not in said.lower()
        assert not (tmp_path / "openrouter-key").exists()
    finally:
        made.close()


def test_changing_the_gateway_on_the_settings_screen_makes_the_client_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    made = Session(tmp_path)
    try:
        assert made.client.api_key == "sk-orca"
        made.change_setting("provider", "OpenRouter")
        assert made.client.api_key == "sk-or" and "openrouter.ai" in str(made.client.base_url)
        from aihawk import tui

        assert tui.KEY_QUESTION == "Your OpenRouter key." and tui.KEY_ENV == "OPENROUTER_API_KEY"
        with pytest.raises(ConfigError):
            made.change_setting("provider", "nowhere")
        made.change_setting("provider", "")
        assert made.client.api_key == "sk-orca" and tui.KEY_QUESTION == "Your OrcaRouter key."
    finally:
        made.close()


def test_the_browser_is_given_no_model_key_of_any_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in providers.KEY_ENVS:
        monkeypatch.setenv(name, "secret")
    environment = server_parameters(tmp_path, tmp_path).env or {}
    assert not set(providers.KEY_ENVS) & set(environment)
