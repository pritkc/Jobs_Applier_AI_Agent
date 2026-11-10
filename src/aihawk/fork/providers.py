"""OpenRouter as the model gateway, and Jev called on TypeSafe directly.

With nothing set, every wrapper here hands straight through to upstream's code: OrcaRouter, its key, its client, as before.

- `provider: openrouter` in aihawk.yaml, or the variable AIHAWK_PROVIDER (which wins), sends every chat call and the price list
  to OpenRouter with its own key: OPENROUTER_API_KEY, or the file `openrouter-key` that the app's key question writes. The
  client is upstream's own, copied onto OpenRouter's address, so its retries and timeout stay upstream's.
- On OpenRouter each chat call asks to be routed only to providers that take all its parameters, so a form's tools and a plan's
  JSON schema are never dropped. The request's id for the cost record comes from the body, where OpenRouter gives it.
- Jev's /systemone calls go to TypeSafe itself when there is a TypeSafe key (TYPESAFE_API_KEY or the file `typesafe-key`), or
  when `decision_provider: typesafe`; `decision_provider: gateway` keeps them on the gateway. Direct, the `typesafe/` prefix is
  taken off the model's name. OpenRouter serves /systemone too, so Jev also works through OpenRouter with no TypeSafe key.
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import AsyncOpenAI, AuthenticationError

from aihawk import apply, llm, session, shortlist, tui
from aihawk.config import CONFIG_FILE, DEFAULT_BASE_URL, KEY_ENV, KEY_FILE, KEY_URL, Config, ConfigError
from aihawk.fork._patch import rebind, replace, target

PROVIDER_ENV = "AIHAWK_PROVIDER"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"
TYPESAFE_KEY_FILE = "typesafe-key"
TYPESAFE_KEY_URL = "https://typesafe.ai"
DEFAULT_TYPESAFE_BASE_URL = "https://api.typesafe.ai/v1"
DECISION_PROVIDERS = ("auto", "gateway", "typesafe")
# The settings this feature reads from aihawk.yaml, beside upstream's: upstream's loader leaves keys it does not know alone.
SETTINGS = ("provider", "decision_provider", "typesafe_base_url")


@dataclass(frozen=True)
class Gateway:
    name: str
    label: str
    base_url: str
    key_env: str
    key_file: str
    key_url: str
    headers: tuple[tuple[str, str], ...] = ()


ORCAROUTER = Gateway("orcarouter", "OrcaRouter", DEFAULT_BASE_URL, KEY_ENV, KEY_FILE, KEY_URL)
OPENROUTER = Gateway(
    "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter-key", "https://openrouter.ai/keys",
    # OpenRouter's attribution headers: they name the app in OpenRouter's dashboard and carry nothing about the person.
    headers=(("HTTP-Referer", "https://github.com/feder-cr/Jobs_Applier_AI_Agent"), ("X-Title", "AIHawk")),
)
GATEWAYS = {gateway.name: gateway for gateway in (ORCAROUTER, OPENROUTER)}
# Every variable that may hold a model key: the browser is given none of them.
KEY_ENVS = (*(gateway.key_env for gateway in GATEWAYS.values()), TYPESAFE_KEY_ENV)

# The config each gateway client was made for, so Jev's call can find where it should go; and each one's TypeSafe client.
_configs: weakref.WeakKeyDictionary[AsyncOpenAI, Config] = weakref.WeakKeyDictionary()
_direct: weakref.WeakKeyDictionary[AsyncOpenAI, tuple[str, str, AsyncOpenAI]] = weakref.WeakKeyDictionary()
# Set when TypeSafe refused its key, so the screen's next refuse_key forgets that key and not the gateway's.
_typesafe_refused = False


# What the settings say.


def settings(config: Config) -> dict[str, Any]:
    """aihawk.yaml as it stands; empty when there is none or it does not read (upstream's loader reports a broken file)."""
    path = config.workspace / CONFIG_FILE
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _choice(value: object, name: str, choices: tuple[str, ...]) -> str:
    word = str(value).strip().lower()
    if word not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return word


def gateway(config: Config) -> Gateway:
    """The gateway in use: AIHAWK_PROVIDER's, when it is set, or the settings', OrcaRouter when neither says."""
    from_environment = os.environ.get(PROVIDER_ENV, "").strip()
    if from_environment:
        return GATEWAYS[_choice(from_environment, PROVIDER_ENV, tuple(GATEWAYS))]
    return GATEWAYS[_choice(settings(config).get("provider") or ORCAROUTER.name, "provider", tuple(GATEWAYS))]


def decision_provider(config: Config) -> str:
    return _choice(settings(config).get("decision_provider") or "auto", "decision_provider", DECISION_PROVIDERS)


def typesafe_base_url(config: Config) -> str:
    return str(settings(config).get("typesafe_base_url") or DEFAULT_TYPESAFE_BASE_URL).rstrip("/")


def _key(variable: str, path: Path) -> str:
    from_environment = os.environ.get(variable, "").strip()
    if from_environment:
        return from_environment
    return path.read_text(encoding="utf-8-sig").strip() if path.is_file() else ""


def typesafe_key_file(config: Config) -> Path:
    return config.workspace / TYPESAFE_KEY_FILE


def typesafe_key(config: Config) -> str:
    return _key(TYPESAFE_KEY_ENV, typesafe_key_file(config))


def save_typesafe_key(config: Config, key: str) -> None:
    """A TypeSafe key typed in, kept like the gateway's: in its own file, readable by this user only."""
    config.workspace.mkdir(parents=True, exist_ok=True)
    path = typesafe_key_file(config)
    path.touch(mode=0o600, exist_ok=True)
    path.write_text(key.strip(), encoding="utf-8")


def decision_direct(config: Config) -> bool:
    """Whether Jev's calls go to TypeSafe itself rather than through the gateway."""
    chosen = decision_provider(config)
    return chosen == "typesafe" or (chosen == "auto" and bool(typesafe_key(config)))


def decision_model(config: Config, model: str) -> str:
    """Jev's name as the service that answers knows it: TypeSafe's own API names its models without the vendor's prefix."""
    return model.removeprefix("typesafe/") if decision_direct(config) else model


# Who a client or a failure belongs to.


def _host(url: object) -> str:
    try:
        return httpx.URL(str(url)).host if url else ""
    except Exception:  # noqa: BLE001  A fake client's address, or none: no gateway.
        return ""


def gateway_of(client: Any) -> Gateway | None:
    host = _host(getattr(client, "base_url", ""))
    return next((gateway for gateway in GATEWAYS.values() if host and host == _host(gateway.base_url)), None)


def _typesafe_host(host: str) -> bool:
    return host.endswith("typesafe.ai") or any(host == _host(url) for url, _, _ in list(_direct.values()))


def label_of_host(host: str) -> str:
    if host and _typesafe_host(host):
        return "TypeSafe"
    return next((gateway.label for gateway in GATEWAYS.values() if host and host == _host(gateway.base_url)), "OrcaRouter")


def service(error: BaseException) -> str:
    """The service a failed call went to, from its request's address; OrcaRouter, upstream's word, when it cannot be told."""
    request = getattr(error, "request", None)
    return label_of_host(_host(getattr(request, "url", ""))) if request is not None else "OrcaRouter"


def _renamed(text: str, who: str) -> str:
    return text if who == "OrcaRouter" else text.replace("OrcaRouter", who)


# The wrappers. Each calls upstream's original.


def _wrap_make_client(original: Callable[[Config], AsyncOpenAI]) -> Callable[[Config], AsyncOpenAI]:
    def make_client(config: Config) -> AsyncOpenAI:
        chosen = gateway(config)
        if chosen is not ORCAROUTER and not config.api_key():
            raise ConfigError(f"no model key: set {chosen.key_env} (create one at {chosen.key_url})")
        client = original(config)
        if chosen is not ORCAROUTER:
            # Upstream's client, its retries and timeout kept, on the other gateway's address.
            client = client.copy(base_url=chosen.base_url, default_headers=dict(chosen.headers))
        _configs[client] = config
        return client

    return make_client


def _wrap_chat(original: Callable[..., Any]) -> Callable[..., Any]:
    async def chat(client: Any, **request: Any) -> tuple[Any, str]:
        on_openrouter = gateway_of(client) is OPENROUTER
        if on_openrouter:
            extra = dict(request.pop("extra_body", None) or {})
            extra.setdefault("provider", {"require_parameters": True})
            request["extra_body"] = extra
        response, request_id = await original(client, **request)
        if on_openrouter and not request_id:
            # OpenRouter gives the generation's id in the body; its /generation record is read by that id.
            request_id = str(getattr(response, "id", "") or "")
        return response, request_id

    return chat


def _wrap_explain(original: Callable[[Exception], str]) -> Callable[[Exception], str]:
    def explain(error: Exception) -> str:
        return _renamed(original(error), service(error))

    return explain


def direct_client(client: AsyncOpenAI, config: Config) -> AsyncOpenAI:
    """The gateway client copied onto TypeSafe with TypeSafe's key: the same retries and timeout, none of the gateway's headers."""
    url, key = typesafe_base_url(config), typesafe_key(config)
    if not key:
        raise ConfigError(f"decision_provider is typesafe but there is no TypeSafe key: set {TYPESAFE_KEY_ENV} (get one at {TYPESAFE_KEY_URL})")
    kept = _direct.get(client)
    if kept is not None and kept[:2] == (url, key):
        return kept[2]
    made = client.copy(base_url=url, api_key=key, set_default_headers={})
    _direct[client] = (url, key, made)
    return made


def _wrap_fits(original: Callable[..., Any]) -> Callable[..., Any]:
    async def fits(client: Any, model: str, cv: str, jobs: list[Any], asked: str, progress: Callable[[int], None], why: Callable[[str], None]) -> Any:
        global _typesafe_refused
        config = _configs.get(client) if isinstance(client, AsyncOpenAI) else None
        direct = config is not None and decision_direct(config)
        if direct:
            client, model = direct_client(client, config), decision_model(config, model)
        who = "TypeSafe" if direct else (gateway_of(client) or ORCAROUTER).label

        def said(line: str) -> None:
            why(_renamed(line, who))

        try:
            return await original(client, model, cv, jobs, asked, progress, said)
        except AuthenticationError:
            _typesafe_refused = direct
            raise

    return fits


def _wrap_refuse_key(original: Callable[[Any], str]) -> Callable[[Any], str]:
    def refuse_key(self: Any) -> str:
        global _typesafe_refused
        if _typesafe_refused:
            _typesafe_refused = False
            if os.environ.get(TYPESAFE_KEY_ENV, "").strip():
                return f"TypeSafe refused the key in the {TYPESAFE_KEY_ENV} variable. Set it to a key from {TYPESAFE_KEY_URL}, then start AIHawk again."
            typesafe_key_file(self.config).unlink(missing_ok=True)
            return "TypeSafe refused the key, so AIHawk forgot it: scoring goes through the gateway until there is another."
        chosen = gateway(self.config)
        said = original(self)
        if chosen is ORCAROUTER:
            return said
        return said.replace("OrcaRouter", chosen.label).replace(KEY_ENV, chosen.key_env).replace(KEY_URL, chosen.key_url)

    return refuse_key


def _show_gateway(chosen: Gateway) -> None:
    """The key question and the key's row name the gateway in use. Upstream's screen reads these names when it asks."""
    tui.KEY_ENV = chosen.key_env
    tui.KEY_QUESTION = _ORIGINAL_QUESTION.replace("OrcaRouter", chosen.label)
    tui.KEY_HINT = _ORIGINAL_HINT.replace(KEY_URL, chosen.key_url)


def _wrap_session_init(original: Callable[..., None]) -> Callable[..., None]:
    def __init__(self: Any, workspace: Path) -> None:
        original(self, workspace)
        _show_gateway(gateway(self.config))

    return __init__


def _wrap_change_setting(original: Callable[..., None]) -> Callable[..., None]:
    def change_setting(self: Any, name: str, value: str) -> None:
        if name == "provider":
            if os.environ.get(PROVIDER_ENV, "").strip():
                raise ConfigError(f"The gateway comes from the {PROVIDER_ENV} variable, which wins over this setting.")
            value = _choice(value, "provider", tuple(GATEWAYS)) if value.strip() else value
        elif name == "decision_provider":
            value = _choice(value, "decision_provider", DECISION_PROVIDERS) if value.strip() else value
        before = gateway(self.config)
        original(self, name, value)
        if name in SETTINGS:
            # The clients are made again for the new route; the price list too when the gateway changed.
            self._client = None
            now = gateway(self.config)
            if now is not before:
                llm.SPEND.prices, llm.SPEND.loaded = {}, False
            _show_gateway(now)

    return change_setting


def _wrap_setting_rows(original: Callable[[Any], list[tuple[str, str, str]]]) -> Callable[[Any], list[tuple[str, str, str]]]:
    def setting_rows(self: Any) -> list[tuple[str, str, str]]:
        config = self.session.config
        chosen = gateway(config)
        provider_row = ("provider", f"Gateway ({' or '.join(GATEWAYS)})", chosen.name)
        decision_row = ("decision_provider", "Scoring goes to (auto, gateway or typesafe)", decision_provider(config))
        rows: list[tuple[str, str, str]] = []
        for key, label, value in original(self):
            if key == "key":
                rows.append(provider_row)
                label = label.replace("OrcaRouter", chosen.label)
            rows.append((key, label, value))
            if key == "decision_model":
                rows.append(decision_row)
        # A screen whose rows upstream renamed still offers both settings, at the end.
        rows += [row for row in (provider_row, decision_row) if row not in rows]
        return rows

    return setting_rows


def _wrap_server_parameters(original: Callable[..., Any]) -> Callable[..., Any]:
    def server_parameters(upload_dir: Path, home: Path) -> Any:
        parameters = original(upload_dir, home)
        for name in KEY_ENVS:
            (parameters.env or {}).pop(name, None)
        return parameters

    return server_parameters


_ORIGINAL_QUESTION = ""
_ORIGINAL_HINT = ""


def activate() -> None:
    """Find every upstream piece this feature wraps (stopping if one changed), then wrap them."""
    global _ORIGINAL_QUESTION, _ORIGINAL_HINT
    pieces = {
        "make_client": target(llm, "make_client", ("config",)),
        "chat": target(llm, "chat", ("client", "request")),
        "explain": target(llm, "explain", ("error",)),
        "fits": target(shortlist, "fits", ("client", "model", "cv", "jobs", "asked", "progress", "why")),
        "server_parameters": target(apply, "server_parameters", ("upload_dir", "home")),
        "init": target(session.Session, "__init__", ("self", "workspace")),
        "refuse_key": target(session.Session, "refuse_key", ("self",)),
        "change_setting": target(session.Session, "change_setting", ("self", "name", "value")),
        "setting_rows": target(tui.AIHawk, "setting_rows", ("self",)),
        "api_key": target(Config, "api_key", ("self",)),
        "key_file": target(Config, "key_file", ("self",)),
        "key_from_environment": target(Config, "key_from_environment", ("self",)),
    }
    for name in ("KEY_QUESTION", "KEY_HINT", "KEY_ENV"):
        target(tui, name, ())
    _ORIGINAL_QUESTION, _ORIGINAL_HINT = tui.KEY_QUESTION, tui.KEY_HINT
    for name in ("KEY_QUESTION", "KEY_HINT", "KEY_ENV"):
        # Remembered as they are, so that switching the fork off brings back upstream's words.
        replace(tui, name, getattr(tui, name))

    # Each gateway keeps its own key: its own variable, and its own file (`key` stays OrcaRouter's, as upstream has it).
    replace(Config, "key_file", property(lambda self: self.workspace / gateway(self).key_file))
    replace(Config, "api_key", lambda self: _key(gateway(self).key_env, self.key_file))
    replace(Config, "key_from_environment", lambda self: bool(os.environ.get(gateway(self).key_env, "").strip()))

    for name, wrap in (("make_client", _wrap_make_client), ("chat", _wrap_chat), ("explain", _wrap_explain), ("fits", _wrap_fits),
                       ("server_parameters", _wrap_server_parameters)):
        rebind(pieces[name], wrap(pieces[name]))
    replace(session.Session, "__init__", _wrap_session_init(pieces["init"]))
    replace(session.Session, "refuse_key", _wrap_refuse_key(pieces["refuse_key"]))
    replace(session.Session, "change_setting", _wrap_change_setting(pieces["change_setting"]))
    replace(tui.AIHawk, "setting_rows", _wrap_setting_rows(pieces["setting_rows"]))
