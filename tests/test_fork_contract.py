"""The fork's drift guard: what its features wrap still exists upstream in the shape they were written against, and the one
hook in upstream's code is still there. After merging upstream, a failure here names what to update in src/aihawk/fork."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import aihawk.fork
from aihawk.fork import _patch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def fork_on_for_this_module_only() -> Any:
    """The fork's features on for these tests, and off after them: upstream's own tests run on upstream's code alone."""
    aihawk.fork.activate()
    yield
    aihawk.fork.deactivate()


def test_switching_the_fork_off_gives_back_upstreams_code_exactly() -> None:
    from aihawk import llm, session, tui
    from aihawk.config import Config

    wrapped = (llm.chat, llm.make_client, session.Session.refuse_key, tui.AIHawk.setting_rows, vars(Config)["key_file"])
    aihawk.fork.deactivate()
    try:
        assert llm.chat is wrapped[0].__wrapped__ and llm.make_client is wrapped[1].__wrapped__
        assert session.Session.refuse_key is not wrapped[2] and tui.AIHawk.setting_rows is not wrapped[3]
        assert vars(Config)["key_file"] is not wrapped[4] and tui.KEY_QUESTION == "Your OrcaRouter key."
    finally:
        aihawk.fork.activate()


def test_every_feature_finds_what_it_wraps() -> None:
    # Switching on from upstream's code as it is now: a feature whose target changed raises UpstreamChanged here.
    aihawk.fork.deactivate()
    aihawk.fork.activate()
    assert aihawk.fork._active


def test_the_hook_that_switches_the_features_on_is_in_upstreams_cli() -> None:
    assert "import aihawk.fork" in (ROOT / "src" / "aihawk" / "cli.py").read_text(encoding="utf-8")


def test_a_changed_upstream_function_stops_startup_with_its_name() -> None:
    import types

    module = types.ModuleType("aihawk.example")

    def chat(other: object) -> None: ...

    module.chat = chat  # type: ignore[attr-defined]
    with pytest.raises(_patch.UpstreamChanged, match="aihawk.example.chat"):
        _patch.target(module, "chat", ("client",))
    with pytest.raises(_patch.UpstreamChanged, match="no longer exists"):
        _patch.target(module, "gone", ())
