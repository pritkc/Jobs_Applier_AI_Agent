"""How this fork's features attach to upstream's code without editing it.

Each feature wraps a few upstream functions or methods at startup. Two rules keep that safe across upstream updates:

- `target` finds what a feature wraps and checks its parameters. When upstream renames, removes or reshapes it, AIHawk stops at
  startup and names it, instead of quietly running without the feature. `tests/test_fork_contract.py` checks the same in CI.
- A wrapper calls the original and changes only what the feature needs, so upstream's own fixes to that function still run.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

# Every name a feature changed, with what it named before: `restore` puts them all back.
_undo: list[tuple[Any, str, Any]] = []


class UpstreamChanged(RuntimeError):
    """Upstream code that a fork feature wraps is no longer what the feature was written against."""


def target(owner: ModuleType | type, name: str, params: tuple[str, ...]) -> Any:
    """`owner.name`, after checking that it still exists and still takes `params` first, in that order."""
    where = f"{getattr(owner, '__module__', '') + '.' if isinstance(owner, type) else ''}{owner.__name__}.{name}"
    found = inspect.getattr_static(owner, name, None) if isinstance(owner, type) else getattr(owner, name, None)
    if found is None:
        raise UpstreamChanged(f"{where} no longer exists upstream: update src/aihawk/fork (see docs/fork.md)")
    function = found.fget if isinstance(found, property) else found
    if params:
        actual = tuple(inspect.signature(function).parameters)
        if actual[: len(params)] != params:
            raise UpstreamChanged(f"{where} now takes {actual}, not {params}: update src/aihawk/fork (see docs/fork.md)")
    return found


def rebind(original: Callable[..., Any], replacement: Callable[..., Any]) -> None:
    """Every aihawk module's name for `original` now names `replacement`. Upstream modules import names directly
    (`from aihawk.llm import chat`), so the wrapper must replace each copy of the name, not only the defining one."""
    replacement.__wrapped__ = original  # type: ignore[attr-defined]
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("aihawk") or module_name.startswith("aihawk.fork"):
            continue
        for attribute, value in list(vars(module).items()):
            if value is original:
                _undo.append((module, attribute, original))
                setattr(module, attribute, replacement)


def replace(owner: Any, name: str, value: Any) -> None:
    """`owner.name` set to `value`, remembering what it was (a class's own attribute, as the class holds it)."""
    _undo.append((owner, name, vars(owner)[name]))
    setattr(owner, name, value)


def restore() -> None:
    """Upstream's code exactly as it was before any feature switched on."""
    while _undo:
        owner, name, value = _undo.pop()
        setattr(owner, name, value)
