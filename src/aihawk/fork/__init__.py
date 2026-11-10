"""This fork's additions to AIHawk, kept apart from upstream's code so upstream's updates merge without conflicts.

Importing this package switches the features on, once. Upstream's code imports it in one place only, `cli.main`. To add a
feature: write `src/aihawk/fork/<feature>.py` with an `activate()` that wraps what it needs through `_patch.target`, and add it
to FEATURES. docs/fork.md explains the whole approach.
"""

from __future__ import annotations

from aihawk.fork import _patch, providers

FEATURES = (providers,)

_active = False


def activate() -> None:
    global _active
    if _active:
        return
    for feature in FEATURES:
        feature.activate()
    _active = True


def deactivate() -> None:
    """Every feature off, upstream's code as it was: for tests, so upstream's own tests never run with the fork switched on."""
    global _active
    _patch.restore()
    _active = False


activate()
