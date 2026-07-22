"""Tests for the lazy component Registry.

Stringly-typed targets break silently on refactor: rename a class and the
string still parses, so nothing fails until a run asks for it by name.  The
anti-rot test below resolves EVERY registered string, so a stale registration
fails CI instead of a training run.  One test, catches every rot.
"""

import pytest

from src.learning.registry import Registry

ALL_ENTRIES = sorted(Registry.entries().items())


# --------------------------------------------------------------------------- #
# Anti-rot: every registered string must resolve to the class it names
# --------------------------------------------------------------------------- #
def test_registry_is_not_empty():
    """Guards the anti-rot test itself: an empty parametrize list passes vacuously."""
    assert ALL_ENTRIES


@pytest.mark.parametrize(
    "key,target", ALL_ENTRIES,
    ids=[f"{cat}/{name}" for (cat, name), _ in ALL_ENTRIES],
)
def test_every_registered_target_resolves(key, target):
    category, name = key
    target_cls = Registry.resolve(category, name)

    assert isinstance(target_cls, type), (
        f"{target!r} resolved to {target_cls!r}, which is not a class"
    )
    _, qualname = target.split(":")
    assert target_cls.__name__ == qualname


# --------------------------------------------------------------------------- #
# When resolution DOES fail (accepted run-time failure mode), the error must
# name the broken registration, not surface as a bare importlib traceback.
# --------------------------------------------------------------------------- #
def test_unknown_name_lists_available():
    with pytest.raises(ValueError, match="available"):
        Registry.create("decoder", "banana")


def test_moved_module_fails_naming_the_registration():
    Registry.register("decoder", "_test_broken", "src.learning.no_such_module:X")
    try:
        with pytest.raises(ImportError, match="_test_broken.*moved or renamed"):
            Registry.resolve("decoder", "_test_broken")
    finally:
        del Registry._entries[("decoder", "_test_broken")]


def test_renamed_class_fails_naming_the_registration():
    Registry.register("decoder", "_test_broken", "src.learning.registry:NoSuchClass")
    try:
        with pytest.raises(ImportError, match="NoSuchClass.*renamed without updating"):
            Registry.resolve("decoder", "_test_broken")
    finally:
        del Registry._entries[("decoder", "_test_broken")]
