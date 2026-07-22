"""Rot guard: verify all registered resnet_lddmm components can be imported."""

from importlib import import_module
import pytest

# Import to trigger any registrations
import src.resnet_lddmm.registrations  # noqa: F401
from src.learning.registry import Registry


def get_resnet_lddmm_entries():
    """Filter Registry entries to only those from resnet_lddmm package."""
    entries = []
    for (category, name), target in Registry._entries.items():
        if target.startswith("src.resnet_lddmm."):
            entries.append((category, name, target))
    return entries


@pytest.mark.parametrize(
    "category,name,target",
    get_resnet_lddmm_entries(),
    ids=lambda x: f"{x[0]}:{x[1]}",
)
def test_registered_component_resolves(category, name, target):
    """Verify each registered component's target string can be imported and resolved."""
    module_path, qualname = target.split(":")
    module = import_module(module_path)
    cls = getattr(module, qualname)
    assert cls is not None


def test_registrations_module_imports():
    """Verify the registrations module itself imports cleanly."""
    import src.resnet_lddmm.registrations  # noqa: F401
