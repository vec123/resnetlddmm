"""Lazy component Registry.

Maps ``(category, name)`` 
-> a ``"module.path:ClassName"`` STRING, resolved
(imported) only on the first ``Registry.create()`` call for that entry.

Storing strings instead of live imports means: 
nothing is imported until something asks for it by name.
"""

from importlib import import_module


class Registry:
    """Maps (category, name) -> "module:QualName",
      resolved on first use."""

    _entries: dict = {}

    @classmethod
    def register(cls, category: str, name: str, target: str) -> None:
        """`target` is a STRING like "src.learning.models.folding_decoder:FoldingDecoder".
        Storing a string (not the class) is what keeps the import lazy."""
        cls._entries[(category, name)] = target

    @classmethod
    def create(cls, category: str, name: str, **kwargs):
        """Resolve, import, instantiate. Unknown name -> ValueError listing valid names."""
        key = (category, name)
        if key not in cls._entries:
            raise ValueError(
                f"no {category!r} registered under name {name!r}; "
                f"available: {cls.available(category)}"
            )
        module_path, qualname = cls._entries[key].split(":")
        module = import_module(module_path)
        target_cls = getattr(module, qualname)
        return target_cls(**kwargs)

    @classmethod
    def available(cls, category: str) -> list:
        """Names in a category -- powers --help and error messages."""
        return sorted(name for (cat, name) in cls._entries if cat == category)


# --------------------------------------------------------------------------- #
# Registrations: one line per component that exists TODAY. 
# Each target string is checked against the real file it points to 
# --------------------------------------------------------------------------- #
Registry.register("encoder", "group_encoder",
                   "src.learning.models.group_encoder:GroupEncoder")

Registry.register("decoder", "folding",
                   "src.learning.models.folding_decoder:FoldingDecoder")

Registry.register("decoder", "sphere_folding",
                   "src.learning.models.folding_decoder:SphereFoldingDecoder")

Registry.register("transformer", "se3",
                   "src.learning.modules.equivariant.transformer:SE3Transformer")
Registry.register("transformer", "equiformer",
                   "src.learning.modules.equivariant.equiformer:EquiformerTransformer")

Registry.register("graph_builder", "radius",
                   "src.learning.data.builders:RadiusGraphBuilder")
Registry.register("graph_builder", "knn",
                   "src.learning.data.builders:KNNGraphBuilder")

Registry.register("latent_head", "gaussian",
                   "src.learning.models.latent_heads:GaussianLatentHead")
Registry.register("latent_head", "deterministic",
                   "src.learning.models.latent_heads:DeterministicLatentHead")

Registry.register("readout", "attention",
                   "src.learning.modules.transformers.perceiver_encoder:PerceiverReducer")
# readout="mean" has no class of its own -- the LatentHead base computes it inline
# as a weighted global_add_pool (latent_heads.py:_reduce), not through a component.
# Nothing to register until that path is extracted into a Strategy class of its own.

