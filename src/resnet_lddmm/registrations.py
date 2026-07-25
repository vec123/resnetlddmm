"""Registrations for resnet_lddmm components.

Each task adds its component's line here when created.
The registry resolves (category, name) pairs to lazy-loaded classes.
"""

from src.learning.registry import Registry

# Each implemented component goes here.
# Format: Registry.register("category", "name", "module.path:ClassName")

# STEPS T4: Conditioning ABC
Registry.register("conditioning", "none", "src.resnet_lddmm.conditioning.base:NoConditioning")

# STEPS T6: VelocityField + TimeVaryingField + StationaryField
Registry.register("field", "time_varying", "src.resnet_lddmm.fields.time_varying:TimeVaryingField")
Registry.register("field", "stationary", "src.resnet_lddmm.fields.time_varying:StationaryField")

# STEPS T9: DataTerm + CDData + L2Data + EMDData
Registry.register("data_term", "chamfer", "src.resnet_lddmm.losses.data_terms:CDData")
Registry.register("data_term", "l2", "src.resnet_lddmm.losses.data_terms:L2Data")
Registry.register("data_term", "emd", "src.resnet_lddmm.losses.data_terms:EMDData")

# STEPS T10: ShapeCode + NoCode
Registry.register("code", "none", "src.resnet_lddmm.codes.none:NoCode")

# IsometryLoss (shape-preserving regularization)
Registry.register("iso_loss", "isometry", "src.resnet_lddmm.losses.data_terms:IsometryLoss")
