"""Registrations for resnet_lddmm components.

Each task adds its component's line here when created.
The registry resolves (category, name) pairs to lazy-loaded classes.
"""

from src.learning.registry import Registry

# Each implemented component goes here.
# Format: Registry.register("category", "name", "module.path:ClassName")

# STEPS T4: Conditioning ABC
Registry.register("conditioning", "none", "src.resnet_lddmm.conditioning.base:NoConditioning")

# STEPS T6: VelocityField + TimeVaryingField
Registry.register("field", "time_varying", "src.resnet_lddmm.fields.time_varying:TimeVaryingField")
