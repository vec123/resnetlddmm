"""Registrations for resnet_lddmm components.

Each task adds its component's line here when created.
The registry resolves (category, name) pairs to lazy-loaded classes.
"""

from src.learning.registry import Registry

# Each implemented component goes here.
# Format: Registry.register("category", "name", "module.path:ClassName")

# STEPS T4: Conditioning ABC
Registry.register("conditioning", "none", "src.resnet_lddmm.conditioning.base:NoConditioning")

# STEPS T22: Concat + FiLM + PositionAware conditioning
Registry.register("conditioning", "concat", "src.resnet_lddmm.conditioning.film:ConcatConditioning")
Registry.register("conditioning", "film", "src.resnet_lddmm.conditioning.film:FiLMConditioning")
Registry.register("conditioning", "position_aware", "src.resnet_lddmm.conditioning.position_aware:PositionAware")

# STEPS T6: VelocityField + TimeVaryingField + StationaryField
Registry.register("field", "time_varying", "src.resnet_lddmm.fields.time_varying:TimeVaryingField")
Registry.register("field", "stationary", "src.resnet_lddmm.fields.time_varying:StationaryField")

# SO(3)-equivariant stationary field using e3nn SelfInteraction
Registry.register("field", "equivariant_stationary", "src.resnet_lddmm.fields.equivariant:EquivariantStationaryField")

# SO(3)-equivariant field with message passing over template
Registry.register("field", "equivariant_contextual", "src.resnet_lddmm.fields.equivariant_with_context:EquivariantContextualField")

# Simple equivariant contextual field using MLP aggregation
Registry.register("field", "equivariant_contextual_simple", "src.resnet_lddmm.fields.equivariant_contextual_simple:EquivariantContextualFieldSimple")

# Equivariant template field: velocity parameterized by template geometry
Registry.register("field", "equivariant_template", "src.resnet_lddmm.fields.equivariant_template_field:EquivariantTemplateField")

# STEPS T9: DataTerm + CDData + L2Data + EMDData
Registry.register("data_term", "chamfer", "src.resnet_lddmm.losses.data_terms:CDData")
Registry.register("data_term", "l2", "src.resnet_lddmm.losses.data_terms:L2Data")
Registry.register("data_term", "emd", "src.resnet_lddmm.losses.data_terms:EMDData")

# STEPS T33: WeightedCD / PCD / NCD data terms
Registry.register("data_term", "weighted_chamfer", "src.resnet_lddmm.losses.data_terms:WeightedCDData")
Registry.register("data_term", "pcd", "src.resnet_lddmm.losses.data_terms:PCDData")
Registry.register("data_term", "ncd", "src.resnet_lddmm.losses.data_terms:NCDData")

# STEPS T34: SinkhornData (lazy-loaded geomloss)
Registry.register("data_term", "sinkhorn", "src.resnet_lddmm.losses.data_terms:SinkhornData")

# STEPS T10: ShapeCode + NoCode
Registry.register("code", "none", "src.resnet_lddmm.codes.none:NoCode")

# STEPS T27: AutoDecoderCodes (per-shape latent codes)
Registry.register("code", "auto_decoder", "src.resnet_lddmm.codes.auto_decoder:AutoDecoderCodes")

# STEPS T31: EncoderCodes (amortised codes via equivariant encoder)
Registry.register("code", "encoder", "src.resnet_lddmm.codes.encoder:EncoderCodes")

# IsometryLoss (shape-preserving regularization)
Registry.register("iso_loss", "isometry", "src.resnet_lddmm.losses.data_terms:IsometryLoss")

# Config-selected loss terms (loss.terms in the YAML). Each takes a LossContext
# and returns a scalar or None; see losses/terms.py.
Registry.register("loss_term", "equivariant_deformation_loss", "src.resnet_lddmm.losses.terms:EquivariantDeformationLoss")
Registry.register("loss_term", "pose_supervision_loss", "src.resnet_lddmm.losses.terms:PoseSupervisionLoss")

# Augmentation: random group transformations (SE(3), SO(3), or none)
Registry.register("augmentation", "none", "src.resnet_lddmm.augmentation.none:NoAugmentation")
Registry.register("augmentation", "so3", "src.resnet_lddmm.augmentation.so3:SO3Augmentation")
Registry.register("augmentation", "se3", "src.resnet_lddmm.augmentation.se3:SE3Augmentation")
