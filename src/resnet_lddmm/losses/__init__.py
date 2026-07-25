"""Loss term modules for registration."""

from src.resnet_lddmm.losses.data_terms import DataTerm, CDData, L2Data
from src.resnet_lddmm.losses.mapping_error import (
    UnidirectionalMappingError,
    BidirectionalMappingError,
)

__all__ = [
    "DataTerm",
    "CDData",
    "L2Data",
    "UnidirectionalMappingError",
    "BidirectionalMappingError",
]
