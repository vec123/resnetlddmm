import warnings
from dataclasses import dataclass, field, replace
from typing import Optional, List, Tuple, Union
RangeOrFixed = Union[float, Tuple[float, float]]

@dataclass
class DataConfig:
    """Dataset loading + graph-construction policy.
    Graph-construction knobs live on ``GraphSpec``
    """
    data_path: str = "DATA_ROOT"
    # names of subdirectories
    parts: Optional[List[str]] = field(default_factory=lambda: ["mouth", "nose"])
    load_fields: bool = True
    val_fraction: float = 0.2
    shuffle: bool = True
    seed: Optional[int] = None

    graph_builder: str = "radius"   # registry key: "radius" | "knn" (T4, T6)
    graph_spec: GraphSpec = field(default_factory=GraphSpec)
    resample_graph: bo