
from vtk.util import numpy_support


def extract_vtp_points_cells(poly_data):

    points = numpy_support.vtk_to_numpy(poly_data.GetPoints().GetData())
    cells = numpy_support.vtk_to_numpy(poly_data.GetPolys().GetConnectivityArray())
    return points, cells.reshape(-1, 3)


def extract_vtp_point_fields(poly_data, field_names):
    """Read named point-data arrays from a vtkPolyData as numpy arrays.

    Returns a dict mapping each requested name to its array -- shape ``[N]`` for a scalar
    field, ``[N, C]`` for a multi-component field (e.g. ``[N, 3]`` normals) -- or ``None``
    when the field is absent. Missing fields are expected (bare point clouds carry no
    area/normal), so callers must handle ``None`` (see ``load_dataset``'s ones/zeros
    fallback)."""
    point_data = poly_data.GetPointData()
    result = {}
    for name in field_names:
        arr = point_data.GetArray(name) if point_data is not None else None
        result[name] = numpy_support.vtk_to_numpy(arr) if arr is not None else None
    return result