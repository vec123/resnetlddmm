
from vtk.util import numpy_support
import torch
import numpy as np
import vtk

def add_point_field(polydata, field_data, field_name="field", verbose = False):
    """Adds a scalar or vector field to the existing PolyData."""
    
    if verbose:
        print(f"adding field {field_data.shape}")
    if isinstance(field_data, torch.Tensor):
        field_data = field_data.detach().cpu().numpy()
        if verbose:
            print(f"convert to {field_data.shape}")

    # Ensure numpy array and correct shape
    field_data = np.asarray(field_data)
    if field_data.ndim == 1:
        vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
        vtk_array.SetNumberOfComponents(1)
    elif field_data.ndim == 2:
        vtk_array = numpy_support.numpy_to_vtk(field_data, deep=True)
        vtk_array.SetNumberOfComponents(field_data.shape[1])
    else:
        raise ValueError(f"Unsupported field_data ndim={field_data.ndim}; expected 1 or 2.")

    vtk_array.SetName(field_name)

    # Remove any existing array with the same name so we overwrite instead of duplicate.
    existing = polydata.GetPointData().GetArray(field_name)
    if existing is not None:
        polydata.GetPointData().RemoveArray(field_name)

    polydata.GetPointData().AddArray(vtk_array)

    if vtk_array.GetNumberOfComponents() == 3:
        polydata.GetPointData().SetActiveVectors(field_name)
    elif vtk_array.GetNumberOfComponents() == 1:
        polydata.GetPointData().SetActiveScalars(field_name)

    return polydata

