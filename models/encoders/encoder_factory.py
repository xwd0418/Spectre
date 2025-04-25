import sys
import pathlib
# Add the current directory to sys.path
sys.path.insert(0, pathlib.Path(__file__).parent.parent.parent.absolute().__str__())

from typing import Tuple
from models.encoder import CoordinateEncoder, SignCoordinateEncoder
from models.encoders.graycode_encoder import GraycodeEncoder

def build_encoder(coord_enc: str, dim_model: int, 
                  dim_coords: Tuple[int, int, int], wavelength_bounds=None, 
                  gce_resolution=None, use_peak_values=False):
  assert (sum(dim_coords) == dim_model)
  if coord_enc == "ce":
    return CoordinateEncoder(dim_model, dim_coords, wavelength_bounds)
  elif coord_enc == "sce":  # when using sce, you use 1 less wavelength bound
    return SignCoordinateEncoder(dim_model, dim_coords, wavelength_bounds, use_peak_values)
  elif coord_enc == "gce":
    return GraycodeEncoder(dim_model, dim_coords, gce_resolution)
  else:
    raise NotImplementedError(f"Encoder type {coord_enc} not implemented")

def build_encoder_from_args(
  coord_enc: str,
  dim_model: int,
  dim_coords: Tuple[int, int, int],
  encoder_args: dict
):
  assert (sum(dim_coords) == dim_model)
  if coord_enc == "ce":
    return CoordinateEncoder(dim_model, dim_coords, encoder_args["wavelength_bounds"])
  elif coord_enc == "sce":  # when using sce, you use 1 less wavelength bound
    return SignCoordinateEncoder(dim_model, dim_coords, encoder_args["wavelength_bounds"])
  elif coord_enc == "gce":
    return GraycodeEncoder(dim_model, dim_coords, encoder_args["gce_resolution"])
  else:
    raise NotImplementedError(f"Encoder type {coord_enc} not implemented")