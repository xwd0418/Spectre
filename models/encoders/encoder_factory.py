from encoder import CoordinateEncoder, SignCoordinateEncoder
from models.encoders.graycode_encoder import GraycodeEncoder

def build_encoder(coord_enc: str, dim_model: int, dim_coords, wavelength_bounds=None, gce_resolution=None):
  dim_coords = tuple([int(v) for v in dim_coords.split(",")])
  assert (sum(dim_coords) == dim_model)
  if coord_enc == "ce":
    return CoordinateEncoder(dim_model, dim_coords, wavelength_bounds)
  elif coord_enc == "sce":  # when using sce, you use 1 less wavelength bound
    return SignCoordinateEncoder(dim_model, dim_coords, wavelength_bounds)
  elif coord_enc == "gce":
    return GraycodeEncoder(dim_model, dim_coords, gce_resolution)
  else:
    raise NotImplementedError(f"Encoder type {coord_enc} not implemented")
