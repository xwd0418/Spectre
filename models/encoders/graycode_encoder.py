import logging
import torch
import graycode
import numpy as np

class GraycodeEncoder(torch.nn.Module):
  """
  Generate graycode encoding

  Parameters
  ----------
  dim_model : int, optional
      The latent dimensionality used by the Transformer model.
  dim_coords: tuple or None
      A tuple specifying the number of features to use for encoding 
      each coordinate. Must sum to dim model
  resolution: highest resolution of graycode encoding
  """

  def __init__(
      self,
      dim_model,
      dim_coords,
      resolution=1
  ):
    super().__init__()
    assert (sum(dim_coords) == dim_model)
    self.positional_encoders = []
    self.dim_coords = dim_coords
    assert (not torch.isclose(resolution, 0))
    self.resolution = resolution

  def forward(self, X):
    """Encode coordinates
    Parameters
    ----------
    X : torch.Tensor of shape (batch_size, n_coords, n_dimensions)
        The coordinates to embed
    Returns
    -------
    torch.Tensor of shape (batch_size, n_coords, dim_model)
        The encoded coordinates
    """
    assert (X.shape[2] == len(self.dim_coords))
    embeddings = []
    for dim, encoder in enumerate(self.positional_encoders):
      values = X[:,:,[dim]]
      b_s, n_c, _ = values.size()
      bufsize = torch.zeros((b_s, n_c, self.dim_coords[dim]))
      mask = 2**torch.arange(self.dim_coords[dim])[None, None, :]
      embeddings.append(encoder(X[:, :, [dim]]))

    return torch.cat(embeddings, dim=2)
