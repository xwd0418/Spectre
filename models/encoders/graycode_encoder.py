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
    assert (resolution is not None)
    assert (not np.isclose(resolution, 0))
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
    
    # Stack together many (b_s, n_c, d) tensors
    encodings = []
    device = X.device
    for dim, dim_size in enumerate(self.dim_coords):
      raw_values = X[:,:,[dim]].divide(self.resolution).round().type(torch.int32)
      b_s, n_c, _ = raw_values.size()
      
      # Encoded Sign
      encodings.append(torch.where(raw_values > 0, 1, 0))
      raw_values = torch.abs(raw_values)
      # Element-Wise Graycode
      for b in range(b_s):
        for c in range(n_c):
          raw_values[b,c,0] = graycode.tc_to_gray_code(abs(raw_values[b,c,0]))
      
      mask = 2**torch.arange(dim_size - 1)[None, None, :].to(device)
      # broadcast [b_s, n_c, 1] with [1,1,dim_size]
      encoded_values = raw_values.bitwise_and(mask).ne(0).type(torch.int32)
      
      # Encoded Values
      encodings.append(encoded_values)

    final = torch.cat(encodings, dim=2)
    return final
