import torch
import numpy as np

class PositionalEncoder(torch.nn.Module):
    """Encode positions using sine and cosine waves.
    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use. 
    max_wavelength : float
        The maximum wavelength to use.
    """
    def __init__(
            self,
            dim_model=128,
            min_wavelength=.001,
            max_wavelength=10000
        ):
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)
    
        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)
    
    def forward(self, X):
        """Encode positions
        Parameters
        ----------
        X : torch.Tensor of shape (n_positions)
            The positions to encode
        Returns
        -------
        torch.Tensor of shape (n_positions, dim_model)
            The encoded positions
        """
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)

class CoordinateEncoder(torch.nn.Module):
    """
    Generate positional encoding of coordinates

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality used by the Transformer model.
    dim_coords: tuple or None
        A tuple specifying the number of features to use for encoding 
        each coordinate. Must sum to dim model
    wavelength_bounds : list(tuple), optional
        A list of tuples of (minimum, maximum) wavelengths for
        each dimension to be encoded 
    """
    def __init__(
            self,
            dim_model,
            dim_coords,
            wavelength_bounds=None,
        ):
        super().__init__()
        assert(sum(dim_coords) == dim_model)
        if wavelength_bounds:
            assert(len(wavelength_bounds) == len(dim_coords))

        self.positional_encoders = []
        self.dim_coords = dim_coords
        for idx, dim in enumerate(dim_coords):
            if wavelength_bounds:
                min_wavelength = wavelength_bounds[idx][0]
                max_wavelength = wavelength_bounds[idx][1]
                p = PositionalEncoder(dim_model=dim,
                                    min_wavelength=min_wavelength,
                                    max_wavelength=max_wavelength)
            else:
                p = PositionalEncoder(dim_model=dim)
            self.positional_encoders.append(p)

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
        assert(X.shape[2] == len(self.dim_coords))
        embeddings = []
        for dim, encoder in enumerate(self.positional_encoders):
            embeddings.append(encoder(X[:, :, [dim]]))
        
        return torch.cat(embeddings, dim=2)