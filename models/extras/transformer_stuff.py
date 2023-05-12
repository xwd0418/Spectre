import torch

def generate_square_subsequent_mask(sz, device="cpu"):
  """ 
  Method copied from Pytorch nn.Transformer.
  Generate a square mask for the sequence. The masked positions are filled with float('-inf').
  Unmasked positions are filled with float(0.0).
  Essentially creaes a Lower-Triangular Matrix of Zeros (including diagonal), where upper triangle (excluding diagonal) is -inf

  Args:
      sz (int): Size of mask to generate

  Returns:
      torch.Tensor: Square autoregressive mask for decode 

  """
  # Upper triangular binary mask => Lower triangular binary mask
  mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
  # replace True (Lower Triangle) with 0, False (Upper excluding diag) with -inf
  mask = mask.float().masked_fill(mask == 0, float('-inf')
                                  ).masked_fill(mask == 1, float(0.0))
  return mask

def build_padding_mask(inp):
  """
  Builds padding mask from input. Assumes (b_s, seq_len, embedding_dim)

  """
  # sum along data dimension,
  zeros = ~inp.sum(dim=2).bool()  # (b_s, seq_len)
  mask = [
      torch.tensor([[False]] * inp.shape[0]).type_as(zeros),
      zeros,
  ]
  mask = torch.cat(mask, dim=1)
  return mask

def positional_embs(d_model, max_seq_len):
    """ Produces a tensor of positional embeddings for the model

    Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
    which are created from sine and cosine waves of varying wavelength
    """

    encs = torch.tensor(
        [dim / d_model for dim in range(0, d_model, 2)])
    encs = 10000 ** encs
    encs = [(torch.sin(pos / encs), torch.cos(pos / encs))
            for pos in range(max_seq_len)]
    encs = [torch.stack(enc, dim=1).flatten()[:d_model] for enc in encs]
    encs = torch.stack(encs)
    return encs