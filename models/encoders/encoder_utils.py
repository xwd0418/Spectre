import torch, graycode

def do_graycode(x, bufsize, resolution):
  buf = torch.zeros(bufsize, dtype=torch.int)
  x_adj = torch.tensor(x / resolution).int()
  bits = 1 if abs(x) < 1e-8 else int(torch.log2(x_adj).floor()) + 1
  mask = 2**torch.arange(bits).int()
  bit_res = x_adj.bitwise_and(mask).ne(0)
  buf[:bits] = bit_res
  return buf