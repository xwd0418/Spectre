BAD = "[](),"

def parse_hsqc(inp_str: str):
  """
    Parse an input string into a 2d list, n x 3
  """
  # remove bad strings
  for c in BAD:
    inp_str = inp_str.replace(c, "")
  # split into rows
  splits = [q.strip() for q in inp_str.split("\n")]
  # remove empty ones
  filtered_splits = [q for q in splits if not q.isspace()]
  # split rows into subarrays
  sub_arrays = [q.split() for q in filtered_splits]
  
  if not all([len(q) == 3 for q in sub_arrays]):
    raise Exception("Not all rows are 3-tuples")
  try:
    float_hsqcs = [[float(v) for v in q] for q in sub_arrays]
    return float_hsqcs
  except:
    raise Exception("Could not cast all 3-tuples to floats")
  