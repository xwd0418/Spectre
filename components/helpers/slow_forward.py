import os, re, torch

from pathlib import Path
from models.ranked_transformer import HsqcRankedTransformer
from utils import ranker

# does a very slow forward pass

def fwomp(single_hsqc):
  """
    Does a hard-coded slow forward pass

    single_hsqc: 2d list (n, 3)

    return: fp list
  """
  folder = Path("/data/smart4.5/new_split")
  experiment = Path("lr_1e-5")
  chkpts = list(f for f in os.listdir(folder / experiment / "checkpoints") if re.search("epoch", f))
  chkpt = chkpts[-1] if len(chkpts) else None
  full_path = os.path.join(folder / experiment / "checkpoints" / chkpt) if chkpt is not None else None

  model = HsqcRankedTransformer.load_from_checkpoint(full_path).cuda()
  model.eval()  

  hsqc = torch.unsqueeze(torch.tensor(single_hsqc), dim = 0).cuda()
  out = model(hsqc)
  out_fp = torch.where(out > 0, 1, 0)

  return out_fp.cpu().tolist()

def fwomp_rank(fp_list):
  """
    Does a slow-rank

    fp_list: single list (6144)

    return: fp list
  """
  ranking_path = "./tempdata/SMILES_ranking_sets/val/rankingset.pt"
  rs = ranker.RankingSet(file_path = ranking_path)
  t_10 = rs.retrieve_idx(torch.tensor(fp_list, dtype=torch.float32).cuda())
  return t_10.cpu().tolist()