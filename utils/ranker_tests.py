import torch
from ranker import RankingSet

data_store = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
], dtype = torch.float)
sample_label = torch.tensor([
    [1, 0, 1, 1],
], dtype = torch.float).cuda()
sample_query = torch.tensor([
    [1, 0, 0, 0],
], dtype = torch.float).cuda()
rs = RankingSet(store = data_store)
v = rs.batched_rank(sample_query, sample_label)
print(v)

# samples = torch.rand((255, 6144)).cuda()
# samples2 = torch.rand((255, 6144)).cuda()

# v = rs.batched_rank(samples, samples2)
