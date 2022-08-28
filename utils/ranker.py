import torch, torch.nn.functional as F
from sklearn.preprocessing import normalize

class RankingSet():
    def __init__(self, store=None, file_path=None, debug=False):
        '''
            Creates a ranking set. Assumes the file specified at file_path is a pickle file of 
            a numpy array of fingerprints.

            self.data should be (n, 6144)
        '''
        self.debug=debug
        if store is not None:
            self.data = F.normalize(store, dim=1, p=2.0).cuda()
        else:
            with open(file_path, "rb") as f:
                self.data = F.normalize(torch.load(f).type(torch.FloatTensor), dim=1, p=2.0).cuda()
                print(f"Ranking Set size: {self.data.size()}")

    def batched_rank(self, queries, truths):
        '''
            Perform a batched ranking

            query(predictions) should be (q, 6144)
            truths should be (q, 6144)

            returns the number of items that have a higher dot product similarity (query_i (dot) truth_i)
        '''
        with torch.no_grad():
            q = queries.size()[0]
            n = self.data.size()[0]
            queries = F.normalize(queries, dim=1, p=2.0).cuda()
            truths = F.normalize(truths, dim=1, p=2.0).cuda()

            thresh = torch.stack([torch.dot(queries[i,:], truths[i,:]) for i in range(q)]).cuda() # (q)

            query_products = self.data @ queries.T # (n x 6144) * (6144 x q) = (n, q)
            truth_products = self.data @ truths.T # (n, q)

            # we want to subtract off any samples in which the truth
            ct_ones = torch.sum(torch.isclose(truth_products, torch.ones(n, q).cuda()), dim = 0) # (q)
            ct_greater = torch.sum(query_products > thresh, dim = 0) # ((n, q) >= (q)) -> (q)
            return ct_greater - ct_ones
