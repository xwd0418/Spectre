import torch, torch.nn.functional as F, pickle
from sklearn.preprocessing import normalize

class RankingSet():
    def __init__(self, store=None, file_path=None, retrieve_path=None,debug=False, device = "cuda"):
        '''
            Creates a ranking set. Assumes the file specified at file_path is a pickle file of 
            a numpy array of fingerprints.

            self.data should be (n, 6144)
        '''
        self.debug=debug
        self.device = device
        if store is not None:
            self.data = F.normalize(store, dim=1, p=2.0).to(device)
        else:
            with open(file_path, "rb") as f:
                self.data = F.normalize(torch.load(f).type(torch.FloatTensor), dim=1, p=2.0).to(device)
        
        if retrieve_path:
            with open(retrieve_path, "rb") as f:
                self.retrieve = pickle.load(f)
            
    @staticmethod
    def round(fp):
        hi = torch.max(fp)
        where = torch.where(fp == hi)
        fp = torch.zeros(6144)
        fp[where] = 1
        return fp

    @staticmethod
    def normalized_to_nonzero(fp):
        hi = torch.max(fp)
        return torch.nonzero(torch.isclose(fp == hi))[:,0]

    def retrieve(self, query, n=10):
        if not self.retrive:
            raise Exception("No retrieval dict")
        query = F.normalize(query, dim=0, p=2.0).to(self.device).unsqueeze(0)
        query_products = self.data @ query.T # (n x 6144) * (6144 x 1) = n x 1

        fps, _ = torch.topk(query_products, k=n)
        out = []
        for fp in fps:
            nonzero = self.normalized_to_nonzero(fp)
            out.append(self.retrieve.get(nonzero, None))
        return out

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
            queries = F.normalize(queries, dim=1, p=2.0).to(self.device)
            truths = F.normalize(truths, dim=1, p=2.0).to(self.device)

            thresh = torch.stack([torch.dot(queries[i,:], truths[i,:]) for i in range(q)]).to(self.device) # (q)

            query_products = self.data @ queries.T # (n x 6144) * (6144 x q) = (n, q)
            truth_products = self.data @ truths.T # (n, q)

            # we want to subtract off any samples in which the truth
            ct_ones = torch.sum(torch.isclose(truth_products, torch.ones(n, q).to(self.device)), dim = 0) # (q)
            ct_greater = torch.sum(query_products > thresh, dim = 0) # ((n, q) >= (q)) -> (q)
            return ct_greater - ct_ones
