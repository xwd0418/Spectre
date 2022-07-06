import numpy as np, pickle
from sklearn.preprocessing import normalize

class RankingSet():
    def __init__(self, file_path):
        '''
            Creates a ranking set. Assumes the file specified at file_path is a pickle file of 
            a numpy array of fingerprints.

            self.data should be (n, 6144)
        '''
        self.file_path = file_path
        with open(file_path, "rb") as f:
            self.data = normalize(pickle.load(f), axis=1, norm="l2")

    def batched_rank(queries, truths):
        '''
            Perform a batched ranking

            query(predictions) should be (q, 6144)
            truths should be (q, 6144)

            returns the number of items that have a higher dot product similarity (query_i (dot) truth_i)
        '''
        q = queries.shape[0]
        queries = normalize(queries, axis=1, norm="l2")
        truths = normalize(truths, axis=1, norm="l2")

        thresh = np.array([np.dot(queries[i,:], truths[i,:]) for i in range(q)]) # (q)

        query_products = self.data @ query.T # (n, q)

        outs = np.zeros(q)
        for i in range(q):
            count_1s = len(np.nonzero(query_products[:,i] == 1.0)[0])
            greater = len(np.nonzero(query_products[:,i] > thresh[i]))
            outs[i] = greater - count_1s
        return outs
