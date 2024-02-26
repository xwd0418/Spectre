import torch
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import normalize
from utils.matmul_precision_wrapper import set_float32_highest_precision

import logging


@set_float32_highest_precision
class RankingSet(torch.nn.Module):
  def __init__(self, store=None, file_path=None, retrieve_path=None, idf_weights=None, debug=False, batch_size = 0):
    '''
      Creates a ranking set. Assumes the file specified at file_path is a pickle file of 
      a numpy array of fingerprints. Fingerprints should be (n, 6144) dimension, and on load,
      the fingerprints will be normalized horizontally. 

      store: whether to use an existing tensor (pass in), or None to signify loading from file

      file_path: file to load, if `store` is None

      idf_weights: optionally choose to perform a weighted ranking

      debug: enable/disable verbosity for debug statements
    '''
    
    super().__init__()
    self.logger = logging.getLogger("lightning")
    self.logger.setLevel(logging.DEBUG)

    self.debug = debug
    self.idf = None
    self.batch_size = batch_size
    self.logger.debug("[Ranker] Initializing Ranker")

    with torch.no_grad():
      if store is not None:
        self.register_buffer("data", F.normalize(store, dim=1, p=2.0), persistent=False)
      else:
        with open(file_path, "rb") as f:
          self.register_buffer("data", F.normalize(torch.load(f).type(
              torch.FloatTensor), dim=1, p=2.0), persistent=False)

      if retrieve_path:
        with open(retrieve_path, "rb") as f:
          self.lookup = pickle.load(f)

      if idf_weights is not None:
        self.register_buffer("idf", idf_weights, persistent=False)
        self.register_buffer("idf_data", F.normalize(
            self.data * self.idf, dim=1, p=2.0), persistent=False)

    self.logger.info(
        f"[Ranking Set] Initialized with {len(self.data)} sample(s)"
    )

  @staticmethod
  def round(fp):
    '''
      Turns a fingerprint into a binary fingerprint, with 1's in indexes
      that have the max value

      fp: the fingerprint to convert
    '''
    hi = torch.max(fp)
    where = torch.where(fp == hi)
    fp = torch.zeros(6144)
    fp[where] = 1
    return fp

  @staticmethod
  def normalized_to_nonzero(fp):
    hi = torch.max(fp)
    nonzero = torch.nonzero(torch.isclose(fp, hi))
    return tuple(nonzero[:, 0].tolist())

  def retrieve_idx(self, query, n=10):
    query = F.normalize(query, dim=0, p=2.0).to(self.device).unsqueeze(0)
    print(self.data.shape, query.shape)
    # (n x 6144) * (6144 x 1) = n x 1
    query_products = (self.data @ query.T).flatten()
    _, idxs = torch.topk(query_products, k=n)
    return idxs

  def retrieve(self, query, n=10):
    if not self.lookup:
      raise Exception("No retrieval dict")
    query = F.normalize(query, dim=0, p=2.0).to(self.device).unsqueeze(0)
    # (n x 6144) * (6144 x 1) = n x 1
    query_products = (self.data @ query.T).flatten()

    _, idxs = torch.topk(query_products, k=n)
    out = []
    for idx in idxs:
      nonzero = self.normalized_to_nonzero(self.data[idx])
      out.append(self.lookup.get(nonzero, None))
    return out

  def dot_prod_rank(self, data, queries, truths, thresh, query_idx_in_rankingset):
    '''
      Perform a dot-product ranking. Assumes that data, queries, and truths are already
      normalized. 

      queries: (q, 6144) fingerprints to query (assumed normalized)

      truths: (q, 6144) of true fingerprints (assumed normalized)

      thresh: (1, q) what is the cutoff for things that ranked higher
      
      query_idx_in_rankingset: the first index of the same sample in the data tensors

      returns: size (q) tensor of how many were higher, minus how many labels were in the 
        ranking set 
    '''
    assert (queries.size() == truths.size())
    with torch.no_grad():
      q = queries.size()[0]
      n = data.size()[0]
      # (n x 6144) * (6144 x q) = (n, q)

      # For all of these tensors, A_ij is the cosine similarity of sample n to query q
      query_products = data @ queries.T  # (n, q)
      
      rows_range = torch.arange(query_products.size(0)).unsqueeze(1)
      # False if the index in self.data and batch_index represent the same sample
      query_indices_in_rankingset = torch.arange(query_idx_in_rankingset*self.batch_size, query_idx_in_rankingset*self.batch_size + q)
      idx_mask = (query_indices_in_rankingset != rows_range)  
      
      # ((n, q) > (1, q)) -> (n, q) -> (1, q)
      ct_greater = torch.sum(
          (query_products >= thresh) & idx_mask.to(query_products).bool(), dim=0, keepdim=True, dtype=torch.int
      )
      if True:
        
        # self.logger.info("thresh: ")
        # self.logger.info(thresh)
        # self.logger.info("ct_greater: ")
        # self.logger.info(ct_greater)
        # self.logger.info(torch.nonzero(query_products > thresh))
        # self.logger.info(query_products[:5, :5])
        truth_products = data @ truths.T  # (n, q)
        print("thresh: ")
        print(thresh)
        print("ct_greater: ")
        print(ct_greater)
        print('data[0]:\n', torch.nonzero(data[0]))
        # print("idx mask :\n ",torch.nonzero(idx_mask==False))
        rows, cols = zip(*torch.nonzero(idx_mask==False))
        print("supposed to be all ones: ", truth_products[rows,cols])
        # print("masked query products: ",query_products[])s
        # print("mask shape: ",equality_mask[:,0].shape)
        # print("mask shape: ",match_mask[:,0].shape)

        # print("where is larger?:\n",torch.nonzero(query_products > thresh))
        # print(query_products[:5, :5])
        print()
      return ct_greater

  def batched_rank(self, queries, truths, query_idx_in_rankingset):
    '''
        Perform a batched ranking

        queries: (q, 6144) predictions binary fingerprint tensor

        truths: (q, 6144) labels binary tensor

        returns: size (q) tensor of how many things in the ranking set had a higher cosine similarity 
          cos(prediction, sample) than a threshold cos(prediction, label)
    '''
    
    with torch.no_grad():
        #   q = queries.size()[0]
        queries = F.normalize(queries, dim=1, p=2.0)  # (q, 6144)
        truths = F.normalize(truths, dim=1, p=2.0)  # (q, 6144)

        # transpose and element-wise dot ->(6144, q)
        # sum dim 0, keepdims -> (1, q)
        thresh = torch.sum((queries * truths).T, dim=0, keepdim=True)

        return self.dot_prod_rank(self.data, queries, truths, thresh, query_idx_in_rankingset)

  def batched_rank_tfidf(self, queries, truths):
    '''
        Perform a batched ranking with tfidf

        query(predictions) should be (q, 6144)
        truths should be (q, 6144)

        returns the number of items that have a higher dot product similarity (query_i (dot) truth_i)
    '''
    if self.idf_data is None:
      raise Exception("No idf weights")
    with torch.no_grad():
      q = queries.size()[0]
      queries = F.normalize(queries.to(self.device) * self.idf, dim=1, p=2.0)
      truths = F.normalize(truths.to(self.device) * self.idf, dim=1, p=2.0)
      thresh = torch.stack([torch.dot(queries[i, :], truths[i, :])
                           for i in range(q)]).to(self.device)  # (q)

      return self.dot_prod_rank(self.idf_data, queries, truths, thresh)
