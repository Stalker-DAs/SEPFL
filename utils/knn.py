import faiss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch

def build_knn(feats,k):
    # faiss.omp_set_num_threads(threads)

    feats = feats.astype('float32')
    size, dim = feats.shape
    index = faiss.IndexFlatIP(dim)
    index.add(feats)
    sims, nbrs = index.search(feats, k=k)
    knns = [(np.array(nbr, dtype=np.int32),
            1 - np.minimum(np.maximum(np.array(sim, dtype=np.float32), 0), 1))
            for nbr, sim in zip(nbrs, sims)]
    return knns


def row_normalize(mx):
    row_sum = np.array(mx.sum(1))
    # if sum<=0,prevent denominator from being zero
    row_sum[row_sum <= 0] = 1
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_indices_values(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data
    shape = np.array(sparse_mx.shape)
    return indices, values, shape


def knn2spmat(dists,nbrs,k,th_sim,use_sim,self_loop):
    eps = 1e-2
    n = len(nbrs)
    if use_sim:
        sims = 1. - dists
        # sims = dists
    row, col = np.where(sims >= th_sim)
    # remove self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    # make symmetric adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    # div row sum to norm
    adj = row_normalize(adj)
    # sparse_mx2torch_sparse
    indices, values, shape = sparse_mx_to_indices_values(adj)
    indices = torch.from_numpy(indices)
    values = torch.from_numpy(values)
    shape = torch.Size(shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def knn2mat(knns,k,use_sim,self_loop):
    eps = 1e-2
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    if use_sim:
        sims = 1. - dists
    row, col = np.where(sims >= 0)
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    # make symmetric adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    # div row sum to norm
    adj = row_normalize(adj)
    adj = adj.todense().A

    return adj

