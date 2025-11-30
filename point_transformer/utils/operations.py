import torch
from sklearn.neighbors import NearestNeighbors


def find_kNN_sklearn(x, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(x)
    distances, indices = nbrs.kneighbors(x)
    return indices  # NxK


def find_kNN(x1, x2, k):
    dist = torch.cdist(x1, x2)
    distances, indicies = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    return distances, indicies


def sample_down(x, n, axis=1):
    # TODO: use farthest-point sampling
    idx = torch.randperm(x.shape[axis])[:n]
    return idx


def feature_interpolation(x, factor):
    # TODO: use interpolation not just duplication
    B, N, C = x.shape
    x = x.unsqueeze(2).expand(B, N, factor, C)
    x = x.reshape(B, N * factor, C)
    return x
