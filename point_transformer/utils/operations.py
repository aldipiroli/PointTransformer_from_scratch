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


def point_interpolation(x, n):
    B, N = x.shape[0], x.shape[1]
    idx1 = torch.randint(0, N, (B, N))
    idx2 = torch.randint(0, N, (B, N))
    return
