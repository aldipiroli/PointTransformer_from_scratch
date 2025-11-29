import torch
from sklearn.neighbors import NearestNeighbors


def find_kNN_sklearn(x, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(x)
    distances, indices = nbrs.kneighbors(x)
    return indices  # NxK


def find_kNN(x, k):
    dist = torch.cdist(x, x)
    distances, indicies = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    return indicies
