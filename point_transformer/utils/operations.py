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
    # TODO: use trilinear interpolation
    B, N, C = x.shape[0], x.shape[1], x.shape[2]
    batch_idx = torch.arange(0, B)[:, None].expand(-1, N)  # B,N

    for _ in range(factor // 2):
        idx1 = torch.randint(0, N, (B, N))  # B,N
        idx2 = torch.randint(0, N, (B, N))  # B,N

        feats1 = x[batch_idx.flatten(), idx1.flatten()].reshape(B, N, C)
        feats2 = x[batch_idx.flatten(), idx2.flatten()].reshape(B, N, C)
        feats_avg = (feats1 + feats2) / 2
        x = torch.cat((x, feats_avg), 1)
    return x  # B,N*factor,C
