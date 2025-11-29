from sklearn.neighbors import NearestNeighbors


def find_kNN(p, K):
    nbrs = NearestNeighbors(n_neighbors=K, algorithm="ball_tree").fit(p)
    distances, indices = nbrs.kneighbors(p)
    return indices  # NxK
