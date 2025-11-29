import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from point_transformer.utils.operations import find_kNN, find_kNN_sklearn


def test_find_kNN():
    B = 2
    N = 64
    K = 8
    x = torch.randn(B, N, 3)
    indicies = find_kNN(x, x, k=K)
    assert indicies.shape == (B, N, K)


def test_kNN_methods():
    B = 2
    N = 64
    K = 8
    x = torch.randn(B, N, 3)
    indicies_torch = find_kNN(x, x, k=K)

    indicies_sklearn = []
    for i in range(B):
        idx = find_kNN_sklearn(x[i], K)
        indicies_sklearn.append(idx)
    indicies_sklearn = torch.tensor(np.array(indicies_sklearn))
    assert torch.equal(indicies_sklearn, indicies_torch)


if __name__ == "__main__":
    test_kNN_methods()
    print("All tests passed!")
