import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from point_transformer.model.model import VectorAttention


def test_vector_attention():
    B = 2
    N = 64
    C = 4
    K = 8

    x = torch.randn(B, N, C)
    x_n = torch.randn(B, N, K, C)
    p = torch.randn(B, N, 3)
    p_n = torch.randn(B, N, K, 3)

    vect_att = VectorAttention(dm=C, d=C)
    out = vect_att(x, x_n, p, p_n)
    assert out.shape == (B, N, C)
    run_training_step(vect_att, out, x)


def run_training_step(model, preds, y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    loss = torch.nn.functional.mse_loss(preds, y)
    assert loss == loss
    loss.backward()
    optimizer.step()
    for p in model.parameters():
        assert p.grad is not None


if __name__ == "__main__":
    test_vector_attention()
    print("All tests passed!")
