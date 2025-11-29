import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from point_transformer.model.model import PointTransformer, PointTransformerBlock


def test_point_transformer():
    B = 2
    N = 64
    C = 4
    K = 8

    x = torch.randn(B, N, C)
    x_n = torch.randn(B, N, K, C)
    p = torch.randn(B, N, 3)
    p_n = torch.randn(B, N, K, 3)

    vect_att = PointTransformer(dm=C, d=C)
    out = vect_att(x, x_n, p, p_n)
    assert out.shape == (B, N, C)
    run_training_step(vect_att, out, x)


def test_point_transformer_block():
    B = 2
    N = 64
    C = 4
    K = 8

    x = torch.randn(B, N, C)
    p = torch.randn(B, N, 3)

    pt_block = PointTransformerBlock(dm=C, d=C, k=K)
    out = pt_block(x, p)
    assert out.shape == x.shape
    run_training_step(pt_block, out, x)


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
    test_point_transformer_block()
    print("All tests passed!")
