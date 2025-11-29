import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from point_transformer.model.model import (
    PointTransformer,
    PointTransformerBlock,
    TransitionDownModule,
    TransitionUpModule,
)
from point_transformer.utils.operations import find_kNN


def test_point_transformer():
    B = 2
    N = 64
    C = 4
    K = 8

    x = torch.randn(B, N, C)
    x_n = torch.randn(B, N, K, C)
    p = torch.randn(B, N, 3)
    p_n = torch.randn(B, N, K, 3)

    vect_att = PointTransformer(d=C)
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
    _, idx = find_kNN(x, x, K)

    pt_block = PointTransformerBlock(d=C, k=K)
    out = pt_block(x, p, idx)
    assert out.shape == x.shape
    run_training_step(pt_block, out, x)


def test_transition_down_module():
    B = 2
    N = 64
    N2 = 32 // 2
    C = 4
    K = 8

    x = torch.randn(B, N, C)
    p = torch.randn(B, N, 3)
    tr_down = TransitionDownModule(K, C)
    out = tr_down(x, p, N2)
    assert out.shape == (B, N2, C)
    x2 = torch.randn(B, N2, C)
    run_training_step(tr_down, out, x2)


def test_transition_up_module():
    B = 2
    N = 64
    N2 = N // 2
    C = 4
    K = 8

    x = torch.randn(B, N2, C)
    p = torch.randn(B, N2, 3)
    tr_up = TransitionUpModule(K, C)
    factor = 2
    out = tr_up(x, p, factor)
    assert out.shape == (B, N, C)
    x2 = torch.randn(B, N, C)
    run_training_step(tr_up, out, x2)


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
    test_transition_up_module()
    print("All tests passed!")
