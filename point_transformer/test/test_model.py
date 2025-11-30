import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from point_transformer.model.model import (
    PointTransformer,
    PointTransformerBlock,
    PointTransformerClassification,
    PointTransformerSemanticSegmentation,
    TransitionDownModule,
    TransitionUpModule,
)


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

    pt_block = PointTransformerBlock(d=C, k=K)
    out = pt_block(x, p)
    assert out.shape == x.shape
    run_training_step(pt_block, out, x)


def test_transition_down_module():
    B = 2
    N = 64
    N2 = 32 // 2
    C1 = 4
    C2 = 8
    K = 8

    x = torch.randn(B, N, C1)
    p = torch.randn(B, N, 3)
    tr_down = TransitionDownModule(K, C1, C2)
    xout, pout = tr_down(x, p, N2)
    assert xout.shape == (B, N2, C2)
    assert pout.shape == (B, N2, 3)
    x2 = torch.randn(B, N2, C2)
    run_training_step(tr_down, xout, x2)


def test_transition_up_module():
    B = 2
    N = 64
    N2 = N // 2
    C = 4
    K = 8

    x = torch.randn(B, N2, C)
    p = torch.randn(B, N2, 3)
    tr_up = TransitionUpModule(K, din=C, dout=C)
    factor = 2
    xout, pout = tr_up(x, p, factor)
    assert xout.shape == (B, N, C)
    assert pout.shape == (B, N, 3)
    x2 = torch.randn(B, N, C)
    run_training_step(tr_up, xout, x2)


def test_point_transformer_semantic_segmentation():
    B = 2
    N = 2048
    C = 4
    K = 8
    n_classes = 24

    x = torch.randn(B, N, C)
    p = torch.randn(B, N, 3)

    cfg = {"MODEL": {"d_in": C, "d": 32, "k": K, "n_classes": n_classes}}
    pt_semseg = PointTransformerSemanticSegmentation(cfg)
    out = pt_semseg(x, p)
    assert out.shape == (B, N, n_classes)
    x2 = torch.randn(B, N, n_classes)
    run_training_step(pt_semseg, out, x2)


def test_point_transformer_classification():
    B = 2
    N = 2048
    C = 4
    K = 8
    n_classes = 24

    x = torch.randn(B, N, C)
    p = torch.randn(B, N, 3)
    cfg = {"MODEL": {"d_in": C, "d": 32, "k": K, "n_classes": n_classes}}
    pt_cls = PointTransformerClassification(cfg)
    out = pt_cls(x, p)
    assert out.shape == (B, n_classes)
    x2 = torch.randn(B, n_classes)
    run_training_step(pt_cls, out, x2)


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
    print("All tests passed!")
