import torch
import torch.nn as nn

from point_transformer.utils.operations import find_kNN, sample_down


class ProjectionBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(din, dout), nn.ReLU(), nn.Linear(dout, dout))

    def forward(self, x):
        out = self.proj(x)
        return out


class PointTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = ProjectionBlock(d, d)
        self.k = ProjectionBlock(d, d)
        self.v = ProjectionBlock(d, d)
        self.pos_enc_tr = ProjectionBlock(3, d)

    def forward(self, x, xn, p, pn):
        Q = self.q(x).unsqueeze(2)  # B,N,1,C
        K = self.k(xn)  # B,N,K,C
        V = self.v(xn)  # B,N,K,C

        pos_enc = p.unsqueeze(2) - pn  # B,N,K,3
        pos_enc = self.pos_enc_tr(pos_enc)  # B,N,K,C

        attn = (Q - K) + pos_enc  # B,N,K,C
        attn = nn.functional.softmax(attn, 2)

        out = attn * (V + pos_enc)
        out = out.sum(2)  # B,N,C
        return out


class PointTransformerBlock(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.k = k
        self.lin1 = ProjectionBlock(d, d)
        self.point_transformer = PointTransformer(d)
        self.lin2 = ProjectionBlock(d, d)

    def forward(self, x, p, idx):
        x_lin1 = self.lin1(x)

        batch_ids = torch.arange(x.shape[0])[:, None, None]  # B,1,1
        xn_lin1 = x_lin1[batch_ids, idx]  # B,N,K,C
        pn = p[batch_ids, idx]  # B,N,K,C
        x_pt = self.point_transformer(x_lin1, xn_lin1, p, pn)

        x_lin2 = self.lin2(x_pt)
        x = x_lin2 + x
        return x  # B,N,C


class TransitionDownModule(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.k = k
        self.projection = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())

    def forward(self, x, p, n2):
        n2_idx = sample_down(p, n2)
        idx = find_kNN(x[:, n2_idx], x, self.k)
        batch_ids = torch.arange(x.shape[0])[:, None, None]  # B,1,1
        x2 = x[batch_ids, idx]  # B,N,K,C
        B, N2, K, C = x2.shape
        x2 = self.projection(x2.reshape(B * N2 * K, C))
        x2 = x2.reshape(B, N2, K, C)  # B,N,K,C
        x2 = torch.max(x2, dim=2)[0]  # B,N,C
        return x2


class PointTransformerSemanticSegmentation(nn.Module):
    def __init__(self, d_in, d, k):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d)
        self.pt1 = PointTransformer(d, d)
