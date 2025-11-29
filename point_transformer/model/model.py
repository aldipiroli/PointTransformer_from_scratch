import torch
import torch.nn as nn

from point_transformer.utils.operations import find_kNN


class ProjectionBlock(nn.Module):
    def __init__(self, dm, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dm, d), nn.ReLU(), nn.Linear(d, d))

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
    def __init__(self, dm, d, k):
        super().__init__()
        self.k = k
        self.lin1 = ProjectionBlock(dm, d)
        self.point_transformer = PointTransformer(d)
        self.lin2 = ProjectionBlock(d, d)

    def forward(self, x, p):
        x_lin1 = self.lin1(x)

        idx = find_kNN(x, self.k)
        batch_ids = torch.arange(x.shape[0])[:, None, None]  # B,1,1
        xn_lin1 = x_lin1[batch_ids, idx]  # B,N,K,C
        pn = p[batch_ids, idx]  # B,N,K,C
        x_pt = self.point_transformer(x_lin1, xn_lin1, p, pn)

        x_lin2 = self.lin2(x_pt)
        x = x_lin2 + x
        return x  # B,N,C
