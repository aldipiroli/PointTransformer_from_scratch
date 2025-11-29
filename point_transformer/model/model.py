import torch.nn as nn


class ProjectionBlock(nn.Module):
    def __init__(self, dm, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dm, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        out = self.proj(x)
        return out


class VectorAttention(nn.Module):
    def __init__(self, dm, d):
        super().__init__()
        self.q = ProjectionBlock(dm, d)
        self.k = ProjectionBlock(dm, d)
        self.v = ProjectionBlock(dm, d)
        self.pos_enc_tr = ProjectionBlock(3, d)

    def forward(self, x, x_n, p, p_n):
        Q = self.q(x).unsqueeze(2)  # B,N,1,C
        K = self.k(x_n)  # B,N,K,C
        V = self.v(x_n)  # B,N,K,C

        pos_enc = p.unsqueeze(2) - p_n  # B,N,K,3
        pos_enc = self.pos_enc_tr(pos_enc)  # B,N,K,C

        attn = (Q - K) + pos_enc  # B,N,K,C
        attn = nn.functional.softmax(attn, 2)

        out = attn * (V + pos_enc)
        out = out.sum(2)  # B,N,C
        return out
