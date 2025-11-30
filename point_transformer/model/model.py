import torch
import torch.nn as nn

from point_transformer.utils.operations import feature_interpolation, find_kNN, sample_down


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

    def forward(self, x, p):
        _, idx = find_kNN(x, x, self.k)
        x_lin1 = self.lin1(x)

        batch_ids = torch.arange(x.shape[0])[:, None, None]  # B,1,1
        xn_lin1 = x_lin1[batch_ids, idx]  # B,N,K,C
        pn = p[batch_ids, idx]  # B,N,K,C
        x_pt = self.point_transformer(x_lin1, xn_lin1, p, pn)

        x_lin2 = self.lin2(x_pt)
        x = x_lin2 + x
        return x  # B,N,C


class TransitionDownModule(nn.Module):
    def __init__(self, k, din, dout):
        super().__init__()
        self.k = k
        self.din = din
        self.dout = dout
        self.projection = nn.Sequential(nn.Linear(din, dout), nn.BatchNorm1d(dout), nn.ReLU(), nn.Linear(dout, dout))

    def forward(self, x, p, n2):
        n2_idx = sample_down(p, n2)
        _, idx = find_kNN(x[:, n2_idx], x, self.k)
        batch_ids = torch.arange(x.shape[0])[:, None, None]  # B,1,1
        x2 = x[batch_ids, idx]  # B,N2,K,C
        p2 = p[:, n2_idx]  # B,N2,C
        B, N2, K, C = x2.shape
        x2 = self.projection(x2.reshape(B * N2 * K, self.din))
        x2 = x2.reshape(B, N2, K, self.dout)  # B,N2,K,C
        x2 = torch.max(x2, dim=2)[0]  # B,N2,C
        return x2, p2


class TransitionUpModule(nn.Module):
    def __init__(self, k, din, dout):
        super().__init__()
        self.k = k
        self.din = din
        self.dout = dout
        self.projection = nn.Sequential(nn.Linear(din, dout), nn.BatchNorm1d(dout), nn.ReLU(), nn.Linear(dout, dout))

    def forward(self, x, p, factor):
        B, N2, C = x.shape
        x = self.projection(x.reshape(B * N2, C))
        x = x.reshape(B, N2, self.dout)
        x2 = feature_interpolation(x, factor)
        p2 = feature_interpolation(p, factor)
        return x2, p2


class PointTransformerSemanticSegmentation(nn.Module):
    def __init__(self, d_in, d, k, n_classes):
        super().__init__()
        self.k = k

        # Input
        self.linear_in = nn.Linear(d_in, d)
        self.pt_in = PointTransformerBlock(d, k)

        # Transition Down
        self.tr_down1 = TransitionDownModule(k, din=d, dout=d * 2)
        self.pt_down1 = PointTransformerBlock(d * 2, k)

        self.tr_down2 = TransitionDownModule(k, din=d * 2, dout=d * 4)
        self.pt_down2 = PointTransformerBlock(d * 4, k)

        self.tr_down3 = TransitionDownModule(k, din=d * 4, dout=d * 8)
        self.pt_down3 = PointTransformerBlock(d * 8, k)

        self.tr_down4 = TransitionDownModule(k, din=d * 8, dout=d * 16)
        self.pt_down4 = PointTransformerBlock(d * 16, k)

        # Mid
        self.linear_mid = nn.Linear(d * 16, d * 16)

        # Transition Up
        self.tr_up4 = TransitionUpModule(k, din=d * 16, dout=d * 8)
        self.pt_up4 = PointTransformerBlock(d * 8, k)

        self.tr_up3 = TransitionUpModule(k, din=d * 8, dout=d * 4)
        self.pt_up3 = PointTransformerBlock(d * 4, k)

        self.tr_up2 = TransitionUpModule(k, din=d * 4, dout=d * 2)
        self.pt_up2 = PointTransformerBlock(d * 2, k)

        self.tr_up1 = TransitionUpModule(k, din=d * 2, dout=d)
        self.pt_up1 = PointTransformerBlock(d, k)

        # Cls head
        self.linear_cls = nn.Linear(d, n_classes)

    def forward(self, x, p):
        N = x.shape[1]
        xin = self.linear_in(x)
        xin = self.pt_in(xin, p)

        # Down Path
        xdown1, pdown1 = self.tr_down1(xin, p, N // 4)
        xdown1 = self.pt_down1(xdown1, pdown1)

        xdown2, pdown2 = self.tr_down2(xdown1, pdown1, N // 16)
        xdown2 = self.pt_down2(xdown2, pdown2)

        xdown3, pdown3 = self.tr_down3(xdown2, pdown2, N // 64)
        xdown3 = self.pt_down3(xdown3, pdown3)

        xdown4, pdown4 = self.tr_down4(xdown3, pdown3, N // 256)
        xdown4 = self.pt_down4(xdown4, pdown4)

        # Mid
        xmid = self.linear_mid(xdown4)

        # Up Path
        xup4, pup4 = self.tr_up4(xmid, pdown4, factor=6)
        xup4 += xdown3
        xup4 = self.pt_up4(xup4, pup4)

        xup3, pup3 = self.tr_up3(xup4, pup4, factor=6)
        xup3 += xdown2
        xup3 = self.pt_up3(xup3, pup3)

        xup2, pup2 = self.tr_up2(xup3, pup3, factor=6)
        xup2 += xdown1
        xup2 = self.pt_up2(xup2, pup2)

        xup1, pup1 = self.tr_up1(xup2, pup2, factor=6)
        xup1 += xin
        xup1 = self.pt_up1(xup1, pup1)

        # Cls Head
        out = self.linear_cls(xup1)
        return out
