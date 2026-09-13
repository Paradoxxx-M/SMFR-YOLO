import math

import torch
import torch.nn as nn

def _c2f8_stage_ratio(stage, mode="spf"):
    stage = int(stage)
    if mode == "spf":
        table = {2: 0.75, 4: 0.65, 6: 0.50, 8: 0.40}
    elif mode == "sim":
        table = {2: 1.00, 4: 0.70, 6: 0.35, 8: 0.15}
    else:
        table = {2: 1.00, 4: 1.00, 6: 1.00, 8: 1.00}
    return table.get(stage, 1.00)


class _C2F8Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        p = d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _C2F8Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True, g=1):
        super().__init__()
        self.cv1 = _C2F8Conv(c, c, 3, 1, g=g)
        self.cv2 = _C2F8Conv(c, c, 3, 1, g=g)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


def _c2f22_epoch_1based():

    try:
        e0 = int(globals().get("_G1_CUR_EPOCH", 999))
    except Exception:
        e0 = 999

    if e0 >= 900:
        try:
            import os
            return int(os.environ.get("C2FOPT_EVAL_EPOCH", "0"))
        except Exception:
            return 0

    return e0 + 1


class SMIC2f(nn.Module):
   
    def __init__(self, c1, c2, n=1, shortcut=False, stage=2, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.stage = int(stage)
        self.rho = _c2f8_stage_ratio(stage, "sim")

        self.cv1 = _C2F8Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList(_C2F8Bottleneck(self.c, shortcut, g) for _ in range(n))
        self.cv2 = _C2F8Conv((2 + n) * self.c, c2, 1, 1)

        self.a = nn.Parameter(torch.full((n,), -3.0))
        self.mu_max = 0.15

    def _release_ratio(self):
        if self.stage != 6:
            return 1.0

        e = _c2f22_epoch_1based()
        if e <= 0:
            return 1.0

        if e <= 70:
            return 1.0

        if e <= 100:
            r_min = 0.20
            phase = float(e - 70) / 30.0
            return r_min + (1.0 - r_min) * (1.0 + math.cos(math.pi * phase)) / 2.0

        return 0.20

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        f0 = y[1]

        rel = self._release_ratio()

        for i, block in enumerate(self.m):
            f_hat = block(y[-1])
            mu = self.rho * self.mu_max * torch.sigmoid(self.a[i])
            mu = mu * rel
            f = (1.0 - mu) * f_hat + mu * f0
            y.append(f)

        return self.cv2(torch.cat(y, dim=1))

