import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

class _FusionResizeMixin:
    @staticmethod
    def _resize(x, size):
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")


class _BiFPNFBase(_FusionResizeMixin, nn.Module):
    """Learnable normalized multi-input feature fusion."""

    def __init__(self, c1_list, fusion="bifpn"):
        super().__init__()
        assert fusion in ("weight", "adaptive", "concat", "bifpn")
        self.fusion = fusion
        self.c1_list = [int(c) for c in c1_list]
        if fusion == "bifpn":
            self.fusion_weight = nn.Parameter(torch.ones(len(c1_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif fusion in ("weight", "adaptive"):
            c0 = self.c1_list[0]
            self.fusion_conv = nn.ModuleList([Conv(c, c0, 1) for c in self.c1_list])
            if fusion == "adaptive":
                self.fusion_adaptive = Conv(c0 * len(c1_list), len(c1_list), 1)

    def forward(self, xs):
        size = xs[0].shape[-2:]
        xs = [self._resize(x, size) for x in xs]
        if self.fusion == "concat":
            return torch.cat(xs, dim=1)
        if self.fusion in ("weight", "adaptive"):
            xs = [conv(x) for conv, x in zip(self.fusion_conv, xs)]
        if self.fusion == "weight":
            return torch.stack(xs, dim=0).sum(dim=0)
        if self.fusion == "adaptive":
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(xs, dim=1)), dim=1)
            weights = torch.split(fusion, [1] * len(xs), dim=1)
            return torch.stack([weights[i] * xs[i] for i in range(len(xs))], dim=0).sum(dim=0)
        weights = self.relu(self.fusion_weight.clone())
        weights = weights / (weights.sum(dim=0) + self.epsilon)
        return torch.stack([weights[i] * xs[i] for i in range(len(xs))], dim=0).sum(dim=0)


class BiFPNF(_BiFPNFBase):
    """Explicit alias to avoid collisions with pre-existing Fusion symbols."""

    pass

