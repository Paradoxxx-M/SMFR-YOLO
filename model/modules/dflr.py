import math

import torch
import torch.nn as nn

from .conv import Conv, DWConv
from .head import Detect

class _ZeroInitProjection(nn.Conv2d):
    def __init__(self, c1, c2=None):
        super().__init__(c1, c2 or c1, 1, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class RegUncertaintyRouterBlock(nn.Module):
    def __init__(self, c, level_id=3):
        super().__init__()
        self.level_id = int(level_id)
        self.small = DWConv(c, c, 3)
        self.large = DWConv(c, c, 5)
        self.router = nn.Sequential(Conv(c + 4, max(c // 8, 8), 3), nn.Conv2d(max(c // 8, 8), 1, 1))
        self.proj = _ZeroInitProjection(c)
        self.last_route_mean = self.last_uncertainty_mean = 0.0
        self.last_used_large_expert = None
        self.collect_stats = False

    def set_collect_stats(self, enabled: bool = True):
        self.collect_stats = bool(enabled)
        return self

    def forward(self, x, uncertainty=None):
        if uncertainty is None:
            uncertainty = x.new_zeros((x.shape[0], 4, *x.shape[-2:]))
        uncertainty = uncertainty.detach()
        if self.collect_stats:
            self.last_uncertainty_mean = float(uncertainty.mean())
        if self.level_id == 5:
            if self.collect_stats:
                self.last_used_large_expert = False
            return x + self.proj(self.small(x))
        gate = torch.sigmoid(self.router(torch.cat((x, uncertainty), 1)))
        if self.collect_stats:
            self.last_route_mean = float(gate.detach().mean())
            self.last_used_large_expert = True
        return x + self.proj((1.0 - gate) * self.small(x) + gate * self.large(x))


class _RegLocalResidualBlock(nn.Module):
    def __init__(self, c, level_id=3):
        super().__init__()
        self.level_id = int(level_id)
        self.transform = nn.Sequential(DWConv(c, c, 3), Conv(c, c, 1), DWConv(c, c, 3))
        self.proj = _ZeroInitProjection(c)
        self.last_delta_rms = 0.0
        self.collect_stats = False

    def set_collect_stats(self, enabled: bool = True):
        self.collect_stats = bool(enabled)
        return self

    def forward(self, x, uncertainty=None):
        if self.level_id == 5:
            return x
        delta = self.proj(self.transform(x))
        if self.collect_stats:
            self.last_delta_rms = float(torch.sqrt(delta.detach().square().mean() + 1e-12))
        return x + delta


class _DFLRBase(Detect):
    block_cls = _RegLocalResidualBlock

    def __init__(self, nc=80, ch=(), block_kwargs=None):
        super().__init__(nc, ch)
        block_kwargs = dict(block_kwargs or {})
        stems, predictors = [], []
        for branch in self.cv2:
            children = list(branch.children())
            stems.append(nn.Sequential(*children[:-1]))
            predictors.append(children[-1])
        self.reg_stem = nn.ModuleList(stems)
        self.box_pred = nn.ModuleList(predictors)
        self.reg_blocks = nn.ModuleList(self.block_cls(self.box_pred[i].in_channels, level_id=i + 3, **block_kwargs) for i in range(self.nl))
        self.collect_stats = False
        del self.cv2

    def set_collect_stats(self, enabled: bool = True):
        self.collect_stats = bool(enabled)
        for block in self.reg_blocks:
            if hasattr(block, "set_collect_stats"):
                block.set_collect_stats(enabled)
        return self

    @staticmethod
    def _uncertainty(logits, reg_max):
        b, _, h, w = logits.shape
        p = logits.view(b, 4, reg_max, h, w).softmax(2)
        return (-(p * p.clamp_min(1e-8).log()).sum(2) / math.log(reg_max)).detach()

    def _reg_logits(self, x):
        out = []
        for i in range(self.nl):
            q = self.reg_stem[i](x[i])
            if isinstance(self.reg_blocks[i], RegUncertaintyRouterBlock):
                with torch.no_grad():
                    coarse = self.box_pred[i](q)
                q = self.reg_blocks[i](q, self._uncertainty(coarse, self.reg_max))
            else:
                q = self.reg_blocks[i](q)
            out.append(self.box_pred[i](q))
        return out

    def forward(self, x):
        boxes = self._reg_logits(x)
        y = [torch.cat((boxes[i], self.cv3[i](x[i])), 1) for i in range(self.nl)]
        return y if self.training else self.inference(y)

    def bias_init(self):
        for box, cls, stride in zip(self.box_pred, self.cv3, self.stride):
            box.bias.data[:] = 1.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class DFLR(_DFLRBase):
    def __init__(self, nc=80, ch=(), refine_scales=(0.15, 0.10, 0.05)):
        super().__init__(nc, ch)
        self.refine_scales = tuple(float(v) for v in refine_scales)
        if len(self.refine_scales) != self.nl:
            raise ValueError(f"refine_scales must have {self.nl} entries, got {self.refine_scales}")
        self.refiners = nn.ModuleList(
            nn.Sequential(Conv(box.in_channels + 8, box.in_channels, 3), _ZeroInitProjection(box.in_channels, 4 * self.reg_max))
            for box in self.box_pred
        )
        self.last_uncertainty_mean_by_level = {3: None, 4: None, 5: None}

    def _reg_logits(self, x):

        out = []
        bins = None

        for i in range(self.nl):

            q = self.reg_stem[i](x[i])
            z0 = self.box_pred[i](q)

            b, _, h, w = z0.shape

            use_fp32_stats = (
                (not self.training)
                and (
                    z0.dtype
                    in (
                        torch.float16,
                        torch.bfloat16,
                    )
                    or torch.onnx.is_in_onnx_export()
                )
            )

            if use_fp32_stats:

                z_stats = z0.float()

                p = z_stats.view(
                    b,
                    4,
                    self.reg_max,
                    h,
                    w,
                ).softmax(2)

                u_stats = (
                    -(
                        p
                        * p.clamp_min(
                            1e-8
                        ).log()
                    ).sum(2)
                    / math.log(
                        self.reg_max
                    )
                )

                if bins is None:
                    bins = torch.arange(
                        self.reg_max,
                        device=z0.device,
                        dtype=torch.float32,
                    ).view(
                        1,
                        1,
                        -1,
                        1,
                        1,
                    )

                mean_stats = (
                    (p * bins).sum(2)
                    / max(
                        self.reg_max - 1,
                        1,
                    )
                )

                u = (
                    u_stats.detach()
                    .to(dtype=q.dtype)
                )

                mean = (
                    mean_stats.detach()
                    .to(dtype=q.dtype)
                )

            else:

                p = z0.view(
                    b,
                    4,
                    self.reg_max,
                    h,
                    w,
                ).softmax(2)

                u = (
                    -(
                        p
                        * p.clamp_min(
                            1e-8
                        ).log()
                    ).sum(2)
                    / math.log(
                        self.reg_max
                    )
                ).detach()

                if bins is None:
                    bins = torch.arange(
                        self.reg_max,
                        device=z0.device,
                        dtype=z0.dtype,
                    ).view(
                        1,
                        1,
                        -1,
                        1,
                        1,
                    )

                mean = (
                    (p * bins).sum(2)
                    / max(
                        self.reg_max - 1,
                        1,
                    )
                ).detach()

            if self.collect_stats:
                self.last_uncertainty_mean_by_level[
                    i + 3
                ] = float(
                    u.mean()
                )

            delta = self.refiners[i](
                torch.cat(
                    (
                        q,
                        u,
                        mean,
                    ),
                    1,
                )
            )

            out.append(
                z0
                + self.refine_scales[i]
                * torch.tanh(delta)
            )

        return out

