import torch

from .base_loss import BaseLoss

_EPS = 1e-6


class DepthAnythingLoss(BaseLoss):
    """
    Depth Anything v3 metric (DA3METRIC-LARGE) distillation loss.
    """
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)
        self.trim_ratio = 0.1

    def _ensure_4d(self, depth):
        if depth.dim() == 2:
            return depth.unsqueeze(0).unsqueeze(0)
        if depth.dim() == 3:
            return depth.unsqueeze(1)
        if depth.dim() == 4:
            return depth
        raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")

    def _trim_and_normalize(self, depth):
        depth = self._ensure_4d(depth).float()

        batch = depth.shape[0]
        flat = depth.view(batch, -1)
        low_q = torch.quantile(flat, self.trim_ratio, dim=1, keepdim=True)
        high_q = torch.quantile(flat, 1.0 - self.trim_ratio, dim=1, keepdim=True)

        low_q = low_q.view(batch, 1, 1, 1)
        high_q = high_q.view(batch, 1, 1, 1)

        too_near = depth <= low_q
        too_far = depth >= high_q
        keep = ~(too_near | too_far)

        keep_flat = keep.view(batch, -1)
        has_keep = keep_flat.any(dim=1)

        inf = torch.tensor(float("inf"), device=depth.device, dtype=depth.dtype)
        ninf = torch.tensor(float("-inf"), device=depth.device, dtype=depth.dtype)
        depth_min = torch.amin(torch.where(keep, depth, inf), dim=(1, 2, 3))
        depth_max = torch.amax(torch.where(keep, depth, ninf), dim=(1, 2, 3))

        depth_min = torch.where(has_keep, depth_min, torch.zeros_like(depth_min))
        depth_max = torch.where(has_keep, depth_max, torch.ones_like(depth_max))

        depth_min = depth_min.view(batch, 1, 1, 1)
        denom = (depth_max - depth_min).clamp(min=_EPS).view(batch, 1, 1, 1)
        normalized = (depth - depth_min) / denom

        normalized = torch.where(
            keep,
            normalized,
            torch.where(too_near, torch.zeros_like(normalized), torch.ones_like(normalized)),
        )
        return normalized

    def forward(self, da_depth, pred_depth):
        """
        Compute log-scale L1 loss between DA3 depth and predicted depth.
        """
        da_norm = self._trim_and_normalize(da_depth)
        pred_norm = self._trim_and_normalize(pred_depth)

        da_log = torch.log(da_norm + _EPS)
        pred_log = torch.log(pred_norm + _EPS)
        return torch.abs(da_log - pred_log).mean()
