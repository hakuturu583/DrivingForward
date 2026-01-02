import inspect
import importlib

import torch
import torch.nn.functional as F

from .base_loss import BaseLoss

_EPS = 1e-6


class DepthAnythingLoss(BaseLoss):
    """
    Depth Anything v3 metric (DA3METRIC-LARGE) distillation loss.
    """
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)
        self.trim_ratio = 0.1
        self.da3_variant = getattr(self, "da3_variant", "metric-large")
        self.da3_weights_path = getattr(self, "da3_weights_path", None)
        self._da3_model = None

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

    def _resolve_da3_model(self):
        if self._da3_model is not None:
            return self._da3_model

        candidates = [
            ("depth_anything_3", ["DepthAnything3", "DepthAnythingV3", "DepthAnything"]),
            ("depth_anything_3.dpt", ["DepthAnything3", "DepthAnythingV3", "DepthAnything"]),
            ("depth_anything_3.model", ["DepthAnything3", "DepthAnythingV3", "DepthAnything"]),
            ("depth_anything_3.depth_anything", ["DepthAnything3", "DepthAnythingV3", "DepthAnything"]),
        ]
        for module_name, class_names in candidates:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            for class_name in class_names:
                model_cls = getattr(module, class_name, None)
                if model_cls is None:
                    continue
                model = self._instantiate_model(model_cls)
                self._da3_model = model
                return model

        factory_candidates = [
            ("depth_anything_3", ["create_model", "build_model", "get_model", "load_model"]),
        ]
        for module_name, func_names in factory_candidates:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            for func_name in func_names:
                factory = getattr(module, func_name, None)
                if factory is None:
                    continue
                model = self._instantiate_factory(factory)
                self._da3_model = model
                return model

        raise ImportError(
            "DepthAnything3 module not found. Install depth-anything-3 or "
            "provide a supported module path."
        )

    def _instantiate_model(self, model_cls):
        init = model_cls.__init__
        sig = inspect.signature(init)
        kwargs = {}
        for key in ("variant", "model", "encoder", "mode", "name"):
            if key in sig.parameters:
                kwargs[key] = self.da3_variant
                break
        model = model_cls(**kwargs) if kwargs else model_cls()
        self._load_weights_if_needed(model)
        return model

    def _instantiate_factory(self, factory):
        sig = inspect.signature(factory)
        kwargs = {}
        for key in ("variant", "model", "encoder", "mode", "name"):
            if key in sig.parameters:
                kwargs[key] = self.da3_variant
                break
        model = factory(**kwargs) if kwargs else factory()
        self._load_weights_if_needed(model)
        return model

    def _load_weights_if_needed(self, model):
        if not self.da3_weights_path:
            return
        state = torch.load(self.da3_weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    def _infer_depth(self, images):
        model = self._resolve_da3_model()
        device = images.device
        if hasattr(model, "to"):
            model = model.to(device)
        model.eval()

        with torch.no_grad():
            if hasattr(model, "infer_image"):
                depth = model.infer_image(images)
            else:
                depth = model(images)

        if isinstance(depth, (list, tuple)):
            depth = depth[0]
        if isinstance(depth, dict):
            depth = depth.get("depth", depth.get("pred", depth.get("output")))
        if depth is None:
            raise RuntimeError("DepthAnything3 inference returned no depth output.")
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        return depth

    def forward(self, images, pred_depth):
        """
        Compute log-scale L1 loss between DA3 depth (from images) and predicted depth.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected images in BCHW format, got {tuple(images.shape)}")
        if images.shape[-2:] != pred_depth.shape[-2:]:
            images = F.interpolate(images, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)

        da_depth = self._infer_depth(images)
        da_norm = self._trim_and_normalize(da_depth)
        pred_norm = self._trim_and_normalize(pred_depth)

        da_log = torch.log(da_norm + _EPS)
        pred_log = torch.log(pred_norm + _EPS)
        return torch.abs(da_log - pred_log).mean()
