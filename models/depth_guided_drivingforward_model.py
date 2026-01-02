import torch

from .drivingforward_model import DrivingForwardModel
from .losses import DepthAnythingLoss


class _NoPoseNet(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("PoseNetwork is disabled; using NuScenes GT pose.")


class DepthGuidedDrivingForwardModel(DrivingForwardModel):
    """
    DrivingForward model that uses NuScenes GT pose and adds DepthAnything loss.
    """
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)
        self.depth_anything_loss = DepthAnythingLoss(cfg, rank)
        if not hasattr(self, "depth_anything_coeff"):
            self.depth_anything_coeff = 1.0
        if not hasattr(self, "da3_image_key"):
            self.da3_image_key = ("color", 0, 0)

    def prepare_model(self, cfg, rank):
        models = {
            "pose_net": _NoPoseNet(),
            "depth_net": self.set_depthnet(cfg),
        }
        if self.gaussian:
            models["gs_net"] = self.set_gaussiannet(cfg)
        return models

    def _select_cam_pose(self, pose, cam):
        if pose.dim() == 4:
            return pose[:, cam, ...]
        if pose.dim() == 3:
            return pose[cam, ...].unsqueeze(0)
        raise ValueError(f"Unsupported pose shape: {tuple(pose.shape)}")

    def predict_pose(self, inputs):
        outputs = {}
        for cam in range(self.num_cams):
            outputs[("cam", cam)] = {}

        for frame_id in self.frame_ids[1:]:
            key = ("cam_T_cam", 0, frame_id)
            if key not in inputs:
                raise KeyError(f"Missing {key} in inputs; enable gt_ego_pose in dataset.")
            cam_T_cam = inputs[key]
            for cam in range(self.num_cams):
                outputs[("cam", cam)][("cam_T_cam", 0, frame_id)] = self._select_cam_pose(
                    cam_T_cam, cam
                )
        return outputs

    def _select_cam_depth(self, depth, cam):
        if depth.dim() == 5:
            return depth[:, cam, ...]
        if depth.dim() == 4:
            return depth[:, cam, ...]
        if depth.dim() == 3:
            return depth[cam, ...].unsqueeze(0)
        raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")

    def compute_depth_anything_loss(self, inputs, outputs):
        depth_loss = 0.0
        for cam in range(self.num_cams):
            pred_depth = outputs[("cam", cam)][("depth", 0, 0)]
            if self.da3_image_key not in inputs:
                raise KeyError(
                    f"Missing {self.da3_image_key} in inputs; provide image for DA3."
                )
            images = inputs[self.da3_image_key]
            images = self._select_cam_depth(images, cam)
            depth_loss += self.depth_anything_loss(images, pred_depth)
        return depth_loss / float(self.num_cams)

    def compute_losses(self, inputs, outputs):
        loss_mean = super().compute_losses(inputs, outputs)
        depth_loss = self.compute_depth_anything_loss(inputs, outputs)
        loss_mean["depth_anything_loss"] = depth_loss
        loss_mean["total_loss"] = loss_mean["total_loss"] + self.depth_anything_coeff * depth_loss
        return loss_mean
