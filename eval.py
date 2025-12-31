import argparse

import os
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
    
import utils
from models import DrivingForwardModel
from trainer import DrivingForwardTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='evaluation script')
    parser.add_argument('--config_file', default='./configs/nuscenes/main.yaml', type=str, help='config yaml file path')
    parser.add_argument('--weight_path', default='./weights', type=str, help='weight path')
    parser.add_argument('--novel_view_mode', default='MF', type=str, help='MF of SF')
    parser.add_argument('--torchscript_dir', default='', type=str, help='directory with torchscript modules')
    args = parser.parse_args() 
    return args

class TorchScriptPoseNet(torch.nn.Module):
    def __init__(self, torchscript_dir, mode, device, fusion_level):
        super().__init__()
        path = os.path.join(torchscript_dir, f"pose_net_{mode}.pt")
        self.pose_net = torch.jit.load(path, map_location=device).eval()
        self.fusion_level = fusion_level

    def forward(self, inputs, frame_ids, _):
        cur_image = inputs[("color_aug", frame_ids[0], 0)]
        next_image = inputs[("color_aug", frame_ids[1], 0)]
        mask = inputs["mask"]
        k = inputs[("K", self.fusion_level + 1)]
        inv_k = inputs[("inv_K", self.fusion_level + 1)]
        extrinsics = inputs["extrinsics"]
        extrinsics_inv = inputs["extrinsics_inv"]
        axis_angle, translation = self.pose_net(
            cur_image,
            next_image,
            mask,
            k,
            inv_k,
            extrinsics,
            extrinsics_inv,
        )
        return axis_angle, translation


class TorchScriptDepthNet(torch.nn.Module):
    def __init__(self, cfg, torchscript_dir, mode, device):
        super().__init__()
        self.num_cams = cfg["data"]["num_cams"]
        self.fusion_level = cfg["model"]["fusion_level"]
        self.mode = mode
        enc_path = os.path.join(torchscript_dir, f"depth_encoder_{mode}.pt")
        dec_path = os.path.join(torchscript_dir, f"depth_decoder_{mode}.pt")
        self.depth_encoder = torch.jit.load(enc_path, map_location=device).eval()
        self.depth_decoder = torch.jit.load(dec_path, map_location=device).eval()

    def _run_one(self, images, mask, k, inv_k, extrinsics, extrinsics_inv):
        feat0, feat1, proj_feat, img_feat0, img_feat1, img_feat2 = self.depth_encoder(
            images, mask, k, inv_k, extrinsics, extrinsics_inv
        )
        disp = self.depth_decoder(feat0, feat1, proj_feat)
        img_feat = (img_feat0, img_feat1, img_feat2)
        return disp, img_feat

    def forward(self, inputs):
        outputs = {}
        for cam in range(self.num_cams):
            outputs[("cam", cam)] = {}

        images = inputs[("color_aug", 0, 0)]
        mask = inputs["mask"]
        k = inputs[("K", self.fusion_level + 1)]
        inv_k = inputs[("inv_K", self.fusion_level + 1)]
        extrinsics = inputs["extrinsics"]
        extrinsics_inv = inputs["extrinsics_inv"]

        disp_cur, img_feat_cur = self._run_one(images, mask, k, inv_k, extrinsics, extrinsics_inv)

        if self.mode == "MF":
            images_last = inputs[("color_aug", -1, 0)]
            images_next = inputs[("color_aug", 1, 0)]
            disp_last, img_feat_last = self._run_one(images_last, mask, k, inv_k, extrinsics, extrinsics_inv)
            disp_next, img_feat_next = self._run_one(images_next, mask, k, inv_k, extrinsics, extrinsics_inv)

        for cam in range(self.num_cams):
            outputs[("cam", cam)][("disp", 0)] = disp_cur[:, cam, ...]
            outputs[("cam", cam)][("img_feat", 0, 0)] = [feat[:, cam, ...] for feat in img_feat_cur]
            if self.mode == "MF":
                outputs[("cam", cam)][("disp", -1, 0)] = disp_last[:, cam, ...]
                outputs[("cam", cam)][("disp", 1, 0)] = disp_next[:, cam, ...]
                outputs[("cam", cam)][("img_feat", -1, 0)] = [feat[:, cam, ...] for feat in img_feat_last]
                outputs[("cam", cam)][("img_feat", 1, 0)] = [feat[:, cam, ...] for feat in img_feat_next]

        return outputs


class TorchScriptGaussianNet(torch.nn.Module):
    def __init__(self, torchscript_dir, mode, device):
        super().__init__()
        enc_path = os.path.join(torchscript_dir, f"gaussian_encoder_{mode}.pt")
        dec_path = os.path.join(torchscript_dir, f"gaussian_decoder_{mode}.pt")
        self.gaussian_encoder = torch.jit.load(enc_path, map_location=device).eval()
        self.gaussian_decoder = torch.jit.load(dec_path, map_location=device).eval()

    def forward(self, img, depth, img_feat):
        depth_feat1, depth_feat2, depth_feat3 = self.gaussian_encoder(depth)
        return self.gaussian_decoder(
            img,
            depth,
            img_feat[0],
            img_feat[1],
            img_feat[2],
            depth_feat1,
            depth_feat2,
            depth_feat3,
        )


class TorchScriptDrivingForwardModel(DrivingForwardModel):
    def __init__(self, cfg, rank, torchscript_dir):
        self.torchscript_dir = torchscript_dir
        super().__init__(cfg, rank)

    def prepare_model(self, cfg, rank):
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        mode = cfg["model"]["novel_view_mode"]
        models = {
            "pose_net": TorchScriptPoseNet(
                self.torchscript_dir,
                mode,
                device,
                cfg["model"]["fusion_level"],
            ),
            "depth_net": TorchScriptDepthNet(cfg, self.torchscript_dir, mode, device),
        }
        if self.gaussian:
            models["gs_net"] = TorchScriptGaussianNet(self.torchscript_dir, mode, device)
        return models

    def load_weights(self):
        if self.rank == 0:
            print("Skipping weight loading for torchscript mode.")


def test(cfg, torchscript_dir):
    print("Evaluating reconstruction")
    if torchscript_dir:
        model = TorchScriptDrivingForwardModel(cfg, 0, torchscript_dir)
    else:
        model = DrivingForwardModel(cfg, 0)
    trainer = DrivingForwardTrainer(cfg, 0, use_tb = False)
    trainer.evaluate(model)

if __name__ == '__main__':
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='eval', weight_path=args.weight_path, novel_view_mode=args.novel_view_mode)
        
    test(cfg, args.torchscript_dir)
