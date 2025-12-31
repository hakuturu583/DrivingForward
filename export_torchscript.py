import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import utils
from network import DepthNetwork, PoseNetwork
from network.blocks import pack_cam_feat, unpack_cam_feat
from models.gaussian import GaussianNetwork


class DepthEncoderSF(nn.Module):
    def __init__(self, depth_net):
        super().__init__()
        self.encoder = depth_net.encoder
        self.conv1x1 = depth_net.conv1x1
        self.fusion_net = depth_net.fusion_net
        self.fusion_level = depth_net.fusion_level
        self.num_cams = depth_net.num_cams

    def _encode_one(self, images, mask, k, inv_k, extrinsics, extrinsics_inv):
        packed_input = pack_cam_feat(images)
        packed_feats = self.encoder(packed_input)

        lev = self.fusion_level
        _, _, up_h, up_w = packed_feats[lev].size()
        packed_feats_list = packed_feats[lev:lev + 1] + [
            F.interpolate(feat, [up_h, up_w], mode="bilinear", align_corners=True)
            for feat in packed_feats[lev + 1:]
        ]
        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))
        feats_agg = unpack_cam_feat(packed_feats_agg, images.size(0), images.size(1))

        fusion_dict = self.fusion_net({
            "mask": mask,
            ("K", lev + 1): k,
            ("inv_K", lev + 1): inv_k,
            "extrinsics": extrinsics,
            "extrinsics_inv": extrinsics_inv,
        }, feats_agg)

        feat0 = packed_feats[0]
        feat1 = packed_feats[1]
        proj_feat = fusion_dict["proj_feat"]

        img_feat0 = unpack_cam_feat(feat0, images.size(0), images.size(1))
        img_feat1 = unpack_cam_feat(feat1, images.size(0), images.size(1))
        img_feat2 = unpack_cam_feat(proj_feat, images.size(0), images.size(1))
        return feat0, feat1, proj_feat, img_feat0, img_feat1, img_feat2

    def forward(self, images, mask, k, inv_k, extrinsics, extrinsics_inv):
        return self._encode_one(images, mask, k, inv_k, extrinsics, extrinsics_inv)


class DepthEncoderMF(nn.Module):
    def __init__(self, depth_net):
        super().__init__()
        self.encoder = depth_net.encoder
        self.conv1x1 = depth_net.conv1x1
        self.fusion_net = depth_net.fusion_net
        self.fusion_level = depth_net.fusion_level
        self.num_cams = depth_net.num_cams

    def _encode_one(self, images, mask, k, inv_k, extrinsics, extrinsics_inv):
        packed_input = pack_cam_feat(images)
        packed_feats = self.encoder(packed_input)

        lev = self.fusion_level
        _, _, up_h, up_w = packed_feats[lev].size()
        packed_feats_list = packed_feats[lev:lev + 1] + [
            F.interpolate(feat, [up_h, up_w], mode="bilinear", align_corners=True)
            for feat in packed_feats[lev + 1:]
        ]
        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))
        feats_agg = unpack_cam_feat(packed_feats_agg, images.size(0), images.size(1))

        fusion_dict = self.fusion_net({
            "mask": mask,
            ("K", lev + 1): k,
            ("inv_K", lev + 1): inv_k,
            "extrinsics": extrinsics,
            "extrinsics_inv": extrinsics_inv,
        }, feats_agg)

        feat0 = packed_feats[0]
        feat1 = packed_feats[1]
        proj_feat = fusion_dict["proj_feat"]

        img_feat0 = unpack_cam_feat(feat0, images.size(0), images.size(1))
        img_feat1 = unpack_cam_feat(feat1, images.size(0), images.size(1))
        img_feat2 = unpack_cam_feat(proj_feat, images.size(0), images.size(1))
        return feat0, feat1, proj_feat, img_feat0, img_feat1, img_feat2

    def forward(
        self,
        images,
        images_last,
        images_next,
        mask,
        k,
        inv_k,
        extrinsics,
        extrinsics_inv,
    ):
        cur = self._encode_one(images, mask, k, inv_k, extrinsics, extrinsics_inv)
        last = self._encode_one(images_last, mask, k, inv_k, extrinsics, extrinsics_inv)
        nxt = self._encode_one(images_next, mask, k, inv_k, extrinsics, extrinsics_inv)
        return cur + last + nxt


class DepthDecoderWrapper(nn.Module):
    def __init__(self, depth_net):
        super().__init__()
        self.decoder = depth_net.decoder
        self.num_cams = depth_net.num_cams

    def forward(self, feat0, feat1, proj_feat):
        outputs = self.decoder([feat0, feat1, proj_feat])
        disp = outputs[("disp", 0)]
        batch_size = disp.size(0) // self.num_cams
        return unpack_cam_feat(disp, batch_size, self.num_cams)


class GaussianEncoderWrapper(nn.Module):
    def __init__(self, gs_net):
        super().__init__()
        self.depth_encoder = gs_net.depth_encoder

    def forward(self, depth):
        return self.depth_encoder(depth)


class GaussianDecoderWrapper(nn.Module):
    def __init__(self, gs_net):
        super().__init__()
        self.decoder3 = gs_net.decoder3
        self.decoder2 = gs_net.decoder2
        self.decoder1 = gs_net.decoder1
        self.up = gs_net.up
        self.out_conv = gs_net.out_conv
        self.out_relu = gs_net.out_relu
        self.rot_head = gs_net.rot_head
        self.scale_head = gs_net.scale_head
        self.opacity_head = gs_net.opacity_head
        self.sh_head = gs_net.sh_head
        self.sh_mask = gs_net.sh_mask

    def forward(
        self,
        img,
        depth,
        img_feat1,
        img_feat2,
        img_feat3,
        depth_feat1,
        depth_feat2,
        depth_feat3,
    ):
        feat3 = torch.cat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.cat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.cat([img_feat1, depth_feat1], dim=1)

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        rot_out = self.rot_head(out)
        rot_out = F.normalize(rot_out, dim=1)

        scale_out = torch.clamp_max(self.scale_head(out), 0.01)
        opacity_out = self.opacity_head(out)

        sh_out = self.sh_head(out)
        sh_out = rearrange(sh_out, "n c h w -> n (h w) c")
        sh_out = rearrange(sh_out, "... (srf c) -> ... srf () c", srf=1)
        sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh_out = sh_out * self.sh_mask
        return rot_out, scale_out, opacity_out, sh_out


class PoseWrapper(nn.Module):
    def __init__(self, pose_net):
        super().__init__()
        self.pose_net = pose_net

    def forward(self, cur_image, next_image):
        inputs = {
            ("color_aug", 0, 0): cur_image,
            ("color_aug", 1, 0): next_image,
        }
        axis_angle, translation = self.pose_net(inputs, [0, 1], None)
        return axis_angle, translation


def load_state_dict(model, weight_path, filename):
    state_path = os.path.join(weight_path, filename)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing weights: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)


def make_dummy_inputs(cfg, mode):
    batch_size = cfg["training"]["batch_size"]
    num_cams = cfg["data"]["num_cams"]
    height = cfg["training"]["height"]
    width = cfg["training"]["width"]

    images = torch.zeros(batch_size, num_cams, 3, height, width)
    images_last = torch.zeros_like(images)
    images_next = torch.zeros_like(images)
    mask = torch.ones(batch_size, num_cams, 1, height, width)

    k = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, num_cams, 1, 1)
    inv_k = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, num_cams, 1, 1)
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
    extrinsics_inv = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)

    if mode == "MF":
        return images, images_last, images_next, mask, k, inv_k, extrinsics, extrinsics_inv
    return images, mask, k, inv_k, extrinsics, extrinsics_inv


def export_mode(cfg, weight_path, out_dir, mode):
    depth_net = DepthNetwork(cfg).eval()
    gs_net = GaussianNetwork(rgb_dim=3, depth_dim=1).eval()
    pose_net = PoseNetwork(cfg).eval()
    load_state_dict(depth_net, weight_path, "depth_net.pth")
    load_state_dict(gs_net, weight_path, "gs_net.pth")
    load_state_dict(pose_net, weight_path, "pose_net.pth")

    if mode == "MF":
        depth_encoder = DepthEncoderMF(depth_net).eval()
        encoder_inputs = make_dummy_inputs(cfg, mode)
    else:
        depth_encoder = DepthEncoderSF(depth_net).eval()
        encoder_inputs = make_dummy_inputs(cfg, mode)

    depth_decoder = DepthDecoderWrapper(depth_net).eval()
    gaussian_encoder = GaussianEncoderWrapper(gs_net).eval()
    gaussian_decoder = GaussianDecoderWrapper(gs_net).eval()

    traced_depth_encoder = torch.jit.trace(depth_encoder, encoder_inputs)

    if mode == "MF":
        feat0, feat1, proj_feat = traced_depth_encoder(*encoder_inputs)[:3]
    else:
        feat0, feat1, proj_feat, _, _, _ = traced_depth_encoder(*encoder_inputs)

    traced_depth_decoder = torch.jit.trace(depth_decoder, (feat0, feat1, proj_feat))

    depth = torch.zeros(cfg["training"]["batch_size"], 1, cfg["training"]["height"], cfg["training"]["width"])
    img = torch.zeros(cfg["training"]["batch_size"], 3, cfg["training"]["height"], cfg["training"]["width"])
    depth_feat1, depth_feat2, depth_feat3 = gaussian_encoder(depth)
    img_feat1 = torch.zeros(cfg["training"]["batch_size"], 64, cfg["training"]["height"] // 2, cfg["training"]["width"] // 2)
    img_feat2 = torch.zeros(cfg["training"]["batch_size"], 64, cfg["training"]["height"] // 4, cfg["training"]["width"] // 4)
    img_feat3 = torch.zeros(cfg["training"]["batch_size"], 128, cfg["training"]["height"] // 8, cfg["training"]["width"] // 8)

    traced_gaussian_encoder = torch.jit.trace(gaussian_encoder, (depth,))
    traced_gaussian_decoder = torch.jit.trace(
        gaussian_decoder,
        (img, depth, img_feat1, img_feat2, img_feat3, depth_feat1, depth_feat2, depth_feat3),
    )

    pose_wrapper = PoseWrapper(pose_net).eval()
    pose_cur = torch.zeros(cfg["training"]["batch_size"], cfg["data"]["num_cams"], 3, cfg["training"]["height"], cfg["training"]["width"])
    pose_next = torch.zeros_like(pose_cur)
    traced_pose = torch.jit.trace(pose_wrapper, (pose_cur, pose_next))

    os.makedirs(out_dir, exist_ok=True)
    torch.jit.save(traced_depth_encoder, os.path.join(out_dir, f"depth_encoder_{mode}.pt"))
    torch.jit.save(traced_depth_decoder, os.path.join(out_dir, f"depth_decoder_{mode}.pt"))
    torch.jit.save(traced_gaussian_encoder, os.path.join(out_dir, f"gaussian_encoder_{mode}.pt"))
    torch.jit.save(traced_gaussian_decoder, os.path.join(out_dir, f"gaussian_decoder_{mode}.pt"))
    torch.jit.save(traced_pose, os.path.join(out_dir, f"pose_net_{mode}.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Export TorchScript modules.")
    parser.add_argument("--config_file", default="./configs/nuscenes/main.yaml", type=str)
    parser.add_argument("--weights_mf", default="./weights_MF", type=str)
    parser.add_argument("--weights_sf", default="./weights_SF", type=str)
    parser.add_argument("--output_dir", default="./torchscript", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    cfg_mf = utils.get_config(args.config_file, mode="eval", weight_path=args.weights_mf, novel_view_mode="MF")
    cfg_sf = utils.get_config(args.config_file, mode="eval", weight_path=args.weights_sf, novel_view_mode="SF")

    cfg_mf["model"]["weights_init"] = False
    cfg_sf["model"]["weights_init"] = False

    export_mode(cfg_mf, args.weights_mf, args.output_dir, "MF")
    export_mode(cfg_sf, args.weights_sf, args.output_dir, "SF")


if __name__ == "__main__":
    main()
