import argparse
from pathlib import Path

import torch
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

import utils
from models import DrivingForwardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Render a sequence in eval order.")
    parser.add_argument(
        "--config_file",
        default="./configs/nuscenes/main.yaml",
        type=str,
        help="config yaml file path",
    )
    parser.add_argument(
        "--weight_path",
        default="./weights",
        type=str,
        help="weight path",
    )
    parser.add_argument(
        "--novel_view_mode",
        default="MF",
        type=str,
        help="MF or SF",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/sequence",
        type=str,
        help="output directory for rendered frames",
    )
    parser.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="max frames to render (0 for all)",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="render every Nth frame",
    )
    parser.add_argument(
        "--save_gt",
        action="store_true",
        help="also save ground-truth and input frames",
    )
    return parser.parse_args()


def prep_image(image):
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prep_image(image)).save(path)


@torch.no_grad()
def render_sequence(cfg, output_dir, max_frames, stride, save_gt):
    model = DrivingForwardModel(cfg, 0)
    model.load_weights()
    model.set_eval()

    eval_dataloader = model.eval_dataloader()
    frame_id = 1 if cfg["model"]["novel_view_mode"] == "SF" else 0

    rendered = 0
    for batch_idx, inputs in enumerate(tqdm(eval_dataloader)):
        if stride > 1 and batch_idx % stride != 0:
            continue
        if max_frames and rendered >= max_frames:
            break

        outputs, _ = model.process_batch(inputs, 0)

        token = inputs["token"][0]
        idx_tensor = inputs.get("idx", None)
        if idx_tensor is not None:
            idx = int(idx_tensor[0])
        else:
            idx = batch_idx
        sample_dir = Path(output_dir) / f"{idx:06d}_{token}"

        for cam in range(model.num_cams):
            image = outputs[("cam", cam)][("gaussian_color", frame_id, 0)]
            save_image(image, sample_dir / f"{cam}.png")

            if save_gt:
                rgb_gt = inputs[("color", frame_id, 0)][:, cam, ...]
                save_image(rgb_gt, sample_dir / f"{cam}_gt.png")
                if cfg["model"]["novel_view_mode"] == "SF":
                    save_image(
                        inputs[("color", 0, 0)][:, cam, ...],
                        sample_dir / f"{cam}_0_gt.png",
                    )
                else:
                    save_image(
                        inputs[("color", -1, 0)][:, cam, ...],
                        sample_dir / f"{cam}_prev_gt.png",
                    )
                    save_image(
                        inputs[("color", 1, 0)][:, cam, ...],
                        sample_dir / f"{cam}_next_gt.png",
                    )

        rendered += 1


if __name__ == "__main__":
    args = parse_args()
    cfg = utils.get_config(
        args.config_file,
        mode="eval",
        weight_path=args.weight_path,
        novel_view_mode=args.novel_view_mode,
    )
    render_sequence(cfg, args.output_dir, args.max_frames, args.stride, args.save_gt)
