import argparse 
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

import utils
from trainer import DrivingForwardTrainer, build_model

def parse_args():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--config_file', default ='./configs/nuscenes/main.yaml', type=str, help='config yaml file')
    parser.add_argument('--novel_view_mode', default='MF', type=str, help='MF of SF')
    args = parser.parse_args()
    return args

def train(cfg):    
    model = build_model(cfg, 0)
    trainer = DrivingForwardTrainer(cfg, 0)
    trainer.learn(model)

if __name__ == '__main__':
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train', novel_view_mode=args.novel_view_mode)

    train(cfg)
