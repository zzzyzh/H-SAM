import argparse
import os
from datetime import datetime

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module
from segment_anything import sam_model_registry

from trainer import trainer
from icecream import ic


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

parser = argparse.ArgumentParser()
# Set-up Model
parser.add_argument('--task', type=str,
                    default='abdomen', help='task name')
parser.add_argument('--dataset', type=str,
                    default='sabs_sammed', help='dataset name')
parser.add_argument('--root_path', type=str,
                    default='../../data', help='root dir for data')
parser.add_argument('--output', type=str, default='../../data/experiments/h_sam')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_sabs', help='list dir')
parser.add_argument('--split', type=str,
                    default='train', help='list dir')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='../../data/experiments/weights/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--scale', type=int, default=127)

# Running Strategy
parser.add_argument('--is_pretrain', type=bool,
                    default=True, help='use pre-train model')
parser.add_argument('--img_size', type=int,
                    default=512, help='input image size')
parser.add_argument('--resolution', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--rank', type=int, default=5, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=2024, help='random seed')

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    now = datetime.now().strftime('%Y%m%d-%H%M')
    args.exp = f'{dataset_name}_{args.img_size}_{now}_{args.scale}'
    snapshot_path = os.path.join(args.output, dataset_name, args.exp)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # net = LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer(args, net, snapshot_path, multimask_output, low_res)
