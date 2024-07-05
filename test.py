import os
import sys
import logging
import argparse
import random
from tqdm import tqdm
from icecream import ic
from importlib import import_module
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from segment_anything import sam_model_registry
from datasets import TestingDataset
from utils import read_gt_masks, create_volume_masks, eval_metrics, vis_pred, get_logger

    
def inference(args, multimask_output, model, test_save_path):
    base_dir = os.path.join(args.root_path, args.task, args.dataset)
    db_val = TestingDataset(base_dir=base_dir, list_dir=args.list_dir, split='test')
    testloader = DataLoader(db_val, batch_size=32, shuffle=False, num_workers=8)
    gt_masks = read_gt_masks(data_root_dir=base_dir, mode='test', img_size=args.img_size, volume=args.volume)

    val_masks = dict()
    model.eval()
    
    with torch.no_grad():
        
        tbar = tqdm((testloader), total = len(testloader), leave=False)
        for sampled_batch in tbar:
            image_batch, label_batch, name_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'] # [b, c, h, w], [b, h, w]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            
            outputs1, outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=label_batch)
            if args.stage == 3:
                output_masks = (outputs1['masks'] + outputs2['masks'])/2
            elif args.stage == 2:
                output_masks = outputs2['masks']
            outs = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0) # [b, 512, 512]
            
            for out, im_name in zip(outs, name_batch):
                val_masks[f'{im_name}.png'] = np.array(out.detach().cpu())
    
    if args.volume:
        val_masks = create_volume_masks(base_dir, args.img_size, args.img_size, val_masks)
        
    iou_results, dice_results = eval_metrics(val_masks, gt_masks, args.num_classes)
    logging.info(f'IoU_Results: {iou_results};')
    logging.info(f'Dice_Results: {dice_results}.')
    if not args.volume:
        vis_pred(val_masks, gt_masks, test_save_path, num_classes=args.num_classes)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set-up Model
    parser.add_argument('--task', type=str,
                        default='abdomen', help='task name')
    parser.add_argument('--dataset', type=str,
                        default='sabs_sammed', help='dataset name')
    parser.add_argument('--root_path', type=str,
                        default='/home/yanzhonghao/data', help='root dir for data')
    parser.add_argument('--output', type=str, default='/home/yanzhonghao/data/experiments/h_sam')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_sabs', help='list dir')
    parser.add_argument('--split', type=str,
                        default='test', help='list dir')
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit model')
    parser.add_argument('--ckpt', type=str, default='/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
    parser.add_argument('--scale', type=int, default=127)
    
    # Running Strategy
    parser.add_argument('--img_size', type=int,
                        default=512, help='input image size')
    parser.add_argument('--resolution', type=int,
                        default=512, help='input patch size of network input')
    parser.add_argument('--num_classes', type=int,
                        default=8, help='output channel of network')
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--rank', type=int, default=5, help='Rank for LoRA adaptation')
    parser.add_argument('--train_time', type=str, default=None)
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--volume', type=bool, default=False, help='whether to evaluate test set in volume')
    parser.add_argument('--vis', type=bool, default=False, help='whether to visualise results')
    parser.add_argument('--seed', type=int,default=2024, help='random seed')
    
    args = parser.parse_args()

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
    now = args.train_time
    args.exp = f'{dataset_name}_{args.img_size}_{now}_{args.scale}'
    snapshot_path = os.path.join(args.output, dataset_name, args.exp)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    args.lora_ckpt = os.path.join(snapshot_path, 'best_ckpt.pth')
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
        
    # initialize log
    log_path = os.path.join(snapshot_path, 'test.log') if not args.volume else os.path.join(snapshot_path, 'test_volume.log')
    loggers = get_logger(log_path)
    loggers.info(str(args))
    
    test_save_path = os.path.join(snapshot_path, 'predictions')
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, multimask_output, net, test_save_path)