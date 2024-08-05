import argparse
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, Focal_loss, read_gt_masks, eval_metrics, get_logger
from torchvision import transforms
from icecream import ic
from PIL import Image
from einops import repeat


def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = ((1 - dice_weight) * loss_ce + dice_weight * loss_dice)
    return loss, loss_ce, loss_dice


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


#椒盐噪声
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img


def trainer(args, model, snapshot_path, multimask_output, low_res, stage=3):
    from datasets import TrainingDataset, TestingDataset, RandomGenerator
    loggers = get_logger(os.path.join(snapshot_path, 'train.log'))
    loggers.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    base_dir = os.path.join(args.root_path, args.task, args.dataset)
    db_train = TrainingDataset(base_dir=base_dir, list_dir=args.list_dir, split='train', scale=args.scale,
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
                                        ]))
    db_val = TestingDataset(base_dir=base_dir, list_dir=args.list_dir, split='val')
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        
    gt_masks = read_gt_masks(data_root_dir=base_dir, mode='val')
    
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loggers.info('model_grad_params:' + str(model_grad_params))
    loggers.info('model_total_params:' + str(model_total_params))
    # for name, params in model.named_parameters(): 
    #     if params.requires_grad:
    #         loggers.info(name)
    
    ce_loss = CrossEntropyLoss()
    cos_loss = CosineEmbeddingLoss()
    l1_loss = nn.L1Loss()
    dice_loss = DiceLoss(num_classes + 1)
    # dice_loss = Focal_loss(num_classes=num_classes+1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    loggers.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0    
    
    for epoch_num in range(max_epoch):
        
        train_loss, train_ce1, train_ce2, train_dice1, train_dice2 = 0, 0, 0, 0, 0
        tbar = tqdm((trainloader), total = len(trainloader), leave=False)

        for sampled_batch in tbar:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label'] # [b, h, w]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            image_transform = image_batch.cpu()
            trans = transforms.Compose([
                transforms.ToPILImage(),
                AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45),p = 0.5),
                AddPepperNoise(0.7,0.9),
                transforms.ToTensor(),
                ])
            img_list = []
            for i in range(image_transform.shape[0]):
                img = image_transform[i]
                img = trans(img)
                img_list.append(img)
            image_transform = torch.stack(img_list,dim=0).cuda()
            outputs1, outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
            loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
            loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
            weight = 0.6**(0.990**epoch_num)
            loss = (1-weight)*loss1 + (weight)*loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
            writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            
            tbar.set_description(f'Train Epoch [{epoch_num}/{max_epoch}]')
            tbar.set_postfix(loss = loss.item())
            
            train_loss += loss.detach().cpu()
            train_ce1 +=  loss_ce1.detach().cpu()
            train_dice1 += loss_dice1.detach().cpu()
            train_ce2 +=  loss_ce2.detach().cpu()
            train_dice2  += loss_dice2.detach().cpu()

        loggers.info(f"Train - Epoch: {epoch_num}/{max_epoch}; Average Train Loss: {train_loss/len(trainloader)}")
        loggers.info(f"Average Train Loss Stage1: {train_ce1/len(trainloader)}, {train_dice1/len(trainloader)}; Average Train Loss Stage1: {train_ce2/len(trainloader)}, {train_dice2/len(trainloader)}")
        
        val_masks = dict()
        model.eval()
        
        with torch.no_grad():
            
            vbar = tqdm((valloader), total = len(valloader), leave=False)
            for sampled_batch in vbar:
                image_batch, label_batch, name_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'] # [b, c, h, w], [b, h, w]
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
                
                outputs1, outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=label_batch)
                if stage == 3:
                    output_masks = (outputs1['masks'] + outputs2['masks'])/2
                elif stage == 2:
                    output_masks = outputs2['masks']
                outs = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0) # [b, 512, 512]
                
                for out, im_name in zip(outs, name_batch):
                    val_masks[f'{im_name}.png'] = np.array(out.detach().cpu())
            
                vbar.set_description(f'Val Epoch [{epoch_num}/{max_epoch}]')
                
        iou_results, dice_results, _, _ = eval_metrics(val_masks, gt_masks, num_classes)
        loggers.info(f'Validation - Epoch: {epoch_num}/{max_epoch-1};')   
        loggers.info(f'IoU_Results: {iou_results};')
        loggers.info(f'Dice_Results: {dice_results}.')
        
        if dice_results['Dice'] > best_performance:
            best_performance = dice_results['Dice']
            save_mode_path = os.path.join(snapshot_path, 'best_ckpt.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            loggers.info(f'Best Dice: {best_performance:.4f} at Epoch {epoch_num+1}')        


    writer.close()
    return "Training Finished!"