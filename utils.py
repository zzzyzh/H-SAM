import os
import cv2
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
import csv
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk
import logging


# sabs
PART_TEST_VOLUME = ["0029", "0003", "0001", "0004", "0025", "0035"] # reference: https://github.com/Beckschen/TransUNet


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss

#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
#                                                                                                   target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes


def set_one_hot(pred, gt):
    _, num_cls, _, _ = pred.shape
    pred = torch.softmax(pred, dim=1)
    gt = F.one_hot(gt.squeeze(1).to(torch.long), num_cls).permute(0,3,1,2)

    return pred, gt

class DiceLoss(nn.Module):
    def __init__(self, num_cls, smooth=1e-3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask, softmax):
        """
        pred: [B, C, H, W]
        mask: [B, C, H, W]
        """
        pred, mask = set_one_hot(pred, mask)
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        dice_loss = 0.0
        
        _, num_cls, _, _ = mask.shape
        for i in range(num_cls):
            p, m = pred[:, i], mask[:, i]
            intersection = torch.sum(p * m)
            union = torch.sum(p*p) + torch.sum(m*m)
            loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = dice_loss + loss
        
        dice_loss = dice_loss / num_cls
        return dice_loss

def save_array_as_nii_volume(data, spacing_raw):
    euler3d = sitk.Euler3DTransform()
    img = sitk.GetImageFromArray(data)
    img.SetSpacing((spacing_raw[2], spacing_raw[3], spacing_raw[1]))
    xsize, ysize, zsize = img.GetSize()
    xspacing, yspacing, zspacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    spacing = [1, 1, 1.2]

    new_size = (
        int(xsize * xspacing / spacing[0]), int(ysize * yspacing / spacing[1]), int(zsize * zspacing / spacing[2]))
    img = sitk.Resample(img, new_size, euler3d, sitk.sitkNearestNeighbor, origin, spacing, direction)

    return img


def read_gt_masks(data_root_dir="/home/yanzhonghao/data/ven/bhx_sammed", mode="val", img_size=512, cls_id=0, volume=False):   
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_eval_masks = dict()
    
    gt_eval_masks_path = os.path.join(data_root_dir, mode, "masks")
    if not volume:
        for mask_name in sorted(os.listdir(gt_eval_masks_path)):
            mask = cv2.imread(os.path.join(gt_eval_masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            if cls_id != 0:
                mask[mask != cls_id] = 0
            mask = torch.from_numpy(mask)
            gt_eval_masks[mask_name] = mask
    else:
        if 'bhx' in data_root_dir:
            spacing_list = np.loadtxt(f'{data_root_dir}/{mode}/spacing.txt', delimiter=',', dtype=float).tolist()
            for _, spacing in enumerate(tqdm(spacing_list)):
                id = str(int(spacing[0])).zfill(4)
                mask_as_png = np.zeros([40, img_size, img_size], dtype='uint8')
                for mask_name in sorted(os.listdir(gt_eval_masks_path)):
                    if f'{id}_' in mask_name:
                        mask = cv2.imread(os.path.join(gt_eval_masks_path, mask_name), 0)
                        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                        if cls_id != 0:
                            mask[mask != cls_id] = 0
                        mask = Image.fromarray(mask)
                        i = int(mask_name.split('.')[0].split('_')[-1])
                        mask_as_png[i, :, :] = mask
                np.transpose(mask_as_png, [2, 0, 1])
                mask_as_nii = save_array_as_nii_volume(mask_as_png, spacing)
                gt_eval_masks[id] = torch.tensor(sitk.GetArrayFromImage(mask_as_nii))
        elif 'sabs' in data_root_dir:
            for id in PART_TEST_VOLUME:
                mask_as_png = np.zeros([200, img_size, img_size], dtype='uint8')
                for mask_name in sorted(os.listdir(gt_eval_masks_path)):
                    if f'{id}_' in mask_name:
                        mask = cv2.imread(os.path.join(gt_eval_masks_path, mask_name), 0)
                        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                        if cls_id != 0:
                            mask[mask != cls_id] = 0
                        i = int(mask_name.split('.')[0].split('_')[-1])
                        mask_as_png[i, :, :] = mask

                gt_eval_masks[id] = torch.tensor(mask_as_png)
                
    return gt_eval_masks


def create_volume_masks(data_root_dir, H, W, val_masks):
    eval_masks = dict()
    
    if 'bhx' in data_root_dir:
        spacing_list = np.loadtxt(f'/home/yanzhonghao/data/ven/bhx_sammed/test/spacing.txt', delimiter=',', dtype=float).tolist()
        
        for _, spacing in enumerate(tqdm(spacing_list)):
            seq = str(int(spacing[0])).zfill(4)
            mask_as_png = np.zeros([40, H, W], dtype='uint8')
            
            for key, value in val_masks.items():
                if seq in key:
                    id = int(key.split('_')[2].split('.')[0])
                    mask_as_png[id] = Image.fromarray(value.astype(np.uint8))

            mask_as_nii = save_array_as_nii_volume(mask_as_png, spacing)
            eval_masks[seq] = sitk.GetArrayFromImage(mask_as_nii)
         
    elif 'sabs' in data_root_dir:
        for seq in tqdm(PART_TEST_VOLUME):
            mask_as_png = np.zeros([200, H, W], dtype='uint8')
            
            for key, value in val_masks.items():
                if seq in key:
                    id = int(key.split('_')[2].split('.')[0])
                    mask_as_png[id] = value.astype(np.uint8)

            eval_masks[seq] = mask_as_png
            
    return eval_masks


def vis_pred(pred_dict, gt_dict, save_dir, num_classes):    
    
    color_mapping = {
        1: (11, 158, 150),   
        2: (27, 0, 255),     
        3: (255, 0, 255),     
        4: (241, 156, 118),    
        5: (27, 255, 255),    
        6: (227, 0, 127),    
        7: (255, 255, 0),    
        8: (0, 255, 0)     
    }

    for _, mask_name in enumerate(tqdm(list(pred_dict.keys()))):
        pred = np.array(pred_dict[mask_name])  # [256, 256]
        gt = np.array(gt_dict[mask_name])
        
        # 初始化 RGB 图像
        pred_vis = np.zeros((*pred.shape, 3), dtype=np.uint8)
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        
        # 应用颜色映射
        for cls, color in color_mapping.items():
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)
            
            pred_vis[pred_mask] = color
            gt_vis[gt_mask] = color

        plt.figure(figsize=(6, 3)) # 设置画布大小
        plt.subplot(1, 2, 1)  # 1行2列的子图中的第1个
        plt.imshow(gt_vis)  # 使用灰度颜色映射
        plt.title('Ground Truth')  # 设置标题
        plt.axis('off')  # 关闭坐标轴
        
        plt.subplot(1, 2, 2)  # 1行2列的子图中的第2个
        plt.imshow(pred_vis)  # 使用灰度颜色映射
        plt.title('Prediction')  # 设置标题
        plt.axis('off')  # 关闭坐标轴
        
        plt.savefig(os.path.join(save_dir, mask_name))
        plt.close()


def eval_metrics(eval_masks, gt_eval_masks, num_classes=4):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        eval_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_eval_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    dice_results = dict()
    iou_results = dict()
    
    all_im_dice_acc = []
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    
    im_dices = {c: [] for c in range(1, num_classes+1)}
    im_ious = {c: [] for c in range(1, num_classes+1)}    
    class_dices = {c: [] for c in range(1, num_classes+1)}
    class_ious = {c: [] for c in range(1, num_classes+1)}
    cum_I, cum_U = 0, 0
    
    for file_name, prediction in eval_masks.items():
        
        full_mask = gt_eval_masks[file_name]
        im_dice = []
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_dice_acc.append(0)
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    im_dices[class_id].append(0)
                    im_ious[class_id].append(0)
                    class_dices[class_id].append(0)
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes+1): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                dice = compute_mask_dice(current_pred, current_target)
                im_dice.append(dice)
                i, u = compute_mask_IU_eval(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                im_dices[class_id].append(dice)
                im_ious[class_id].append(i/u)
                class_dices[class_id].append(dice)
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_dice) > 0:
            all_im_dice_acc.append(np.mean(im_dice))
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    mean_im_dice = np.mean(all_im_dice_acc)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)
    mean_class_dice = torch.tensor([torch.tensor(values).float().mean() for c, values in class_dices.items() if len(values) > 0]).mean().item()
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    mean_imm_iou = cum_I / (cum_U + 1e-15)
    
    final_im_dice = torch.zeros(9)
    final_class_im_iou = torch.zeros(9)
    dice_per_class = []
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_im_dice[c-1] = torch.tensor(im_dices[c]).float().mean()
        dice_per_class.append(round((final_im_dice[c-1]*100).item(), 2))
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 2))
        
    iou_results["challengIoU"] = round(mean_im_iou_challenge*100,2)
    iou_results["IoU"] = round(mean_im_iou*100,2)
    iou_results["mcIoU"] = round(mean_class_iou*100,2)
    iou_results["mIoU"] = round(mean_imm_iou*100,2)
    iou_results["cIoU_per_class"] = cIoU_per_class
    
    dice_results["Dice"] = round(mean_im_dice*100,2)
    dice_results["mcDice"] = round(mean_class_dice*100,2)
    dice_results["Dice_per_class"] = dice_per_class
    
    return iou_results, dice_results


def compute_mask_dice(masks, target):
    """compute dice used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    intersection = (masks * target).sum()
    union = (masks + target).sum()
    dice = (2. * intersection) / union
    return dice


def compute_mask_IU_eval(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def get_logger(filename, write_mode="w", verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, write_mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger