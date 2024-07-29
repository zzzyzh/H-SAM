# 0.1
CUDA_VISIBLE_DEVICES=2 python train.py --split=train --scale=882 --task=ven --dataset=bhx_sammed --num_classes=4 --list_dir=./lists/lists_bhx --batch_size=16 --base_lr=0.003 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth

# 0.05
CUDA_VISIBLE_DEVICES=2 python train.py --split=train --scale=441 --task=ven --dataset=bhx_sammed --num_classes=4 --list_dir=./lists/lists_bhx --batch_size=16 --base_lr=0.003 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth

# 0.01
CUDA_VISIBLE_DEVICES=2 python train.py --split=train --scale=88 --task=ven --dataset=bhx_sammed --num_classes=4 --list_dir=./lists/lists_bhx --batch_size=16 --base_lr=0.003 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth