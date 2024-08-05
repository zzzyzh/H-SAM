# 0.1
CUDA_VISIBLE_DEVICES=1,2 python train.py --n_gpu=2 --split=train --scale=127 --batch_size=6 --base_lr=0.005 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth

# 0.05
CUDA_VISIBLE_DEVICES=1,2 python train.py --n_gpu=2 --split=train --scale=63 --batch_size=6 --base_lr=0.003 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth

# 0.01
CUDA_VISIBLE_DEVICES=1,2 python train.py --n_gpu=2 --split=train --scale=12 --batch_size=6 --base_lr=0.001 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
