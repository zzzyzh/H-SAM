# 0.1
python train.py --split=train --scale=127 --batch_size=16 --base_lr=0.005 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth

# 0.05
python train.py --split=train --scale=63 --batch_size=16 --base_lr=0.005 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth

# 0.01
python train.py --split=train --scale=12 --batch_size=16 --base_lr=0.005 --img_size=512 --warmup --AdamW --max_epochs=300 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
