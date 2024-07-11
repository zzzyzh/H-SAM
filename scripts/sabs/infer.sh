# 0.1
CUDA_VISIBLE_DEVICES=1 python test.py --split=test --num_classes=8 --scale=127 --train_time 20240704-1611 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=1 python test.py --split=test --num_classes=8 --scale=127 --train_time 20240704-1611 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth --volume=True

# 0.05
CUDA_VISIBLE_DEVICES=1 python test.py --split=test --num_classes=8 --scale=63 --train_time 20240704-1912 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=1 python test.py --split=test --num_classes=8 --scale=63 --train_time 20240704-1912 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth --volume=True

# 0.01
CUDA_VISIBLE_DEVICES=1 python test.py --split=test --num_classes=8 --scale=12 --train_time 20240704-2142 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=1 python test.py --split=test --num_classes=8 --scale=12 --train_time 20240704-2142 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth --volume=True