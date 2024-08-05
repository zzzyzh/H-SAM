# 0.1
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=127 --train_time 20240801-1202 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --vis=True
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=127 --train_time 20240801-1202 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --volume=True
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=127 --train_time 20240801-1202 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --tsne=True

# 0.05
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=63 --train_time 20240801-1625 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=63 --train_time 20240801-1625 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --volume=True

# 0.01
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=12 --train_time 20240801-1923 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --num_classes=8 --scale=12 --train_time 20240801-1923 --list_dir=./lists/lists_sabs --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --volume=True
