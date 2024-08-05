# 0.1
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=882 --train_time 20240801-2215 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=882 --train_time 20240801-2215 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --volume=True
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=882 --train_time 20240801-2215 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --tsne=True

# 0.05
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=441 --train_time 20240802-1111 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=441 --train_time 20240802-1111 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --volume=True

# 0.01
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=88 --train_time 20240802-2024 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth
CUDA_VISIBLE_DEVICES=0 python test.py --split=test --task=ven --dataset=bhx_sammed --num_classes=4 --scale=88 --train_time 20240802-2024 --list_dir=./lists/lists_bhx --img_size=512 --vit_name=vit_b --ckpt=../../data/experiments/weights/sam_vit_b_01ec64.pth --volume=True