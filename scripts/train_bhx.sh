CUDA_VISIBLE_DEVICES="3"  python train.py --split=train --task=ven --dataset=bhx_sammed_priori --num_classes=4 --list_dir=./lists/lists_bhx --batch_size=2 --base_lr=0.0025 --img_size=512 --warmup --AdamW --max_epochs=300 --stop_epoch=300 --vit_name=vit_b --ckpt=/home/yanzhonghao/data/experiments/weights/sam_vit_b_01ec64.pth