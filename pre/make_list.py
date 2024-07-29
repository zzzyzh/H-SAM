import os
import random

    
def pre_train(perc=0.1):
    mask_path = os.path.join(root_path, 'train', 'masks')
    slices=[]
    cnt = 0
    for mask_name in sorted(os.listdir(mask_path)):
        slices.append(mask_name.split('.')[0])
        cnt += 1

    sample_num = int(cnt*perc)
    slices_sel = random.sample(slices, sample_num) 
    with open(f'{save_path}_{"train"}_{len(slices_sel)}.txt', 'w') as f:
        for sli in slices_sel:
            f.writelines(sli)
            f.writelines('\n')
    

def pre_val_test():
    for mode in ['val', 'test']:
        slices=[]
        mask_path = os.path.join(root_path, mode, 'masks')
        for mask_name in sorted(os.listdir(mask_path)):
            slices.append(mask_name.split('.')[0])
                
        with open(f'{save_path}_{mode}.txt', 'w') as f:
            for sli in slices:
                f.writelines(sli)
                f.writelines('\n')


if __name__ == "__main__":
    random.seed(2024)
    # root_path = '/home/yanzhonghao/data/ven/bhx_sammed'
    # save_path = '/home/yanzhonghao/medical_sam/H-SAM/lists/lists_bhx'
    root_path = '/home/yanzhonghao/data/abdomen/sabs_sammed'
    save_path = '/home/yanzhonghao/medical_sam/H-SAM/lists/lists_sabs'

    pre_train(perc=0.1)
    pre_train(perc=0.05)
    pre_train(perc=0.01)

    pre_val_test()
