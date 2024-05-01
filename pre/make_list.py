import os
import random

    
def pre_train():
    mask_path = os.path.join(root_path, mode, 'masks')
    slices=[]
    cnt = 0
    perc = 0.1  
    for mask_name in sorted(os.listdir(mask_path)):
        slices.append(mask_name.split('.')[0])
        cnt += 1

    sample_num = int(cnt*perc)
    slices_sel = random.sample(slices, sample_num) 
    with open(f'{save_path}_{mode}.txt', 'w') as f:
        for sli in slices_sel:
            f.writelines(sli)
            f.writelines('\n')
    

def pre_test():
    slices=[]
    for m in mode:
        mask_path = os.path.join(root_path, m, 'masks')
        for mask_name in sorted(os.listdir(mask_path)):
            slices.append(mask_name.split('.')[0])
            
    with open(f'{save_path}_{mode[-1]}.txt', 'w') as f:
        for sli in slices:
            f.writelines(sli)
            f.writelines('\n')


if __name__ == "__main__":
    # root_path = '/home/yanzhonghao/data/ven/bhx_sammed_priori'
    # save_path = '/home/yanzhonghao/medical_sam/H-SAM/lists/lists_bhx'
    root_path = '/home/yanzhonghao/data/abdomen/sabs_sammed'
    save_path = '/home/yanzhonghao/medical_sam/H-SAM/lists/lists_sabs'

    # mode = 'train'
    mode = ['val', 'test']  

    # pre_train()
    pre_test()