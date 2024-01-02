import argparse

def parse_args():    
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg, cls, val_ad')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-exp_name', type=str, default='msa-3d-sam-btcv', help='net type')
    parser.add_argument('-sam_ckpt', type=str, default='./checkpoint/sam/sam_vit_b_01ec64.pth', help='sam checkpoint address')
    parser.add_argument('-cls', type=int, default=0, help='output class or not')

    # gpu
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-distributed', type=int, default=1, help='use multi GPU or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-multigpu_device', type=str, default='0,1', help='use which gpus')

    # pretrain
    parser.add_argument('-weights', type=str, default =0, help='the weights file you want to test')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')

    #prompt
    parser.add_argument('-prompt', type=str, default='single', help='prompt type:single, multi, box')

    # data
    parser.add_argument('-thd', type=bool, default=True, help='3d or not')
    parser.add_argument('-data_path', type=str, default='./data', help='The path of segmentation data')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-roi_size', type=int, default=96, help='resolution of roi')
    parser.add_argument('-chunk', type=int, default=96, help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4, help='sample pos and neg')

    #train
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-val_freq',type=int,default=50, help='interval between each validation')

    # evaluate
    parser.add_argument('-evl_chunk', type=int, default=None, help='evaluation chunk')

    opt = parser.parse_args()

    return opt
