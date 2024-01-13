import torch
import torch.nn as nn
from torch.autograd import Function
import os
import sys
import cfg
import time
import random
import logging
import numpy as np
from datetime import datetime
import dateutil.tz
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    set_track_meta,
)

args = cfg.parse_args()

class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)
        torch.cuda.is_available()
        if device != 'None':
            if len(device) == 1:
                self.gpu_list = [int(device)]
            else:
                self.gpu_list = list(map(int, device.split(',')))
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)

def get_network(args, device, net, use_gpu=True, distribution= 1):
    """ return given network
    """

    if net == 'sam':
        from segment_anything import SamPredictor, sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        net = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt).to(device.output_device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        if distribution:
            net = torch.nn.DataParallel(net, device_ids=device.gpu_list, output_device=device.output_device)           
            net = net.cuda(device=device.output_device)
        else:
            net = net.to(device=device.output_device)

    return net

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def get_decath_loader(args, device):
    train_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device.output_device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], strict_check=False, channel_dim='no_channel'),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device.output_device, track_meta=True),
        ]
    )

    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files

def generate_click_prompt(img, msk):
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    if args.prompt == 'single':
        for i in range(d):
            pt_list_s = []
            msk_list_s = []
            for j in range(b):
                msk_s = msk[j,:,:,i]
                indices = torch.nonzero(msk_s)
                if indices.size(0) == 0:
                    random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                    new_s = msk_s
                else:
                    random_index = random.choice(indices)
                    label = msk_s[random_index[0], random_index[1]]
                    new_s = torch.zeros_like(msk_s)
                    new_s = (msk_s == label).to(dtype = torch.float)
                pt_list_s.append(random_index)
                msk_list_s.append(new_s)
            pts = torch.stack(pt_list_s, dim=0)
            msks = torch.stack(msk_list_s, dim=0)
            pt_list.append(pts)
            msk_list.append(msks)
    elif args.prompt == 'multi':
        for i in range(d):
            pt_list_s = []
            msk_list_s = []
            for j in range(b):
                msk_s = msk[j,:,:,i]
                indices = torch.nonzero(msk_s)
                if indices.size(0) == 0:
                    random_index_1 = torch.randint(0, h, (2,)).to(device = msk.device)
                    random_index_2 = torch.randint(0, h, (2,)).to(device = msk.device)
                    random_index_3 = torch.randint(0, h, (2,)).to(device = msk.device)
                    new_s_1 = msk_s
                    new_s_2 = msk_s
                    new_s_3 = msk_s
                else:
                    random_index_1 = random.choice(indices)
                    random_index_2 = random.choice(indices)
                    random_index_3 = random.choice(indices)
                    label_1 = msk_s[random_index_1[0], random_index_1[1]]
                    label_2 = msk_s[random_index_2[0], random_index_2[1]]
                    label_3 = msk_s[random_index_3[0], random_index_3[1]]
                    new_s_1 = torch.zeros_like(msk_s)
                    new_s_2 = torch.zeros_like(msk_s)
                    new_s_3 = torch.zeros_like(msk_s)
                    new_s_1 = (msk_s == label_1).to(dtype = torch.float)
                    new_s_2 = (msk_s == label_2).to(dtype = torch.float)
                    new_s_3 = (msk_s == label_3).to(dtype = torch.float)
                pt_list_s.append(random_index_1)
                pt_list_s.append(random_index_2)
                pt_list_s.append(random_index_3)
                msk_list_s.append(new_s_1)
                msk_list_s.append(new_s_2)
                msk_list_s.append(new_s_3)
            pts = torch.stack(pt_list_s, dim=0)
            msks = torch.stack(msk_list_s, dim=0)
            pt_list.append(pts)
            msk_list.append(msks)
    elif args.prompt == 'box':
        for i in range(d):
            pt_list_s = []
            msk_list_s = []
            for j in range(b):
                msk_s = msk[j,:,:,i]
                msk_np = msk_s.cpu().numpy()
                if len(np.unique(msk_np)) > 1:
                    y_indices, x_indices = np.where(msk_np > 0)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min = max(0, x_min - np.random.randint(5, 20))
                    x_max = min(w, x_max + np.random.randint(5, 20))
                    y_min = max(0, y_min - np.random.randint(5, 20))
                    y_max = min(h, y_max + np.random.randint(5, 20))
                    bbox = torch.tensor([x_min, y_min, x_max, y_max]).to(device = msk.device)
                else:
                    bbox = torch.tensor([0, 0, h, h]).to(device = msk.device)
                new_s = msk_s
                pt_list_s.append(bbox)
                msk_list_s.append(new_s)            
            pts = torch.stack(pt_list_s, dim=0)
            msks = torch.stack(msk_list_s, dim=0)
            pt_list.append(pts)
            msk_list.append(msks)
    elif args.prompt == 'grid':
        points_per_side = 32
        offset = 1 / (2 * points_per_side)
        for i in range(d):
            pt_list_s = []
            msk_list_s = []
            for j in range(b):
                msk_s = msk[j,:,:,i]
                points_side_x = np.linspace(offset * w, (1 - offset) * w, points_per_side)
                points_side_y = np.linspace(offset * h, (1 - offset) * h, points_per_side)
                points_x = np.tile(points_side_x[None, :], (points_per_side, 1))
                points_y = np.tile(points_side_y[:, None], (1, points_per_side))
                points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
                for point in points:
                    pt = torch.tensor([point[0], point[1]]).to(device = msk.device)
                    new_s = msk_s[point[0], point[1]].to(dtype = torch.int)
                    pt_list_s.append(pt)
                    msk_list_s.append(new_s)
            pts = torch.stack(pt_list_s, dim=0)
            msks = torch.stack(msk_list_s, dim=0)
            pt_list.append(pts)
            msk_list.append(msks)


    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)
