import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
from utils import *
from monai.losses import DiceCELoss
from einops import rearrange

def train_sam(args, net: nn.Module, device, optimizer, train_loader, epoch, cls = 0):
    hard = 0
    epoch_loss = 0
    ind = 0

    net.train()
    optimizer.zero_grad()

    epoch_loss = 0

    loss_seg = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    loss_cls = nn.CrossEntropyLoss()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = device.output_device)
            masks = pack['label'].to(dtype = torch.float32, device = device.output_device)
            if 'pt' not in pack:
                if cls:
                    imgs, pt, masks, target = generate_click_prompt(imgs, masks)
                else:
                    imgs, pt, masks, _ = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            pt = rearrange(pt, 'b n d -> (b d) n')
            imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
            masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

            imgs = imgs.repeat(1,3,1,1)
            point_labels = torch.ones(imgs.size(0))

            imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
            masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device.output_device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device.output_device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
            imgs = imgs.to(dtype = mask_type,device = device.output_device)
            
            '''Train'''
            if args.distributed:
                with torch.no_grad():
                    imge= net.module.image_encoder(imgs)

                    se, de = net.module.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )

                if cls:
                    pred, _, output_class = net.module.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.module.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=True,
                    )
                else:
                    pred, _ = net.module.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.module.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )
            else:
                with torch.no_grad():
                    imge= net.image_encoder(imgs)

                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )

                if cls:
                    pred, _, output_class = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=True,
                    )
                else:  
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=False,
                    )

            if cls:
                loss = loss_seg(pred, masks) + loss_cls(output_class, target)
            else:
                loss = loss_seg(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.update()

    return loss

def validation_sam(args, net: nn.Module, device, val_loader, epoch, cls = 0):
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    loss_seg = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    loss_cls = nn.CrossEntropyLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = device.output_device)
            masksw = pack['label'].to(dtype = torch.float32, device = device.output_device)
            if 'pt' not in pack:
                if cls:
                    imgsw, ptw, masksw, targetw = generate_click_prompt(imgsw, masksw)
                else:
                    imgsw, ptw, masksw, _ = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                pt = ptw[:,:,buoy: buoy + evl_ch]

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch
                
                if args.prompt == 'single':
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

                    mask_type = torch.float32
                    ind += 1
                    b_size,c,w,h = imgs.size()
                    longsize = w if w >=h else h

                    if point_labels[0] != -1:
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device.output_device)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device.output_device)
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                        pt = (coords_torch, labels_torch)
                elif args.prompt == 'multi':
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

                    mask_type = torch.float32
                    ind += 1
                    b_size,c,w,h = imgs.size()
                    longsize = w if w >=h else h

                    if point_labels[0] != -1:
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device.output_device)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device.output_device)
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                        pt = (coords_torch, labels_torch)
                elif args.prompt == 'box':
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

                    mask_type = torch.float32
                    ind += 1
                elif args.prompt == 'grid':
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

                    mask_type = torch.float32
                    ind += 1
                    b_size,c,w,h = imgs.size()
                    longsize = w if w >=h else h

                    if point_labels[0] != -1:
                        point_coords = pt
                        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device.output_device)
                        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device.output_device)
                        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                        pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                imgs = imgs.to(dtype = mask_type,device = device.output_device)
                
                '''test'''
                with torch.no_grad():
                    if args.distributed:
                        imge= net.module.image_encoder(imgs)

                        if args.prompt == 'single':
                            se, de = net.module.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )
                        elif args.prompt == 'multi':
                            se, de = net.module.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )
                        elif args.prompt == 'box':
                            se, de = net.module.prompt_encoder(
                                points=None,
                                boxes=pt,
                                masks=None,
                            )
                        elif args.prompt == 'grid':
                            se, de = net.module.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )

                        if cls:
                            pred, _, output_class = net.module.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.module.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=True,
                            )
                        else:
                            pred, _ = net.module.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.module.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=False,
                            )
                    else:
                        imge= net.image_encoder(imgs)

                        if args.prompt == 'single':
                            se, de = net.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )
                        elif args.prompt == 'multi':
                            se, de = net.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )
                        elif args.prompt == 'box':
                            se, de = net.prompt_encoder(
                                points=None,
                                boxes=pt,
                                masks=None,
                            )
                        elif args.prompt == 'grid':
                            se, de = net.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )

                        if cls:
                            pred, _, output_class = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=True,
                            )
                        else:
                            pred, _ = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=False,
                            )

                    if cls:
                        tot += loss_seg(pred, masks) + loss_cls(output_class, targetw)
                    else:
                        tot += loss_seg(pred, masks)                    

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])
