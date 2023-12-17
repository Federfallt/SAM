import torchvision.transforms as transforms
import cfg
import function
from utils import *

args = cfg.parse_args()

device = GpuDataParallel()
if args.distributed:
    device.set_device(args.multigpu_device)
else:
    device.set_device(args.gpu_device)

net = get_network(args, device, args.net, use_gpu=args.gpu, distribution = args.distributed)

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args, device)

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

epoch = 1
net.eval()
tol, (eiou, edice) = function.validation_sam(args, net, device, nice_test_loader, epoch)
logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
