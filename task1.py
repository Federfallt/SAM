import torchvision.transforms as transforms
import cfg
from evaluate import validation_sam
from utils import *

args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device = GPUdevice, distribution = args.distributed)

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

nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)

'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

epoch = 1
net.eval()
tol, (eiou, edice) = validation_sam(args, nice_test_loader, epoch, net)
logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
