import os
import random
import argparse
import logging
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from datasets.bp4d_loader import bp4d_load
from models.emonet_split import EmoNet
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_callbacks import CallBackEvaluation, CallBackLogging, CallBackModelCheckpoint
from utils.utils import lr_change
from utils.metrics import get_acc, get_f1

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='parameters for face action unit recognition network')
parser.add_argument('--batchsize', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--epochs', type=int, default=12, metavar='N', help='training epochs')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='5,8')
parser.add_argument('--data', type=str, default='bp4d', choices=['bp4d'])
parser.add_argument('--subset', type=int, default=1,choices=[1,2,3])
parser.add_argument('--model', type=str, default='EmoNet')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate')
parser.add_argument('--wd', type=float, default=0,help='weight decay')
parser.add_argument('--scale', type=float, default=260,help='scale in crop')
parser.add_argument('--rect', action='store_true', help='use rect to crop data')

def main():
    args = parser.parse_args()
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cur_path = os.path.abspath(os.curdir)

    if args.data=='bp4d':
        args.nclasses=12
    else:
        raise ValueError()

    args.output = cur_path.replace('aunet_train','Results')+'/AUNet/'+str(args.data)+'/'\
                  +str(args.model)+'lr'+str(args.lr)+'wd'+str(args.wd)+'_S_'+str(args.scale)+'_T_'+str(args.subset)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    log_root = logging.getLogger()
    init_logging(log_root, args.output)

    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_valid_noflip = transforms.Compose([transforms.ToTensor()])

    print('loading train set')
    train_data = bp4d_load(data_name=args.data, phase='train', subset=args.subset, transforms=transform_train,
                          scale=args.scale, rect=args.rect, seed=manualSeed)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    au_weights = torch.from_numpy(train_data.AU_weight).float().cuda()

    print('loading val set')
    val_data = bp4d_load(data_name=args.data, phase='test', subset=args.subset, transforms=transform_valid_noflip,scale=args.scale, rect=args.rect)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)



    if args.model == 'emonet':
        model = EmoNet(n_classes=args.nclasses)
    else:
        raise ValueError()

    model = nn.DataParallel(model).cuda()
    params = list(model.parameters())
    sub_params = [p for p in params if p.requires_grad]
    print('num of params', sum(p.numel() for p in sub_params))
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(sub_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    else:
        raise ValueError()

    total_step = int(len(train_data) / args.batchsize * args.epochs)
    callback_logging = CallBackLogging(len(train_loader)//4, total_step, args.batchsize, None)
    callback_checkpoint = CallBackModelCheckpoint(args.output)
    callback_validation = CallBackEvaluation(val_loader, None, subset='val')
    criterion = nn.BCEWithLogitsLoss(pos_weight=au_weights)
    # training
    global_step = 0
    losses = AverageMeter()
    acces = AverageMeter()
    f1es = AverageMeter()
    for epoch in range(args.epochs):
        model.train()

        # set fan in eval mode
        if args.model == 'emonet':
            model.module.feature.eval()

        for index, data in enumerate(train_loader):
            global_step += 1
            images = data['img'].cuda()
            label = data['label'].cuda()
            pred = model(images)
            loss = criterion(pred,label)
            if epoch==0 and index==1:
                torchvision.utils.save_image(images, '%s/samples.png' % args.output,normalize=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            pred = pred.data.cpu().float()
            label = label.data.cpu().float()
            f1_score = get_f1(label,pred)
            acc = get_acc(label, pred)

            losses.update(loss.detach().item(), 1)
            acces.update(acc.mean().detach().item(), batch_size)
            f1es.update(f1_score.mean().detach().item(), batch_size)
            callback_logging(global_step, losses, acces, f1es, epoch, optimizer)

        val_results = callback_validation(epoch, model)
        callback_checkpoint(epoch, model)
        lr_change(epoch+1,optimizer)

if __name__ == '__main__':
    main()