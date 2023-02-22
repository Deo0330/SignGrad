import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import logging
import torchvision.models as models
from models.vit_pytorch.vit import ViT_SP
import math
import numpy as np

from torch.optim import lr_scheduler
from models.resnet_ws import l_resnet50, l_resnet18, l_resnet101
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

from optims.diffgrad_v2 import diffgrad
from optims.tanangulargrad import tanangulargrad
from optims.SignGrad import SignGrad
from optims.AdamP import AdamP
from optims.SignAdamP import SignAdamP
from optims.SignAdamW import SignAdamW
from optims.AdamW import AdamW



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_optim(optim_name, learning_rate, net):
    if   optim_name == 'sgd':            optimizer = optim.SGD(     net.parameters(), lr=learning_rate, momentum=0.9)
    elif optim_name == 'adam':           optimizer = optim.Adam(    net.parameters(), lr=learning_rate)
    elif optim_name == 'diffgrad':    optimizer = diffgrad(      net.parameters(), lr=learning_rate)
    elif optim_name == 'tanangulargrad': optimizer = tanangulargrad(net.parameters(), lr=learning_rate)
    elif optim_name == 'signgrad':        optimizer = SignGrad(       net.parameters(), lr=learning_rate)
    elif optim_name == 'adamw':          optimizer = AdamW(   net.parameters(), lr=learning_rate, weight_decay = 0.05)
    elif optim_name == 'signadamw':       optimizer = SignAdamW(   net.parameters(), lr=learning_rate, weight_decay = 0.05)
    elif optim_name == 'adamp':          optimizer = AdamP(   net.parameters(), lr=learning_rate)
    elif optim_name == 'signadamp':          optimizer = SignAdamP(   net.parameters(), lr=learning_rate)
    else:
        print('==> Optimizer not found...')
        exit()
    return optimizer

def get_model(modelname):
    num_classes=100
    if modelname=='r18':
        model = models.resnet18()
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif modelname=='r50':
        model = models.resnet50()
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif modelname=='r101':
        model = models.resnet101()
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif modelname=='r18ws':
      model = l_resnet18(num_classes=num_classes)
    elif modelname=='r50ws':
      model = l_resnet50(num_classes=num_classes)
    elif modelname=='r101ws':
      model = l_resnet101(num_classes=num_classes)
    elif modelname == 'vit':       model = ViT_SP(img_size=224, patch_size = 4, num_classes=num_classes, dim=192, mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,stochastic_depth=0.1)
    else:
        print('==> Network not found...')
        exit()

    for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.uniform_()
                m.bias.data.zero_()
    return model

def get_loaders(args):
    print('==> Preparing MINI-Imagenet data...')
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
         ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

def train(train_loader, model, criterion, optimizer, epoch, args,device='cuda', log_interval=320,logger=None):
    logger.info('\nEpoch: %d' % epoch)
    model.train()
    total = 0
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to('cuda'), target.to('cuda')

        output = model(inputs)
        loss = criterion(output, target)

        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        train_loss += loss.item()
        total += target.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            idx = batch_idx + epoch * (len(train_loader))
            # writer.add_scalar('Loss/train_detail', train_loss/(batch_idx+1), idx)
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    train_loss/(batch_idx+1),
                ))
    acc=100.*correct/total
    logger.info('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),acc))
    # writer.add_scalar('Loss/train', train_loss/(batch_idx+1), idx)
    # writer.add_scalar('Accuracy/train', acc, epoch)
    return acc, train_loss/(batch_idx+1)

def validate(val_loader, model, criterion,epoch, args, device='cuda',logger=None):
    model.eval()

    val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(inputs)
            loss = criterion(output, target)

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            val_loss +=loss.item()
        acc = 100.*correct/total
        logger.info('Testing: Loss: {:.4f} | Acc: {:.4f}'.format(val_loss/(batch_idx+1), acc))
    # writer.add_scalar('Accuracy/val', acc, epoch)
    # writer.add_scalar('Loss/val', val_loss/(batch_idx+1), epoch)
    return acc, loss

def main(args,savename):

    log_dir = 'runs/'+savename
    logname = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    if not os.path.isdir('log/'+savename):
                    os.mkdir('log/'+savename)
                    os.mknod('log/'+savename+'/'+str(logname)+'.log')
    logger = get_logger('log/'+savename+'/'+str(logname)+'.log')
    logger.info('Tensorboard: tensorboard --logdir={}'.format(log_dir))
    logger.info('start training!')
    log_interval = 10240/args.batch_size
    # with SummaryWriter(log_dir) as writer:

    args.arch = args.model
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(args.seed)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model = get_model(args.model)
    if device == 'cuda':
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optim(args.alg, args.lr, model)
    
    if args.model == "r50":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    elif args.model == "vit":
        exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    train_loader, val_loader = get_loaders(args)
        
    if args.resume == 'recent':
        logger.info('==> Resuming from recent checkpoint..')
        checkpoint = torch.load('./checkpoint/'+savename+'/recent.t7')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']+1
    elif args.resume == 'best':
        logger.info('==> Resuming from best checkpoint..')
        checkpoint = torch.load('./checkpoint/'+savename+'/best.t7')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']+1
    else:
        best_acc = -1
        start_epoch = 0

    if not os.path.isdir('checkpoint/'+savename):
                os.mkdir('checkpoint/'+savename)
    for epoch in range(start_epoch, args.epochs):

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args, device=device, log_interval = log_interval,logger = logger)
        exp_lr_scheduler.step()
        val_acc, val_loss = validate(val_loader, model, criterion,epoch,args, device=device,logger = logger)
        logger.info("lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))



        if epoch % 5==0:
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': exp_lr_scheduler.state_dict()
            }
            torch.save(state, './checkpoint/'+savename+'/recent.t7')
            logger.info('Saving..')

            # Save checkpoint.
        if val_acc > best_acc:
            
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule': exp_lr_scheduler.state_dict(),
            }
            torch.save(state, './checkpoint/'+savename+'/best.t7')
            logger.info('Best Saving..')
            best_acc = val_acc

    logger.info('Best Acc: {:.2f}'.format(best_acc))
    logger.info('finish training!')

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Mini-ImageNet Training')
    parser.add_argument('-b', '--batch_size', default=16, type=int,metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('--resume', default='no', type=str, metavar='PATH',help='recent or best')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
    parser.add_argument('--seed', default=1, type=int,help='seed for initializing training. ')
    parser.add_argument('--model', default='r50', type=str, help='model')
    parser.add_argument('--alg', default='adam', type=str, help='optimizer')
    parser.add_argument('--note', default='original', type=str, help='optim_type')
    args = parser.parse_args()
    savename = 'optim_{}_model_{}_lr_{}_batchsize_{}_workers_{}_epochs_{}_seed_{}_note_{}'.format(args.alg,args.model,args.lr,args.batch_size,args.workers,args.epochs,args.seed,args.note)
    main(args,savename)
