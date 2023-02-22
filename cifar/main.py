'''Train CIFAR with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import sys
import os
import argparse
import logging
import time
import numpy as np
from torchvision import datasets, models
from models import *
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from models.vit_pytorch.vit import ViT_SP
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

def get_loaders(dsetname, bsize):
    print('==> Preparing ' + dsetname + ' data...')
    if dsetname == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        torchdset = torchvision.datasets.CIFAR10
    elif dsetname == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        torchdset = torchvision.datasets.CIFAR100
    else:
        print('==> Dataset not avaiable...')
        exit()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchdset(root='./data/'+dsetname+'/', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=4,drop_last=True)
    testset = torchdset(root='./data/'+dsetname+'/', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=4)

    return train_loader, test_loader

def get_model(modelname, Num_classes):
    if   modelname == 'v16':       net = VGG('VGG16',    Num_classes=Num_classes)
    elif modelname == 'r18':       net = ResNet18(       Num_classes=Num_classes)
    elif modelname == 'r34':       net = ResNet34(       Num_classes=Num_classes)
    elif modelname == 'r50':       net = ResNet50(       Num_classes=Num_classes)
    elif modelname == 'r101':      net = ResNet101(      Num_classes=Num_classes)
    elif modelname == 'rx29':      net = ResNeXt29_4x64d(Num_classes=Num_classes)
    elif modelname == 'dla':       net = DLA(            Num_classes=Num_classes)
    elif modelname == 'd121':      net = DenseNet121(    Num_classes=Num_classes)
    elif modelname == 'mobilenetv2':      net = MobileNetV2(    Num_classes=Num_classes)
    elif modelname == 'preres152' : net = PreActResNet152()
    elif modelname == 'ViT':       net = ViT_SP(img_size=32, patch_size = 4, num_classes=Num_classes, dim=192, mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,stochastic_depth=0.1)
    else:
        print('==> Network not found...')
        exit()
    return net

def get_optim(optim_name, learning_rate,wd, net,nesterov):
    
    if optim_name == 'adam':           	 optimizer = optim.Adam(    net.parameters(), lr=learning_rate,weight_decay=wd)
    elif optim_name == 'sgd':            	 optimizer = optim.SGD(           net.parameters(), lr=learning_rate, momentum=0.9,weight_decay=wd)
    elif optim_name == 'diffgrad':    	 optimizer = diffgrad(      net.parameters(), lr=learning_rate,weight_decay=wd)
    elif optim_name == 'tanangulargrad':  optimizer = tanangulargrad(net.parameters(), lr=learning_rate,weight_decay=wd)
    elif optim_name == 'signgrad':        	 optimizer = SignGrad(       net.parameters(), lr=learning_rate,weight_decay=wd)
    elif optim_name == 'adamp':         	 optimizer = AdamP(   net.parameters(), lr=learning_rate, weight_decay=wd)
    elif optim_name == 'signadamp':      	 optimizer = SignAdamP(   net.parameters(), lr=learning_rate, weight_decay=wd)
    elif optim_name == 'adamw':          	 optimizer = AdamW(   net.parameters(), lr=learning_rate,weight_decay = wd)
    elif optim_name == 'signadamw':        optimizer = SignAdamW(   net.parameters(), lr=learning_rate,weight_decay = wd)
    else:
        print('==> Optimizer not found...')
        exit()
    return optimizer

def train(train_loader, epoch, net, optimizer,scheduler_lr, criterion, writer, device='cuda', log_interval=320, logger=None):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % log_interval == 0:

            idx = batch_idx + epoch * (len(train_loader))
            writer.add_scalar('Loss/train_detail', train_loss/(batch_idx+1), idx)
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    train_loss/(batch_idx+1),
                )
            )
    acc=100.*correct/total
    logger.info('Training: Loss: {:.4f} | Acc: {:.4f}'.format(train_loss/(batch_idx+1),acc))
    writer.add_scalar('Loss/train', train_loss/(batch_idx+1), idx)
    writer.add_scalar('Accuracy/train', acc, epoch)
    
    return acc, train_loss/(batch_idx+1)


def test(test_loader, epoch, net, criterion, writer, device='cuda', logger=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    acc=100.*correct/total
    logger.info('Testing: Loss: {:.4f} | Acc: {:.4f}'.format(test_loss, acc))
    writer.add_scalar('Accuracy/test', acc, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    return acc, test_loss

def main(args,savename):
    log_dir = 'runs/'+savename
    logname = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    if not os.path.isdir('log/'+savename):
        os.mkdir('log/'+savename)
        os.mknod('log/'+savename+'/'+str(logname)+'.log')
    if not os.path.isdir('checkpoint/'+savename):
            os.mkdir('checkpoint/'+savename)
    logger = get_logger('log/'+savename+'/'+str(logname)+'.log')
    logger.info('Tensorboard: tensorboard --logdir={}'.format(log_dir))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_interval = 10240/args.bs
    with SummaryWriter(log_dir) as writer:
        # Random seed
        logger.info(args.manualSeed)
        if args.manualSeed is None:
            args.manualSeed = random.randint(1, 10000)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)
        if device == 'cuda':
            torch.cuda.manual_seed(args.manualSeed)
            torch.cuda.manual_seed_all(args.manualSeed)

        train_loader, test_loader = get_loaders(args.dataset, args.bs)

        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100

        net = get_model(args.model, num_classes)

        if device == 'cuda':
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        optimizer = get_optim(args.alg, args.lr,args.weight_decay, net,args.nesterov)
        criterion = nn.CrossEntropyLoss()
        if args.model == "r50":
            scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)  
        elif args.model == "ViT_SP":
            scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        if args.resume == "best":
            logger.info('==> Resuming from best checkpoint..')
            assert os.path.isdir('checkpoint/'+ savename), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/'+savename +'/best_checkpoint.t7')
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_lr.load_state_dict(checkpoint['lr_schedule'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1
        elif args.resume == "recent":
            logger.info('==> Resuming from recent checkpoint..')
            assert os.path.isdir('checkpoint/'+ savename), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/'+savename +'/recent_checkpoint.t7')
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_lr.load_state_dict(checkpoint['lr_schedule'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1
        else:
            best_acc = -1
            start_epoch = 0

        if args.cifar100pretrain :
            pretrain_model = torch.load(args.cifar100pretrain)
            model_dict = net.state_dict()
            if args.dataset == "cifar10":
                pretrained_dict = {k: v for k, v in pretrain_model["net"].items() if (k in model_dict and 'mlp_head.1' not in k )}
            elif args.dataset == "cifar100":
                pretrained_dict = {k: v for k, v in pretrain_model["net"].items() if (k in model_dict and 'pos_embedding' not in k and 'to_patch_embedding.1' not in k)}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        if args.miniimagepretrain :
            pretrain_model = torch.load(args.miniimagepretrain)
            model_dict = net.state_dict()
            if args.dataset == "cifar10":
                pretrained_dict = {k: v for k, v in pretrain_model["model"].items() if (k in model_dict and 'mlp_head.1' not in k and 'pos_embedding' not in k and 'to_patch_embedding.1' not in k)}
            elif args.dataset == "cifar100":
                pretrained_dict = {k: v for k, v in pretrain_model["model"].items() if (k in model_dict and 'pos_embedding' not in k and 'to_patch_embedding.1' not in k)}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)


        # args.epochs
        for epoch in range(start_epoch, args.epochs):
            
            train_acc, train_loss = train(train_loader, epoch, net, optimizer,scheduler_lr, criterion, writer,device=device, log_interval = log_interval, logger = logger)
            val_acc, val_loss = test(test_loader, epoch, net, criterion,writer, device=device,logger = logger)
            scheduler_lr.step()
            for name, param in net.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

            logger.info("lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            if epoch % 5==0:
                state = {
                    'net': net.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_schedule': scheduler_lr.state_dict()
                }
                torch.save(state, './checkpoint/'+savename+ '/recent_checkpoint.t7')
                logger.info('recent Saving..')

            if val_acc > best_acc:
                
                state = {
                    'net': net.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_schedule': scheduler_lr.state_dict(),
                }
                torch.save(state, './checkpoint/'+savename+'/best_checkpoint.t7')
                logger.info('best Saving..')
                best_acc = val_acc

        logger.info('Best Acc: {:.2f}'.format(best_acc))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--dataset', type=str, default='cifar10', \
                                choices=['cifar10', 'cifar100'], \
                                help='dataset (options: cifar10, cifar100)')
    parser.add_argument('--resume', '-r', type =str,default='no', help='resume from the best or recent checkpoint')
    parser.add_argument('--miniimagepretrain', type =str, help='path to miniimagenet pretrained model')
    parser.add_argument('--cifar100pretrain', type =str, help='path to cifar100 pretrained model')
    parser.add_argument('--nesterov', action = "store_true", help='True or False')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--model', type=str, default='r50', \
                                help='model (options: v16, r18, r34, r50, r101, rx29, dla, d121,vit,vit_for_smalldataset)')
    parser.add_argument('--bs', default=128, type=int, help='batchsize')
    parser.add_argument('--alg', type=str, default='adam', \
                                help='dataset (options: sgd, adam, adamw, diffgrad_v2, tanangulargrad, trygrad )')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--manualSeed', default=1111, type=int, help='random seed')
    args = parser.parse_args()
    savename = 'optim_{}_dataset_{}_model_{}_lr_{}_batchsize_{}_epochs_{}_manualSeed_{}'.format(args.alg,args.dataset,args.model,args.lr,args.bs,args.epochs,args.manualSeed)
    main(args,savename)