
from __future__ import print_function

import argparse
import os
import time

import torch
import sys
torch.set_printoptions(threshold=sys.maxsize)

from sklearn.metrics import confusion_matrix
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from models.custom import *

from utils import  AverageMeter, accuracy


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
args = parser.parse_args()

# ------local config------ #
args.gpu_id = '0'

args.generate_CM = True

args.dataset = 'cifar100'
args.workers = 4
args.train_batch = 128
args.data_path = '/home/wzn/Datasets/Classification'
args.parapath = '../Experiments/LengthNormalization2/CIFAR100/cifar100_vgg19bn_wd5e-4_norm-none-2/model_best.pth.tar'

args.arch = 'vgg19_bn'
args.loss_type = 'softmax'         # softmax, cosine, theta
set_gl_variable(linear=LinearNorm, normlinear='21')

args.scale_change = False
args.scale_second_large = 4.0
args.scale_second_small = 16.0

args.CMsavepath = 'ConfusionMatrixToWeightMatrix/cifar100_vgg19bn.npy'

args.print_freq = 50
# ------local config------ #

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# cuda
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


def main():
    global args
    if args.generate_CM == True:
        generate_confusion_matrix()

def generate_confusion_matrix():
    global args
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        args.num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        args.num_classes = 100
    trainset = dataloader(root=args.data_path, train=True, download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=args.num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            depth=args.depth,
            block_name=args.block_name,
        )
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # 　load model parameters
    print('==>Load model parameters..')
    assert os.path.isfile(args.parapath), 'Error: no model parameter directory found!'
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    # Generate the Confusion Matrix
    target, pred = test(trainloader, model, use_cuda)
    CM = confusion_matrix(target, pred)
    np.save(args.CMsavepath, CM)

def generate_cosinesimilar_matrix():
    global args
    pass

def test(testloader, model, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    pred_list = []
    target_list = []

    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs, features = model(inputs, targets)
            outputs, cosine = outputs

            if args.loss_type == 'softmax':
                pred = outputs
            else:
                pred = cosine
            # measure accuracy
            prec1, prec5 = accuracy(pred.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            #　record the predict and target array for confusion matrix
            pred_list.extend(pred.argmax(dim=1).data.cpu().tolist())
            target_list.extend(targets.data.cpu().tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'   
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(testloader), batch_time=batch_time,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f}\t'
          ' * Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return (target_list, pred_list)


if __name__ == '__main__':
    main()

