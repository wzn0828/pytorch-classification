
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

from utils import AverageMeter, accuracy


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
args = parser.parse_args()

# ------local config------ #
args.gpu_id = '0'

args.generate_CM = True

args.dataset = 'cifar100'
args.workers = 4
args.batch = 128
args.data_path = '/home/wzn/Datasets/Classification'
args.parapath = '../Experiments/FixWeight/cifar100_vgg19bn_wd5e-4_norm-l24/model_best.pth.tar'

args.arch = 'vgg19_bn'
args.loss_type = 'softmax'         # softmax, cosine, theta
set_gl_variable(linear=LinearNorm, normlinear='24')

args.scale_change = False
args.scale_second_large = 4.0
args.scale_second_small = 16.0

args.CMsavepath = '../FixedWeightMatrix/confusitionmatrix/cifar100_vgg19bn_norm-l24_test.npy'

args.generate_weight_bycosine = False
args.generate_weight_bytheta = False
args.min_theta = 30.
args.max_theta = 90.
args.cosineSavePath = '../ConfusionMatrixToWeightMatrix/cifar100_vgg19bn_test_cosine.npy'
args.weightSavePath = '../ConfusionMatrixToWeightMatrix/cifar100_vgg19bn_test_weight.npy'

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
    if args.generate_weight_bycosine == True:
        generate_weight_bycosine()
    if args.generate_weight_bytheta == True:
        generate_weight_bytheta()

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
    dataset = dataloader(root=args.data_path, train=False, download=True, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
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
    checkpoint = torch.load(args.parapath)
    model.load_state_dict(checkpoint['state_dict'])
    # Generate the Confusion Matrix
    target, pred = test(dataloader, model, use_cuda)
    CM = confusion_matrix(target, pred)
    np.save(args.CMsavepath, CM)


def generate_weight_bycosine():

    global args

    # load　confusion matrix
    CM = np.load(args.CMsavepath)
    CM = torch.tensor(CM).float()

    # normalization
    CM = CM + CM.t()
    diag = torch.diagonal(CM).unsqueeze(dim=1).float()
    CM = CM / diag

    diagOfCM = torch.diag(torch.diag(CM))
    CM_except_diag = CM - diagOfCM

    # mapping to cosine similarity:
    # except the diagonal elements,
    # the maximum mapped to s, which is a hyperparameter; and the minimum mapped to -1
    max_cosine = math.cos(args.min_theta / 180.0 * math.pi)
    min_cosine = math.cos(args.max_theta / 180.0 * math.pi)

    max_CM = CM_except_diag.max()
    min_CM = CM.min()

    cosine = (CM_except_diag-min_CM)/ (max_CM - min_CM) * (max_cosine - min_cosine) + min_cosine
    cosine = cosine - torch.diag(torch.diag(cosine)) + torch.eye(CM.size(0))
    cosine = cosine.clamp(-1, 1)

    # save
    # np.save(args.cosineSavePath, cosine)

    # eigendecomposition
    e, v = torch.symeig(cosine, eigenvectors=True)

    print(e)


def generate_weight_bytheta():

    global args

    # load　confusion matrix
    CM = np.load(args.CMsavepath)
    CM = torch.tensor(CM).float()

    # normalization
    CM = CM + CM.t()
    diag = torch.diagonal(CM).unsqueeze(dim=1).float()
    CM = CM / diag

    diagOfCM = torch.diag(torch.diag(CM))
    CM_except_diag = CM - diagOfCM

    # mapping to cosine similarity:
    # except the diagonal elements,
    # the maximum mapped to s, which is a hyperparameter; and the minimum mapped to -1
    # max_cosine = math.cos(args.min_theta / 180.0 * math.pi)
    # min_cosine = math.cos(args.max_theta / 180.0 * math.pi)

    max_CM = CM_except_diag.max()
    min_CM = CM.min()

    theta = (max_CM - CM_except_diag) / (max_CM-min_CM) * (args.max_theta-args.min_theta) + args.min_theta
    cosine = torch.cos(theta/180.*math.pi)

    cosine = torch.round(cosine*1000)/1000

    cosine = cosine - torch.diag(torch.diag(cosine)) + torch.eye(CM.size(0))
    cosine = cosine.clamp(-1, 1)

    # save
    np.save(args.cosineSavePath, cosine)

    # using svd, as eigendecomposition is not so precise
    u, s, v = torch.svd(cosine)
    print(s)

    p = (u + v) / 2.0
    weight = torch.matmul(p, s.sqrt().diag())
    weight = weight / weight.norm(p=2, dim=1, keepdim=True)

    # save
    np.save(args.weightSavePath, weight.numpy())

    aa = torch.matmul(weight, weight.t())
    diff = aa - cosine
    print(diff.max(), diff.min())

    print(weight.norm(dim=1))


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




    CM=torch.tensor(np.load('/home/wzn/PycharmProjects/pytorch-classification/FixedWeightMatrix/confusitionmatrix/cifar100_vgg19bn_norm-l24_test.npy'))
    CM=CM.cuda()
    CM.fill_diagonal_(100)



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

            newpred=[]
            for indice in pred.data.argmax(dim=1):
                newpred.append(CM[indice, :])
            pred = torch.stack(newpred, dim=0)







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

