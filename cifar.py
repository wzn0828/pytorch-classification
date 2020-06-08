'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import sys
torch.set_printoptions(threshold=sys.maxsize)

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from models.custom import *
import models.custom as custom

from utils.misc import add_summary_value
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
# from libs import InPlaceABNSync as libs_IABNS

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# ------local config------ #
args.gpu_id = '0'

args.dataset = 'cifar100'
args.arch = 'vgg19'
args.lr = 0.01
set_gl_variable(linear=LinearNorm, normlinear=None)
args.checkpoint = 'Experiments/AngularLoss/CIFAR100/cifar100_vgg19_norm-none_angular-all-0.07'
args.record_min_angle = True

args.angular_loss_classify = True
args.angular_loss_hidden = True
args.angular_loss_weight = 0.07
args.angular_loss_type = 'mma'      # mma, cosine, s-kernel, orthogonal
args.loss_skernel_s = 0
args.loss_skernel_ratio = 0.3

args.loss_type = 'softmax'         # softmax, cosine, theta
args.loss_margin = 0               # only effective when loss is cosine or theta
args.loss_only_margin_right = False

args.classify_weight_load = False
args.classify_weight_path = 'FixedWeightMatrix/loss-hpn/final-512d-10c.npy'

args.margin_change = False
args.margin_change_epoch = 122
args.margin_second = 0.1

args.scale_change = False
args.scale_change_epoch = 130
args.scale_second_large = 16.0
args.scale_second_small = 4.0

args.ring_loss = False
args.normlosstype = 'SmoothL1'     # 'L2',  'SmoothL1', 'SAFN' , 'L1'
args.feature_radius = 16.0
args.weight_L2norm = 0.01
args.rl_change = True
args.rl_change_epoch = 122
args.rl_second_radius = 4.0
args.rl_second_weight = 0.02

args.train_batch = 128
args.schedule = [81, 122]
args.epochs = 164

args.tensorboard_paras = ['.g', '.v', '.lens']

args.data_path = '/home/wzn/Datasets/Classification'

args.manualSeed = 123
args.print_freq = 50
# ------local config------ #

global tb_summary_writer
tb_summary_writer = tb and tb.SummaryWriter(args.checkpoint)

state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '9006'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed, for deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
# still need to set the work_init_fn to random.seed in train_dataloader, if multi numworkers

best_acc = 0  # best test accuracy
batch_iters = 0  # record　the iterations of batches　

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        args.num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        args.num_classes = 100


    trainset = dataloader(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, worker_init_fn=random.seed)

    testset = dataloader(root=args.data_path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

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
                    block_name=args.block_name,
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

    # load classify weight
    if args.classify_weight_load:
        model.classifier.weight.data = torch.tensor(np.load(args.classify_weight_path))

    if args.classify_weight_load == False and custom._normlinear in ['24', '27']:
        model.classifier.weight.data.fill_(-1.)
        model.classifier.weight.data.fill_diagonal_(args.num_classes-1)

    # initialize the g of weight normalization
    if custom._normlinear and custom._normlinear in ['21', '22', '23', '24', '25', '26', '27']:
        weight = model.classifier.weight.data
        weight_norm = weight.norm(dim=1, keepdim=True)
        model.classifier.weight.data = weight / weight_norm

    model = torch.nn.DataParallel(model).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create loss function
    if args.loss_type == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = similarity_loss

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = args.dataset + '-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.schedule, gamma=args.gamma, last_epoch=start_epoch-1)

    # if args.arch in ['resnet'] and args.depth >= 110:
    # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
    # then switch back. In this implementation it will correspond for first epoch.
    if not args.resume:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        # print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        # return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        state['lr'] = optimizer.param_groups[0]['lr']
        add_summary_value(tb_summary_writer, 'learning_rate', state['lr'], epoch + 1)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        if args.scale_change and epoch > args.scale_change_epoch:
            if custom._normlinear=='23':
                model.module.classifier.scale.data.fill_(args.scale_second_small)
            else:
                model.module.classifier.scale_small = args.scale_second_small
                model.module.classifier.scale_large = args.scale_second_large

        if args.margin_change and epoch > args.margin_change_epoch:
            model.module.classifier.m = args.margin_second

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch+1, use_cuda)

        lr_scheduler.step(epoch+1)

        test_loss, test_acc = test(testloader, model, criterion, epoch+1, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    global batch_iters
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_norm = AverageMeter()
    losses_classify_angular = AverageMeter()
    losses_hidden_angular = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, features = model(inputs)
        outputs, cosine = outputs
        if args.loss_type == 'softmax':
            pred = outputs
        else:
            pred = cosine
        loss = criterion(pred, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # larger norm more transferable: SAFN
        if args.ring_loss:
            if args.rl_change and epoch > args.rl_change_epoch:
                args.feature_radius = args.rl_second_radius
                args.weight_L2norm = args.rl_second_weight
            feature_L2norm_loss = get_L2norm_loss_self_driven(features)
            losses_norm.update(feature_L2norm_loss.item(), inputs.size(0))
            loss = loss + feature_L2norm_loss

        #　angular loss
        if args.angular_loss_classify:
            angular_loss_classify = get_angular_loss(model.module.classifier.weight)
            losses_classify_angular.update(angular_loss_classify.item())
            if args.angular_loss_weight != 0:
                loss = loss + args.angular_loss_weight * angular_loss_classify

        if args.angular_loss_hidden:
            for name, m in model.module.named_modules():
                # if isinstance(m, custom.Con2d_Class):
                if name is 'classifier':
                    continue
                elif isinstance(m, (custom.Linear_Class, custom.Con2d_Class)):
                    angular_loss_hidden = get_angular_loss(m.weight)
                    losses_hidden_angular.update(angular_loss_hidden.item())
                    if args.angular_loss_weight != 0:
                        loss = loss + args.angular_loss_weight * angular_loss_hidden

        if args.record_min_angle == True and batch_iters % args.print_freq == 0:
            for name, m in model.module.named_modules():
                if isinstance(m, (custom.Linear_Class, custom.Con2d_Class)):
                    min_angle = get_min_angle(m.weight)
                    add_summary_value(tb_summary_writer, 'Min_angle/' + name, min_angle, batch_iters)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, batch_idx, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

        batch_iters += 1

    add_summary_value(tb_summary_writer, 'Loss/train', losses.avg, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/Loss_norm_train', losses_norm.avg, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/Angular_Loss_Classify', losses_classify_angular.avg, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/Angular_Loss_Hidden', losses_hidden_angular.avg, epoch)
    add_summary_value(tb_summary_writer, 'Top1/train', top1.avg, epoch)
    add_summary_value(tb_summary_writer, 'Top5/train', top5.avg, epoch)
    add_summary_value(tb_summary_writer, 'Logits[0][0]', outputs[0][0], epoch)
    if model.module.classifier.bias is not None:
        add_summary_value(tb_summary_writer, 'fc_bias[0]', model.module.classifier.bias[0], epoch)
        tb_summary_writer.add_histogram('Hists/fc_bias', model.module.classifier.bias, epoch)

    # compute the cosine of classify layer
    tb_summary_writer.add_histogram('Cosine/', cosine, epoch)

    # the length of features
    feature_norms = features.data.norm(p=2, dim=1).cpu()
    tb_summary_writer.add_histogram('Hists/Feature_Length/train', feature_norms, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/Feature_Length/train', feature_norms.mean(), epoch)

    x_lens = model.module.classifier.x[0].data.norm(p=2, dim=1, keepdim=True)
    tb_summary_writer.add_histogram('Hists/X_Lens/train', x_lens, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/X_Lens/train', x_lens.mean().item(), epoch)

    # tensorboard weights cosine and weight norm in classification layer
    mean_cosine, max_cosine, weight_norm = compute_weight_cosine(model)
    add_summary_value(tb_summary_writer, 'Scalars/Weights_Mean_Cosine', mean_cosine, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/Weights_Max_Cosine', max_cosine, epoch)
    add_summary_value(tb_summary_writer, 'Scalars/Weight_Length_Mean', weight_norm.mean(), epoch)
    tb_summary_writer.add_histogram('Hists/Weight_Norm', weight_norm, epoch)

    if args.tensorboard_paras is not None:
        for name, para in model.module.named_parameters():
            if para is not None:
                for para_name in args.tensorboard_paras:
                    if para_name in name:
                        # if para_name == '.g':
                        #     para.data = torch.abs(para.data)
                        tb_summary_writer.add_histogram('Weights/' + name.replace('.', '/'), para, epoch)
                        add_summary_value(tb_summary_writer, 'Scalars/' + name.replace('.', '/'), para.mean(), epoch)
                        if para.grad is not None:
                            tb_summary_writer.add_histogram('Grads/' + name.replace('.', '/'), para.grad, epoch)
                        # last several epochs, tensorboard clearly
                        if epoch > args.epochs-10:
                            tb_summary_writer.add_histogram('Weights_10/' + name.replace('.', '/'), para, epoch)
                            add_summary_value(tb_summary_writer, 'Scalars_10/' + name.replace('.', '/'), para.mean(),
                                              epoch)
                            if para.grad is not None:
                                tb_summary_writer.add_histogram('Grads_10/' + name.replace('.', '/'), para.grad, epoch)

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        # bar = Bar('Processing', max=len(testloader))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs, features = model(inputs)
            outputs, cosine = outputs

            if args.loss_type == 'softmax':
                pred = outputs
            else:
                pred = cosine
            loss = criterion(pred, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(pred.data, targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(testloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f}\t'
          ' * Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if epoch is not None:
        add_summary_value(tb_summary_writer, 'Loss/test', losses.avg, epoch)
        add_summary_value(tb_summary_writer, 'Top1/test', top1.avg, epoch)
        add_summary_value(tb_summary_writer, 'Top5/test', top5.avg, epoch)
        # the length of features
        tb_summary_writer.add_histogram('Hists/Feature_Length/test', features.norm(dim=1), epoch)
        add_summary_value(tb_summary_writer, 'Scalars/Feature_Length/test', features.norm(dim=1).mean(), epoch)

        x_lens = model.module.classifier.x[0].norm(p=2, dim=1, keepdim=True)
        tb_summary_writer.add_histogram('Hists/X_Lens/test', x_lens, epoch)
        add_summary_value(tb_summary_writer, 'Scalars/X_Lens/test', x_lens.mean().item(), epoch)

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='model.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def get_L2norm_loss_self_driven(x):
    x_len = x.norm(p=2, dim=1, keepdim=True)
    if args.normlosstype == 'SAFN':
        radius = x_len.detach()
        assert radius.requires_grad == False
        radius = radius + 1.0
        l = 0.5*((x_len - radius) ** 2).mean()
    elif args.normlosstype == 'L2':
        diff = x_len - args.feature_radius
        l = 0.5 * torch.pow(diff, 2).mean()
    elif args.normlosstype == 'SmoothL1':
        l = F.smooth_l1_loss(x_len, torch.full_like(x_len, args.feature_radius))
    elif args.normlosstype == 'L1':
        l = F.l1_loss(x_len, torch.full_like(x_len, args.feature_radius))

    return args.weight_L2norm * l


def get_min_angle(weight):
    # for convolution layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # 　compute minimum angle
    weight_ = F.normalize(weight, p=2, dim=1)
    product = torch.matmul(weight_, weight_.t())
    min_angle = torch.acos((product - 2. * torch.diag(torch.diag(product))).max()).item() / math.pi * 180

    return min_angle


def get_angular_loss(weight):
    '''
    :param weight: parameter of model, out_features *　in_features
    :return: angular loss
    '''
    # for convolution layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    if args.angular_loss_type == 's-kernel':
        total = weight.size(0)
        num = math.ceil(total * args.loss_skernel_ratio)
        if num == 1:
            print('num is one！!!')
        index = torch.randperm(total)[:num]
        weight = weight[index]

    # Dot product of normalized prototypes is cosine similarity.
    weight_ = F.normalize(weight, p=2, dim=1)

    if args.angular_loss_type in ['mma', 'cosine', 'orthogonal']:
        product = torch.matmul(weight_, weight_.t())

    if args.angular_loss_type == 'mma':
        # Remove diagnonal from loss
        product_ = product - 2. * torch.diag(torch.diag(product))
        # Maxmize the minimum theta.
        loss = -torch.acos(product_.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    elif args.angular_loss_type == 'cosine':
        # Remove diagnonal from loss
        product_ = product - 2. * torch.diag(torch.diag(product))
        loss = product_.max(dim=1)[0].clamp(-0.99999, 0.99999).mean()
    elif args.angular_loss_type == 's-kernel':
        size = weight_.size(0)
        diff = weight_.view(size, 1, -1) - weight_.view(1, size, -1)
        norm = diff.norm(p=2, dim=-1)[tuple(torch.triu_indices(size, size, 1))]

        # diff = []
        # for i in range(weight_.size(0)-1):
        #     diff.append(weight_[i] - weight_[i+1:])
        # norm = torch.cat(diff).norm(p=2, dim=1)

        if args.loss_skernel_s > 0:
            loss = norm.pow(-args.loss_skernel_s).mean()
        elif args.loss_skernel_s == 0:
            loss = -torch.log(norm).mean()

    elif args.angular_loss_type == 'orthogonal':
        product_ = product - torch.diag(torch.diag(product))
        loss = torch.pow(product_.norm(), 2) * 0.5
    else:
         print('Unsupported angular loss type！')

    return loss


def compute_weight_cosine(model):
    weight = model.module.classifier.weight.data.cpu()
    weight_norm = weight.norm(dim=1, keepdim=True)
    weight = weight / weight_norm

    cosine_similarity = torch.matmul(weight, weight.t()).tril(diagonal=-1)
    mean_cosine_similarity = cosine_similarity.sum() / ((weight.size(0) * (weight.size(0) - 1.0)) / 2.0)

    return mean_cosine_similarity, cosine_similarity.max(), weight_norm

def similarity_loss(cosine, label):
    nB = len(cosine)  # Batchsize

    # pick the labeled cos theta
    idx_ = torch.arange(0, nB, dtype=torch.long)
    labeled_cos = cosine[idx_, label]  # B

    if args.loss_margin != 0:
        labeled_theta = torch.acos(labeled_cos)  # B

        if args.loss_only_margin_right == True:
            loss_margin = args.loss_margin * (cosine.argmax(dim=1) == label).to(torch.float)
        else:
            loss_margin = args.loss_margin

        labeled_theta += loss_margin
        labeled_theta.clamp_(max=math.pi)

        if args.loss_type == 'cosine':
            labeled_cos = torch.cos(labeled_theta)

    if args.loss_type == 'cosine':
        return (labeled_cos - 1.0).pow(2).mean()

    elif args.loss_type == 'theta':
        if args.loss_margin == 0:
            labeled_theta = torch.acos(labeled_cos)  # B

        return labeled_theta.pow(2).mean()

    else:
        return 0

if __name__ == '__main__':
    main()
