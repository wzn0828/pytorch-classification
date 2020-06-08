import os
import easydict
import torch
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --------- local configs -------- #

args = easydict.EasyDict()

args.checkpoint_baseline = 'Experiments/AngularLoss/CIFAR100/cifar100_vgg19bn_norm-none_record-min-angle/model.pth.tar'
args.checkpoint_mma = 'Experiments/AngularLoss/CIFAR100/cifar100_vgg19bn_norm-none_record-min-angle_angular-mma-0.07/model.pth.tar'
args.checkpoint_orth = 'Experiments/AngularLoss/CIFAR100/cifar100_vgg19bn_norm-none_record-min-angle_angular-orthogonal-0.0001/model.pth.tar'
args.checkpoint_s0 = 'Experiments/AngularLoss/CIFAR100/cifar100_vgg19bn_norm-none_record-min-angle_angular-skernel-0-1.0-0.3/model.pth.tar'
args.checkpoint_s2 = 'Experiments/AngularLoss/CIFAR100/cifar100_vgg19bn_norm-none_record-min-angle_angular-skernel-2-1.0-0.3/model.pth.tar'

args.weightname = 'module.features.0.weight'
args.alpha = 1

# --------- local configs -------- #

def get_angles(weight):

    size = weight.size(0)
    # for convolution layers, flatten
    if weight.dim() > 2:
        weight = weight.view(size, -1)

    # 　compute minimum angle
    weight_ = F.normalize(weight, p=2, dim=1)
    product = torch.matmul(weight_, weight_.t())

    angles = torch.acos(product)[tuple(torch.triu_indices(size, size, 1))] / math.pi * 180

    return angles


def get_cosine_matrix(weight):

    size = weight.size(0)
    # for convolution layers, flatten
    if weight.dim() > 2:
        weight = weight.view(size, -1)

    # 　compute minimum angle
    weight_ = F.normalize(weight, p=2, dim=1)
    product = torch.matmul(weight_, weight_.t())

    return product


def plot_hist():

    angles_baseline = get_angles(state_dict_baseline[args.weightname]).cpu()
    angles_mma = get_angles(state_dict_mma[args.weightname]).cpu()
    angles_orth = get_angles(state_dict_orth[args.weightname]).cpu()
    angles_s0 = get_angles(state_dict_s0[args.weightname]).cpu()
    angles_s2 = get_angles(state_dict_s2[args.weightname]).cpu()
    plt.figure(1)
    ax1 = plt.subplot(5, 1, 1)
    ax2 = plt.subplot(5, 1, 2)
    ax3 = plt.subplot(5, 1, 3)
    ax4 = plt.subplot(5, 1, 4)
    ax5 = plt.subplot(5, 1, 5)
    plt.sca(ax1)
    plt.hist(angles_baseline, bins=180, range=(0, 180), facecolor='blue', edgecolor='black', alpha=args.alpha)
    plt.sca(ax2)
    plt.hist(angles_s0, bins=180, range=(0, 180), facecolor='yellow', edgecolor='black', alpha=args.alpha)
    plt.sca(ax3)
    plt.hist(angles_s2, bins=180, range=(0, 180), facecolor='grey', edgecolor='black', alpha=args.alpha)
    plt.sca(ax4)
    plt.hist(angles_orth, bins=180, range=(0, 180), facecolor='green', edgecolor='black', alpha=args.alpha)
    plt.sca(ax5)
    plt.hist(angles_mma, bins=180, range=(0, 180), facecolor='red', edgecolor='black', alpha=args.alpha)
    plt.show()


def plot_heatmap():

    cosine_baseline = get_cosine_matrix(state_dict_baseline[args.weightname]).cpu().numpy()
    cosine_mma = get_cosine_matrix(state_dict_mma[args.weightname]).cpu().numpy()
    cosine_orth = get_cosine_matrix(state_dict_orth[args.weightname]).cpu().numpy()
    cosine_s0 = get_cosine_matrix(state_dict_s0[args.weightname]).cpu().numpy()
    cosine_s2 = get_cosine_matrix(state_dict_s2[args.weightname]).cpu().numpy()

    mask = np.zeros_like(cosine_baseline)
    mask[np.triu_indices_from(mask)] = True

    threshold = 0.4
    with sns.axes_style('white'):
        # plt.figure(1)
        # sns.heatmap(cosine_baseline, cmap=('coolwarm'), xticklabels=10, yticklabels=10, mask=mask, square=True, vmin=-1, vmax=1, center=0)
        print('Baseline: number greater than {} is {}'.format(threshold, number_points(cosine_baseline, threshold)))
        print('MMA: number greater than {} is {}'.format(threshold, number_points(cosine_mma, threshold)))
        print('Orth: number greater than {} is {}'.format(threshold, number_points(cosine_orth, threshold)))
        print('S0: number greater than {} is {}'.format(threshold, number_points(cosine_s0, threshold)))

        # plt.figure(2)
        # sns.heatmap(cosine_mma, cmap=plt.cm.get_cmap('coolwarm'), xticklabels=10, yticklabels=10, mask=mask, square=True, vmin=-1, vmax=1, center=0)

        # plt.figure(3)
        # sns.heatmap(cosine_s2, cmap=plt.cm.get_cmap('coolwarm'), xticklabels=10, yticklabels=10, mask=mask, square=True, vmin=-1, vmax=1, center=0)

        # plt.figure(4)
        # sns.heatmap(cosine_s0, cmap=plt.cm.get_cmap('coolwarm'), xticklabels=10, yticklabels=10, mask=mask, square=True, vmin=-1, vmax=1, center=0)

        # plt.figure(5)
        # sns.heatmap(cosine_orth, cmap=plt.cm.get_cmap('coolwarm'), xticklabels=10, yticklabels=10, mask=mask, square=True, vmin=-1, vmax=1, center=0)


    plt.show()


def number_points(cosine, threshold):

    return (cosine[np.triu_indices_from(cosine, k=1)]>threshold).sum()


if __name__ == '__main__':

    # Load checkpoint.
    assert os.path.isfile(args.checkpoint_baseline), 'Error: no checkpoint_baseline directory found!'
    assert os.path.isfile(args.checkpoint_mma), 'Error: no checkpoint_mma directory found!'
    assert os.path.isfile(args.checkpoint_orth), 'Error: no checkpoint_orth directory found!'
    assert os.path.isfile(args.checkpoint_s0), 'Error: no checkpoint_s0 directory found!'
    assert os.path.isfile(args.checkpoint_s2), 'Error: no checkpoint_s2 directory found!'

    state_dict_baseline = torch.load(args.checkpoint_baseline)['state_dict']
    state_dict_mma = torch.load(args.checkpoint_mma)['state_dict']
    state_dict_orth = torch.load(args.checkpoint_orth)['state_dict']
    state_dict_s0 = torch.load(args.checkpoint_s0)['state_dict']
    state_dict_s2 = torch.load(args.checkpoint_s2)['state_dict']

    # plot_hist()
    plot_heatmap()

