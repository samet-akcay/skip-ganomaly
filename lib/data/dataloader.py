"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import sys
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
# from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms


##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    ##
    if opt.dataset in ["UCSD/ped1", "UCSD/ped2"]:
        # Folder dataset
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        from .datasets import UCSD
        dataset = {x: UCSD(root=os.path.join(opt.dataroot, x), transform=transform, split=x) \
                   for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}

    elif opt.dataset in ["UCSD.sw/ped1", "UCSD.sw/ped2"]:
    # elif t == 'txt':
    # elif opt.dataset.startswith('UCSD'):
        # Folder dataset
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        from .datasets import UCSDsw
        dataset = {x: UCSDsw(txt_img=os.path.join(opt.dataroot, x+'.txt'),
                             txt_lbl=os.path.join(opt.dataroot, 'label.txt'),
                             transform=transform, split=x) for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}

    if opt.dataset in ['cifar10']:
        from .datasets import get_cifar_anomaly_dataset
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)

        dataset['train'].train_data, dataset['train'].train_labels, \
        dataset['test'].test_data, dataset['test'].test_labels = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].train_data,
            trn_lbl=dataset['train'].train_labels,
            tst_img=dataset['test'].test_data,
            tst_lbl=dataset['test'].test_labels,
            abn_cls_idx=classes[opt.anomaly_class]
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

    elif opt.dataset in ['mnist']:
        from .datasets import get_mnist_anomaly_dataset
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        from torchvision.datasets import MNIST
        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].train_data, dataset['train'].train_labels, \
        dataset['test'].test_data, dataset['test'].test_labels = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].train_data,
            trn_lbl=dataset['train'].train_labels,
            tst_img=dataset['test'].test_data,
            tst_lbl=dataset['test'].test_labels,
            abn_cls_idx=opt.anomaly_class
        )

        # from .datasets import MNIST
        # dataset = {}
        # dataset['train'] = MNIST(opt, root='./data/mnist', split='train', transform=transform)
        # dataset['test']  = MNIST(opt, root='./data/mnist', split='test',  transform=transform)

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader
    elif opt.dataset in ['mnist2']:
        from torchvision.datasets import MNIST
        from .datasets import get_mnist2_anomaly_dataset
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].train_data, dataset['train'].train_labels, \
        dataset['test'].test_data, dataset['test'].test_labels = get_mnist2_anomaly_dataset(
            trn_img=dataset['train'].train_data,
            trn_lbl=dataset['train'].train_labels,
            tst_img=dataset['test'].test_data,
            tst_lbl=dataset['test'].test_labels,
            nrm_cls_idx=opt.anomaly_class,
            proportion=opt.proportion
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

    elif opt.dataset in ['UCSDped1', 'UCSDped2']:
        opt.dataroot = f'./data/UCSD_Anomaly_Dataset/{opt.dataset}'
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        from .datasets import UCSD
        dataset = {x: UCSD(root=os.path.join(opt.dataroot, x), transform=transform, split=x) \
                   for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

    elif opt.dataset in ['caltech256']:
        from .caltech256 import CALTECH256
        from .caltech256 import get_caltech256_anomaly_dataset
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        dataset= {}
        dataset['train'] = CALTECH256(root=opt.dataroot, transform=transform)
        dataset['test']  = CALTECH256(root=opt.dataroot, transform=transform)
        dataset['train'].samples, dataset['test'].samples = get_caltech256_anomaly_dataset(
            dir=dataset['train'].root, num_inliers=1
        )
        # dataset['train']

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

    else:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader
