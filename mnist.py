""" MNIST
"""

##
import torch
import numpy as np
from torchvision.datasets import MNIST
from pathlib import Path
import imageio
import random
from random import shuffle


from torchvision.datasets import DatasetFolder

# pylint: disable=E1101

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

##
def get_mnist_dataset(root='./data'):
    dst = Path('./data/mnist')
    if not dst.exists():
        # Get training and test set.
        trn = MNIST(root=root, train=True,  download=True)
        tst = MNIST(root=root, train=False, download=True)

        # Merge training and test set.
        img = torch.cat((trn.train_data, tst.test_data), 0)
        lbl = torch.cat((trn.train_labels, tst.test_labels), 0) 

        # Create directories.
        print(">> Creating directories.")
        dst.mkdir()
        [(dst / str(c)).mkdir() for c in range(10) if not (dst / str(c)).exists()]
        print("   Done.")

        # Save images.
        print(">> Saving...")
        [imageio.imwrite(dst/f"{lbl[i]}"/f"{i:05}.png", img[i].numpy()) for i in range(len(img))]
        print("   Done.")

class MyMNIST(DatasetFolder):
    def __init__(self, root, loader=default_loader, extensions='.png'):
        DatasetFolder.__init__(self, root=root, loader=default_loader, extensions=extensions)
        self.root = root

m = MyMNIST(root='./data/mnist')

def create_mnist_anomaly_dataset(samples, digit=0, version='v1', split='train', seed=0, proportion=0.1):
    """ Create MNIST anomaly dataset.
    
    Arguments:
        samples {[type]} -- [description]
    
    Keyword Arguments:
        digit {int}         -- [description] (default: {0})
        version {str}       -- [description] (default: {'v1'})
        split {str}         -- [description] (default: {'train'})
        seed {int}          -- [description] (default: {0})
        proportion {float}  -- [description] (default: {0.1})
    
    Raises:
        NotImplementedError -- [description]
    
    Returns:
        [type] -- [description]
    """

    # Define normality.
    smp = samples
    img = [s[0] for s in smp]
    lbl = [s[1] for s in smp]
    lbl = np.array(lbl)

    if version == 'v1':
        # If version 1
        nrm_idx = [i for i, v in enumerate(lbl) if v != digit]
        abn_idx = [i for i, v in enumerate(lbl) if v == digit]

        random.Random(seed).shuffle(nrm_idx)
        random.Random(seed).shuffle(abn_idx)

    elif version == 'v2':
        # If version 2
        nrm_idx = [i for i, v in enumerate(lbl) if v == digit]
        abn_idx = [i for i, v in enumerate(lbl) if v != digit]

        random.Random(seed).shuffle(nrm_idx)
        random.Random(seed).shuffle(abn_idx)    

        abn_idx = abn_idx[:int(len(abn_idx) * proportion)]

    else:
        raise NotImplementedError('v1 | v2 only.')    

    # Split normal into train and test.
    trn_nrm_idx = nrm_idx[:int(len(nrm_idx) * 0.8)]
    tst_nrm_idx = nrm_idx[int(len(nrm_idx) * 0.8):]

    tst_abn_idx = abn_idx

    trn_idx = trn_nrm_idx
    tst_idx = tst_nrm_idx + tst_abn_idx

    # Normal and abnormal image and labels.
    trn_img = [img[i] for i in trn_idx]
    tst_img = [img[i] for i in tst_idx]

    trn_lbl = lbl[trn_idx]
    trn_lbl[:] = 0

    tst_nrm_img = [img[i] for i in tst_nrm_idx]
    tst_abn_img = [img[i] for i in tst_abn_idx]

    tst_nrm_lbl = lbl[tst_nrm_idx]
    tst_abn_lbl = lbl[tst_abn_idx]

    # New Labels: Normal (0), Abnormal (1)
    tst_nrm_lbl[:] = 0
    tst_abn_lbl[:] = 1

    tst_nrm_lbl = tst_nrm_lbl.tolist()
    tst_abn_lbl = tst_abn_lbl.tolist()

    # 
    tst_img = tst_nrm_img + tst_abn_img
    tst_lbl = tst_nrm_lbl + tst_abn_lbl

    # Train and Test samples in tuple.
    trn_smp = [(trn_img[i], trn_lbl[i]) for i, v in enumerate(trn_img)]
    tst_smp = [(tst_img[i], tst_lbl[i]) for i, v in enumerate(tst_img)]

    return trn_smp if split=='train' else tst_smp
