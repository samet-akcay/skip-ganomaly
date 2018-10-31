import torch.utils.data as data
import torch
import torchvision
from torchvision.datasets import DatasetFolder

from pathlib import Path
from PIL import Image
import os
import os.path
import random
import imageio
import numpy as np

# pylint: disable=E1101

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

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

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nz=100, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': target}

    def __setitem__(self, index, value):
        self.noise[index] = value

    def __len__(self):
        return len(self.imgs)

## - -
class UCSD(data.Dataset):
    """A dataloader to load UCSD anomaly detection dataset
       We follow the following data structure ::

        UCSD_Anomaly_Dataset/UCSDped1/test/abnormal/Test001_061.tif
        UCSD_Anomaly_Dataset/UCSDped1/test/abnormal/Test002_106.tif
        UCSD_Anomaly_Dataset/UCSDped1/test/abnormal/Test003_152.tif

        UCSD_Anomaly_Dataset/UCSDped1/test/normal/Test001_001.tif
        UCSD_Anomaly_Dataset/UCSDped1/test/normal/Test002_040.tif
        UCSD_Anomaly_Dataset/UCSDped1/test/normal/Test003_087.tif

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nz=100, transform=None, target_transform=None,
                 loader=default_loader, split="train"):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = ['abn', 'nrm']
        self.class_to_idx = {'abn': 1, 'nrm': 0}
        self.split = split
        self.imgs = self.make_dataset(self.root, self.class_to_idx)
        self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(list(set([os.path.splitext(f)[0] for f in fnames]))):
                    path_img = os.path.join(root, fname + ".tif")
                    if self.split == "train":
                        item = (path_img, class_to_idx[target])
                    elif self.split == "test":
                        if os.path.exists(os.path.join(root, fname + ".bmp")):
                            path_pixel_gt = os.path.join(root, fname + ".bmp")
                        else:
                            raise FileNotFoundError("Pixel GT not found in test dir.")
                        item = (path_img, class_to_idx[target], path_pixel_gt)
                    else:
                        raise IOError("Unknown split format." +
                                      "Only train and test splits are supported for UCSD dataset class.")
                    images.append(item)

        if self.split == 'test':
            images = [(os.path.split(i[0])[0], os.path.split(i[0])[1], i[1], i[2]) for i in images]
            images = sorted(images, key=lambda i: i[2])
            images = [(os.path.join(i[0], i[1]), i[2], i[3]) for i in images]

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.split == 'train':
            path, frame_gt = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            latentz = self.noise[index]
            # return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': frame_gt}
            return img, frame_gt
        else:
            path, frame_gt, path_pixel_gt = self.imgs[index]
            img = self.loader(path)
            latentz = self.noise[index]
            pixel_gt = self.loader(path_pixel_gt) #TODO need to convert this to tensor.
            if self.transform is not None:
                img = self.transform(img)
                pixel_gt = self.transform(pixel_gt)
            # return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': frame_gt, 'pixel_gt': pixel_gt}
            return img, frame_gt

    def __setitem__(self, index, value):
        self.noise[index] = value

    def __len__(self):
        return len(self.imgs)

## - -
class UCSDsw(data.Dataset):
    """A dataloader to load UCSDsw anomaly detection dataset
       We read filenames from a txt file with the following data structure ::

        UCSD_Anomaly_Dataset/UCSDped1/test/abn/Test001_061.png 1
        UCSD_Anomaly_Dataset/UCSDped1/test/abn/Test002_106.png 1
        UCSD_Anomaly_Dataset/UCSDped1/test/abn/Test003_152.png 1
        UCSD_Anomaly_Dataset/UCSDped1/test/nrm/Test001_001.png 0
        UCSD_Anomaly_Dataset/UCSDped1/test/nrm/Test002_040.png 0
        UCSD_Anomaly_Dataset/UCSDped1/test/nrm/Test003_087.png 0

    Args:
        root (string): Path to txt file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, txt_img, txt_lbl, nz=100, transform=None, target_transform=None,
                 loader=default_loader, split="train"):

        self.txt_img = txt_img
        self.txt_lbl = txt_lbl
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.split = split
        self.classes, self.class_to_idx = self.find_classes()
        self.imgs = self.make_dataset(self.txt_img)
        self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + txt_img + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def find_classes(self):
        """Find classes and class indexes of the dataset.

        Args:
            label_file (string): Txt file containing label info.

        Returns:
            [list]: List of classes and indexes.
        """

        classes = []
        with open(self.txt_lbl, 'r') as file:
            for line in file:
                classes.append(line.split()[0])
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, txt_file):
        """[summary]

        Args:
            txt_file ([type]): [description]
            split (string): Train or Test split

        Returns:
            [list]: List of images in the data split.
        """

        images = []
        with open(txt_file, 'r') as file:
            for line in file:
                fname, class_idx = line.split()
                if self.split == 'test':
                    path_to_pixel_gt = os.path.splitext(fname)[0] + '.bmp'
                    images.append((fname, int(class_idx), path_to_pixel_gt))
                else:
                    images.append((fname, int(class_idx)))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.split == 'train':
            path, frame_gt = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            latentz = self.noise[index]
            return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': frame_gt}
        else:
            path, frame_gt, path_pixel_gt = self.imgs[index]
            img = self.loader(path)
            latentz = self.noise[index]
            pixel_gt = self.loader(path_pixel_gt) #TODO need to convert this to tensor.
            if self.transform is not None:
                img = self.transform(img)
                pixel_gt = self.transform(pixel_gt)
            return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': frame_gt, 'pixel_gt': pixel_gt}

    def __setitem__(self, index, value):
        self.noise[index] = value

    def __len__(self):
        return len(self.imgs)



#  __ __  _ _  _  ___  ___
# |  \  \| \ || |/ __>|_ _|
# |     ||   || |\__ \ | |
# |_|_|_||_\_||_|<___/ |_|
#
class MNIST(DatasetFolder):
    def __init__(self, opt, root, split='train', 
                 loader=default_loader, extensions='.png', transform=None):
        DatasetFolder.__init__(self, root=root, 
                               loader=default_loader, 
                               extensions=extensions, 
                               transform=None)
        self.split = split
        self.root = root
        self.opt = opt
        self.classes = ['0', '1']
        self.class_to_idx = {'0': 0, '1': 1}
        self.transform = transform
        self.samples = self.create_mnist_anomaly_dataset()

    ##
    @staticmethod
    def get_mnist_dataset(root='./data'):
        dst = Path('./data/mnist')
        if not dst.exists():
            # Get training and test set.
            trn = torchvision.datasets.MNIST(root=root, train=True,  download=True)
            tst = torchvision.datasets.MNIST(root=root, train=False, download=True)

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

    ##
    def create_mnist_anomaly_dataset(self):
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

        # Get the dataset
        self.get_mnist_dataset()

        # Initial variables.
        digit = self.opt.anomaly_class
        version = self.opt.dataset
        seed = self.opt.manualseed
        proportion = self.opt.proportion
        split = self.split

        # Define normality.
        smp = self.samples
        img = [s[0] for s in smp]
        lbl = [s[1] for s in smp]
        lbl = np.array(lbl)

        if version == 'mnist':
            # If version 1
            nrm_idx = [i for i, v in enumerate(lbl) if v != digit]
            abn_idx = [i for i, v in enumerate(lbl) if v == digit]

            random.Random(seed).shuffle(nrm_idx)
            random.Random(seed).shuffle(abn_idx)

        elif version == 'mnist2':
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

        return trn_smp if split == 'train' else tst_smp

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
