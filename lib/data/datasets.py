"""
CREATE DATASETS
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

import torch.utils.data as data
import torch

from PIL import Image
import numpy as np
import os
import os.path

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

class CIFAR256(data.Dataset):
    """[summary]

    Args:
        data ([type]): [description]
    """
    def __init__(self, root, nz=100, transform=None, target_transform=None,
                 loader=default_loader, split="train"):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # self.classes = ['abn', 'nrm']
        # self.class_to_idx = {'abn': 1, 'nrm': 0}
        self.classes, self.class_to_idx = find_classes(root)
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


##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist2_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, proportion=0.1):
    """ Create mnist 2 anomaly dataset.

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [tensor] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

    # Get n percent of the abnormal samples.
    abn_tst_idx = abn_tst_idx[torch.randperm(len(abn_tst_idx))]
    abn_tst_idx = abn_tst_idx[:int(len(abn_tst_idx) * proportion)]


    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
