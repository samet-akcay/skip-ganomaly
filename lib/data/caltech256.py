import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
from random import shuffle

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
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

class CALTECH256(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=default_loader, extensions='.jpg', transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


# root = '/home/sam/Projects/gan/ganomaly.v2/data/caltech256/'
# dataset = {}
# trnset = DatasetFolder(root=root)
# tstset = DatasetFolder(root=root)
# dataset['train'] = DatasetFolder(root=root)
# dataset['test']  = DatasetFolder(root=root)

##
def get_caltech256_anomaly_dataset(dir, extensions='.jpg', num_inliers=1):
    """ Create an anomaly detection dataset from CALTECH256.

    Args:
        dir (str): data directory
        class_to_idx (dict): classes and indices.
        extensions (str, optional): Defaults to '.jpg'. File extensions

    Returns:
        [type]: [description]
    """

    dir = os.path.expanduser(dir)
    # i_cls = ['001.ak47', '003.backpack', '005.baseball-glove']  # Inlier  idx no
    i_cls = [i for i in os.listdir(dir) if os.path.isdir(os.path.join(dir, i))]
    shuffle(i_cls)
    i_cls = i_cls[:num_inliers]
    o_cls = '257.clutter' # Outlier idx no

    o_dirs = os.path.join(dir, o_cls)
    i_fnames = []
    i_labels = []

    for i_idx in sorted(i_cls):
        i_dir = os.path.join(dir, i_idx)
        i_fname = [os.path.join(i_dir, i) for i in os.listdir(i_dir) if i.endswith(extensions)]
        i_fname = i_fname[:150]
        i_fnames.append(i_fname)
        i_labels.append([0] * len(i_fname))
        # i_labels.append([class_to_idx[i_idx]] * len(i_fname))

    i_fnames = [item for i_fname in i_fnames for item in i_fname]
    i_labels = [item for i_label in i_labels for item in i_label]
    o_fnames = [os.path.join(o_dirs, i) for i in os.listdir(o_dirs) if i.endswith(extensions)]
    o_labels = [1] * len(o_fnames)
    # o_labels = [class_to_idx[o_cls]] * len(o_fnames)

    # Split the inliers into train and test set
    i_idxs = [i for i in range(len(i_fnames))]
    shuffle(i_idxs)
    i_trn_idx = i_idxs[: int(len(i_idxs) * 0.8)]
    i_tst_idx = i_idxs[int(len(i_idxs) * 0.8):]

    i_trn_fnames = [i_fnames[i] for i in i_trn_idx]
    i_trn_labels = [i_labels[i] for i in i_trn_idx]
    i_tst_fnames = [i_fnames[i] for i in i_tst_idx]
    i_tst_labels = [i_labels[i] for i in i_tst_idx]

    # Randomly sub-sample outliers from clutter class.
    o_tst_idx = [i for i in range(len(o_fnames))]
    shuffle(o_tst_idx)
    o_tst_idx = o_tst_idx[: len(i_tst_idx)]

    o_tst_fnames = [o_fnames[i] for i in o_tst_idx]
    o_tst_labels = [o_labels[i] for i in o_tst_idx]

    # Get training samples
    trn_smp = [(i_fnames[i], i_labels[i]) for i in i_trn_idx]

    # Get test samples.
    tst_fnames = i_tst_fnames + o_tst_fnames
    tst_labels = i_tst_labels + o_tst_labels
    tst_smp = [(tst_fnames[i], tst_labels[i]) for i in range(len(tst_fnames))]

    # Return train and test samples.
    return trn_smp, tst_smp
