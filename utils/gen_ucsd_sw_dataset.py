""" Generate UCSD Sliding Window Patches
"""

##
import os
import shutil

import argparse
import random
import time
import numpy as np
import cv2

##
def parse_arguments():
    """
    Arguments for sliding window data generation.
    """
    parser = argparse.ArgumentParser(description="Generate Patched Data")
    parser.add_argument("--src", default='./data/UCSDSample', help="Path to source dataset")
    parser.add_argument("--dst", default='./data/UCSDSample.sw', help="Path to target dataset")
    parser.add_argument("--crop_background", action='store_true', help="Crop white background.")
    parser.add_argument("--resize", default=256, type=int, help="Resize input image")
    parser.add_argument("--random_crop", default=0, type=int, help="Size of random crop")
    parser.add_argument("--stepSize", default=64, type=int, help="Move window stepSize pixels.")
    parser.add_argument("--winW", default=64, type=int, help="Width of the window.")
    parser.add_argument("--winH", default=64, type=int, help="Height of the window.")
    parser.add_argument("--pyramidScale", default=1, type=float, help="Scale to Gaussian pyramid")
    parser.add_argument("--imext", default='.tif', help='Image format')
    parser.add_argument("--gtext", default='.bmp', help='GT format')
    parser.add_argument("--outf", default='', help="folder to output images")
    parser.add_argument("--show", default=False, help="Show window on image.")

    return parser.parse_args()


##
class ImageTransforms:
    """
    Image Transformations including resize, crop white background, center crop and pyramid.
    """
    ##
    @staticmethod
    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        """ Resize image  """
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # Check if height and width is the same
        if width == height:
            # Then ignore aspect ratio
            dim = (width, height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized


    ##
    @staticmethod
    def crop_background(img_org):
        """ Crop White Background from image.

        Args:
            img_org (np.array): Input image.

        Returns:
            np.array: Output image.
        """
        # Gray Level Image
        img_gry = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

        # Binary Image
        (thresh, img_bin) = cv2.threshold(img_gry, 128, 255,
                                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Bounding Rectangle
        x, y, w, h = cv2.boundingRect(img_bin)
        cv2.rectangle(img_org, (x, y), (x+w, y+h), (0, 0, 0), 1)

        # Cropped Image
        img_crp = img_org[y: y+h, x: x+w]

        return img_crp

    ##
    @staticmethod
    def random_crop(img, size):
        """Random crop throughout the image.

        Args:
            img (np.array)      : Input image
            size (int, tuple)   : Size of the crop.

        Raises:
            IOError: Unknown size format.

        Returns:
            [type]: Cropped image.
        """
        if isinstance(size, int):
            th = size
            tw = size
        elif isinstance(size, tuple):
            th = size[0]
            tw = size[1]
        else:
            raise IOError('Unknown size format {}'.format(size))

        h, w = img.shape[:-1]

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return img[i:i+th, j:j+tw]

    ##
    def pyramid(self, image, scale=1.5, min_size=(32, 32)):
        """ Performs image pyramids to handle scaling issue.

        Args:
            image (np.array): Input image.
            scale (float, optional): Defaults to 1.
            min_size (tuple): Minimum window size to scale.

        Returns:
            [np.array]: Output pyramid.
        """

        # yield the original image
        if scale == 1:
            yield image
            return

        # keep looping over the pyramid
        while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            image = self.resize(image, width=w)

            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
                break

            # yield the next image in the pyramid
            yield image

##
class SWDataGenerator:
    """
    Generate patch based data via Sliding Window.
    """
    def __init__(self, args, transform=None):
        self.args = args

        self.transform = transform
        self.dataset = None
        self.split = None
        self.cls = None
        self.trn_list = {}
        self.tst_list = {}
        self.cls_idxs = {'nrm': '0', 'abn': '1'}

    def copy_src_dataset(self):
        """ Copy source dataset.
        """
        print(">> Copying the dataset...")
        shutil.copytree(src=self.args.src, dst=self.args.dst)
        print("   Done.")

    ##
    @staticmethod
    def show(image):
        """ Show image.
        Args:
            image (np.array): Image to show
        """
        cv2.imshow("Window", image)
        # cv2.waitKey(0)
        time.sleep(0.025)

    ##
    def draw_bbox(self, image, pt1, pt2, color=(0, 255, 0), thickness=2):
        """ Draw bounding box on image.
        Args:
            image (np.array): [description]
            pt1   (tuple): x,y coordinates of the upper-left corner.
            pt2   (tuple): x,y coordinates of the lower-left corner.
            color (tuple): Color of the line.
            thickness (int, optional): Defaults to 2. Line thickness.
        """

        # Draw bounding boxes.
        clone = image.copy()
        cv2.rectangle(clone, pt1, pt2, color, thickness)
        self.show(clone)

    ##
    def get_window(self, image):
        """Slide window across the image, and get window i
        Args:
            image (np.array): Input image.

        Returns:
            (np.array): Window patch from the image
        """
        for y in range(0, image.shape[0], self.args.stepSize):
            for x in range(0, image.shape[1], self.args.stepSize):
                # yield the current window
                yield (x, y, image[y:y + self.args.winH, x:x + self.args.winW])

    def generate_sw_patches(self):
        """ Generate sliding window patches for UCSD dataset.

        Raises:
            IOError: Unknown split format. Only train and test splits supported.
        """
        print(">> Generating sw patches...")
        datasets = [d for d in os.listdir(self.args.src) if os.path.isdir(os.path.join(self.args.src, d))]
        for dataset in datasets:
            print(f"   {dataset}")
            self.dataset = os.path.join(sw.args.src, dataset)
            self.trn_list[dataset] = []
            self.tst_list[dataset] = []
            ## - -
            # Create txt files to store filepaths
            # trn_list = open(os.path.join(self.args.dst, dataset, 'train.txt'), mode='w')
            # tst_list = open(os.path.join(self.args.dst, dataset, 'test.txt'), mode='w')

            # For each split in data.
            splits = [
                s for s in os.listdir(self.dataset) \
                if os.path.isdir(os.path.join(self.dataset, s))
            ]
            for self.split in splits:
                # For each class in split.
                for self.cls in sorted(os.listdir(os.path.join(self.dataset, self.split))):
                    # For each filename in class.
                    fnames = sorted(os.listdir(os.path.join(self.dataset, self.split, self.cls)))
                    for fname in fnames:
                        if fname.endswith(self.args.imext):
                            self.fname = os.path.splitext(fname)[0]

                            # Read original image, and resize.
                            im = cv2.imread(os.path.join(self.dataset, self.split, self.cls, self.fname + self.args.imext))
                            im = self.transform.resize(im, width=self.args.resize, height=self.args.resize)

                            # Read ground truth (gt) for the test set
                            if self.split == 'test':
                                gt = cv2.imread(os.path.join(self.dataset, self.split, self.cls, self.fname + self.args.gtext))
                                gt = self.transform.resize(gt, width=self.args.resize, height=self.args.resize)

                            num_roi = 0
                            for (x, y, imw) in self.get_window(image=im):
                                if imw.shape[0] != self.args.winH or imw.shape[1] != self.args.winW:
                                    continue
                                if self.args.show:
                                    self.draw_bbox(image=im, pt1=(x, y), pt2=(x + self.args.winW, y + self.args.winH))

                                # Save window to the right folder.
                                if self.split == 'train':
                                    basename = os.path.join(self.args.dst, dataset, self.split, self.cls)
                                    dirname  = self.fname +"_" +str(num_roi+1).zfill(3)+".png"
                                    cv2.imwrite(os.path.join(basename, dirname), imw)
                                    # trn_list.write(imw_fname + " " + cls_idxs[self.cls] + '\n')
                                    self.trn_list[dataset].append((basename, dirname, self.cls_idxs[self.cls]))

                                elif self.split == 'test':
                                    # Get the corresponding pixel-level gt for the window i
                                    gtw = gt[y:y + self.args.winH, x:x + self.args.winW]

                                    # If the class is abnormal and there is no anomaly in the window
                                    # then save it onto normal class.
                                    if self.cls == 'abn' and np.sum(gtw) == 0:
                                        basename = os.path.join(self.args.dst, dataset, self.split, 'nrm')
                                        dirnamei = self.fname +"_" +str(num_roi+1).zfill(3)+".png"
                                        dirnameg = self.fname +"_" +str(num_roi+1).zfill(3)+".bmp"

                                        cv2.imwrite(os.path.join(basename, dirnamei), imw)
                                        cv2.imwrite(os.path.join(basename, dirnameg), gtw)
                                        self.tst_list[dataset].append((basename, dirnamei, self.cls_idxs['nrm']))

                                    else:
                                        basename = os.path.join(self.args.dst, dataset, self.split, self.cls)
                                        dirnamei = self.fname +"_" +str(num_roi+1).zfill(3)+".png"
                                        dirnameg = self.fname +"_" +str(num_roi+1).zfill(3)+".bmp"

                                        cv2.imwrite(os.path.join(basename, dirnamei), imw)
                                        cv2.imwrite(os.path.join(basename, dirnameg), gtw)
                                        self.tst_list[dataset].append((basename, dirnamei, self.cls_idxs[self.cls]))
                                else:
                                    raise IOError('Unknown split. Only train and test supported.')
                                num_roi += 1

                            os.remove(os.path.join(self.args.dst, dataset, self.split, self.cls, self.fname + self.args.imext))
                            if self.split == 'test':
                                os.remove(os.path.join(self.args.dst, dataset, self.split, self.cls, self.fname + self.args.gtext))
            # Sort the filenames.
            self.trn_list[dataset] = sorted(self.trn_list[dataset], key=lambda i: i[1])
            self.tst_list[dataset] = sorted(self.tst_list[dataset], key=lambda i: i[1])
            
            # Write onto a txt file
            with open(os.path.join(sw.args.dst, dataset, 'train.txt'), 'w') as trn_txt:
                trn_txt.writelines(['{}/{} {}\n'.format(i[0], i[1], i[2]) for i in self.trn_list[dataset]])

            with open(os.path.join(sw.args.dst, dataset, 'test.txt'), 'w') as tst_txt:
                tst_txt.writelines(['{}/{} {}\n'.format(i[0], i[1], i[2]) for i in self.tst_list[dataset]])
            ## - -
        print("   Done.")

sw = SWDataGenerator(args=parse_arguments(), transform=ImageTransforms())
sw.copy_src_dataset()
sw.generate_sw_patches()