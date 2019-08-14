""" BaseModel
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, data):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.data = data
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    ##
    def seed(self, seed_value):
        """ Seed 

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_lat', self.err_g_lat.item())])

        return errors

    ##
    def reinit_d(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch:int, is_best:bool=False):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(
            self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/netG_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_{epoch}.pth")

    def load_weights(self, epoch=None, is_best:bool=False, path=None):
        """ Load pre-trained weights of NetG and NetD

        Keyword Arguments:
            epoch {int}     -- Epoch to be loaded  (default: {None})
            is_best {bool}  -- Load the best epoch (default: {False})
            path {str}      -- Path to weight file (default: {None})

        Raises:
            Exception -- [description]
            IOError -- [description]
        """

        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        if is_best:
            fname_g = f"netG_best.pth"
            fname_d = f"netD_best.pth"
        else:
            fname_g = f"netG_{epoch}.pth"
            fname_d = f"netD_{epoch}.pth"

        if path is None:
            path_g = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_g}"
            path_d = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_d}"

        # Load the weights of netg and netd.
        print('>> Loading weights...')
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        try:
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.data.train.dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(f">> Training {self.name} on {self.opt.dataset} to detect {self.opt.abnormal_class}")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                        real, fake, _ = self.get_current_images()
                        vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                        vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

                        # Measure inference time.
                        self.times = np.array(self.times)
                        self.times = np.mean(self.times[:100] * 1000)

                        # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

        ##
        def update_learning_rate(self):
            """ Update learning rate based on the rule provided in options.
            """
    
            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('   LR = %.7f' % lr)        