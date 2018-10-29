"""GANomaly
"""
# pylint: disable=C0301,E0602,E1101,W0622,C0103,R0902,R0915

##
import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from lib.loss import Loss
from lib.evaluate import roc
from lib.visualizer import Visualizer
from lib.models.networks import define_D, define_G, get_scheduler

##
class Ganomaly2:
    """GANomaly Class
    """

    def __init__(self, opt, dataloader=None):
        super(Ganomaly2, self).__init__()
        ##
        # Initalize variables.
        self.name = 'ganomaly2'
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if self.opt.gpu_ids != -1 else "cpu")

        # Input, output and loss variables.
        self.input = Input(opt)
        self.output = Output(opt)
        self.loss = Loss()

        # -- Misc variables.
        self.epoch = 0
        self.times = []
        self.steps = 0

        ## Create and Load Models.
        self.netg = define_G(opt)
        # self.netd = NetDv2(self.opt).to(self.device)
        self.netd = define_D(opt)
        # TODO: Create define_D function

        ##
        if self.opt.resume != '': self.load_weights(path=self.opt.resume)
        print(self.netg)
        print(self.netd)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizers  = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    ##
    def set_input(self, input):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        self.input.img.data.resize_(input[0].size()).copy_(input[0])
        self.input.gts.data.resize_(input[1].size()).copy_(input[1])

        # Add Gaussian Noise if requested.
        if self.opt.add_noise:
            self.input.noi = self.input.img.data.new(input[0].size()).normal_(self.opt.mean,self.opt.std)

        # Assign the first batch as fixed input.
        if self.steps == self.opt.batchsize:
            self.input.fix.data.resize_(input[0].size()).copy_(input[0])

    ##
    def update_netd(self):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """
        # BCE
        self.netd.zero_grad()
        # --
        # Train with real
        self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.real_lbl)
        self.output.real_score, self.output.real_feats = self.netd(self.input.img + self.input.noi)
        self.loss.d.real = self.loss.bce(self.output.real_score, self.input.lbl)

        # --
        # Train with fake
        self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.fake_lbl)
        self.output.img = self.netg(self.input.img + self.input.noi)

        self.output.fake_score, self.output.fake_feats = self.netd(self.output.img.detach())
        self.loss.d.fake = self.loss.bce(self.output.fake_score, self.input.lbl)

        # Feature Loss btw real and fake images.
        self.loss.g.enc = self.loss.l2(self.output.fake_feats, self.output.real_feats)

        # --
        # Compute the total loss based on the training type (feature maching | standard bce.)
        if self.opt.netD_training == 'fm': self.loss.d.total = self.loss.g.enc
        elif self.opt.netD_training == 'bce': self.loss.d.total = self.loss.d.real + self.loss.d.fake + self.loss.g.enc
        else: raise Exception('Supported netD trainings are: fm | bce')
        self.loss.d.total.backward(retain_graph=True)
        self.optimizer_d.step()

    ##
    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(z)))  + ||G(z) - x||           #
        # ============================================================ #
        """
        self.netg.zero_grad()
        self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.real_lbl)
        self.output.fake_score, self.output.fake_feats = self.netd(self.output.img)

        self.loss.g.adv   = self.opt.w_adv  * self.loss.bce(self.output.fake_score, self.input.lbl)
        self.loss.g.rec   = self.opt.w_rec  * self.loss.l1(self.output.img, self.input.img)
        self.loss.g.total = self.loss.g.adv + self.loss.g.rec + self.loss.g.enc

        self.loss.g.total.backward()
        self.optimizer_g.step()

    # ##
    # def update_netd(self):
    #     """
    #     Update D network: Ladv = |f(real) - f(fake)|_2
    #     """
    #     ##
    #     # Feature Matching.
    #     self.netd.zero_grad()
    #     # --
    #     # Train with real
    #     self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.real_lbl)
    #     self.output.real_score, self.output.real_feats = self.netd(self.input.img + self.input.noi)

    #     # --
    #     # Train with fake
    #     self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.fake_lbl)
    #     self.output.img = self.netg(self.input.img + self.input.noi)
    #     self.output.fake_score, self.output.fake_feats = self.netd(self.output.img.detach())

    #     # --
    #     # Compute Loss and Backward-Pass.
    #     self.loss.g.enc   = self.opt.w_enc  * self.loss.l2(self.output.fake_feats, self.output.real_feats)
    #     # self.loss.d.total = self.loss.l2(self.output.real_feats, self.output.fake_feats)
    #     self.loss.d.total = self.loss.g.enc
    #     self.loss.d.real = self.loss.d.total
    #     self.loss.d.fake = self.loss.d.total
    #     self.loss.d.total.backward(retain_graph=True)
    #     self.optimizer_d.step()
    # 
    # ##
    # def update_netg(self):
    #     """
    #     # ============================================================ #
    #     # (2) Update G network: log(D(G(z)))  + ||G(z) - x||           #
    #     # ============================================================ #
    #     """
    #     self.netg.zero_grad()
    #     self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.real_lbl)
    #     self.output.fake_score, self.output.fake_feats = self.netd(self.output.img)

    #     self.loss.g.adv   = self.opt.w_adv  * self.loss.bce(self.output.fake_score, self.input.lbl)
    #     self.loss.g.rec   = self.opt.w_rec  * self.loss.l1(self.output.img, self.input.img)
    #     self.loss.g.total = self.loss.g.adv + self.loss.g.rec + self.loss.g.enc

    #     self.loss.g.total.backward(retain_graph=False)
    #     self.optimizer_g.step()


    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def update_learning_rate(self):
        """ Update learning rate based on the rule provided in options.
        """
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        # print('   LR = %.7f' % lr)

    ##
    def optimize(self):
        """ Optimize netD and netG  networks.
        """

        self.update_netd()
        self.update_netg()

        # If D loss is zero, then re-initialize netD
        if self.loss.d.real.item() < 1e-5 or self.loss.d.fake.item() < 1e-5:
            self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([('loss.d.total', self.loss.d.total.item()),
                              ('loss.g.total', self.loss.g.total.item()),
                              ('loss.d.real', self.loss.d.real.item()),
                              ('loss.d.fake', self.loss.d.fake.item()),
                              ('loss.g.adv', self.loss.g.adv.item()),
                              ('loss.g.rec', self.loss.g.rec.item()),
                              ('loss.g.enc', self.loss.g.enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.img.data
        fakes = self.output.img.data
        fixed = self.netg(self.input.fix).data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch, is_best=False):
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

    def load_weights(self, epoch=None, is_best=False, path=None):
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
        else:
            path_g = f"{path}/netG_best.pth"
            path_d = f"{path}/netD_best.pth"

        # Load the weights of netg and netd.
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        try:
            self.opt.iter = torch.load(path_g)['epoch']
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Loaded weights.')

    ##
    def train_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize()

            if self.steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / \
                        len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(
                        self.epoch, counter_ratio, errors)

            if self.steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(
                    self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(f">> Training {self.name}. Epoch {self.epoch + 1} / {self.opt.niter}")
        # self.visualizer.print_current_errors(self.epoch, errors)
    
    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(f">> Training {self.name}.")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch, is_best=True)
            self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
            self.update_learning_rate()
        print(f">> Training {self.name}. [Done]")

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights: self.load_weights(is_best=True)
            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),),
                                         dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),),
                                         dtype=torch.long,
                                         device=self.device)
            # self.latent_i = torch.zeros( size=(len(self.dataloader['test'].dataset), self.opt.nz),
            #                              dtype=torch.float32,
            #                              device=self.device)
            # self.latent_o = torch.zeros( size=(len(self.dataloader['test'].dataset), self.opt.nz),
            #                              dtype=torch.float32,
            #                              device=self.device)

            print(f"   Testing {self.name}")
            self.times = []
            self.steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                
                # Forward - Pass
                self.set_input(data)
                self.output.img = self.netg(self.input.img)

                self.output.real_score, self.output.real_feats = self.netd(self.input.img)
                self.output.fake_score, self.output.fake_feats = self.netd(self.output.img)

                # Calculate the anomaly score.
                si = self.input.img.size()
                sz = self.output.real_feats.size()
                rec = (self.input.img - self.output.img).view(si[0], si[1] * si[2] * si[3])
                lat = (self.output.real_feats - self.output.fake_feats).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat
                # TODO: Anomaly score has been changed.
                # error = self.output.fake_score
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.input.gts.reshape(error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc, eer = roc(self.gt_labels, self.an_scores)
            performance = {'Avg Run Time (ms/batch)': self.times, 'EER': eer, 'AUC': auc}

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

    def demo(self):
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights: self.load_weights(is_best=True)
            self.opt.phase = 'test'

            self.steps = 0
            epoch_iter = 0
            key = 'a'
            for i, data in enumerate(self.dataloader['test'], 0):
                self.steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.output.img = self.netg(self.input.img)

                _, self.output.real_feats = self.netd(self.input.img)
                _, self.output.fake_feats = self.netd(self.output.img)

                while key != 'n':
                    # Show the images on visdom.
                    reals = self.visualizer.normalize(self.input.img.cpu().numpy())
                    fakes = self.visualizer.normalize(self.output.img.cpu().numpy())
                    diffs = self.visualizer.normalize((self.input.img - self.output.img).cpu().numpy())
                    self.visualizer.vis.images(reals, win=1, opts={'title': 'Reals'})
                    self.visualizer.vis.images(fakes, win=2, opts={'title': 'Fakes'})
                    self.visualizer.vis.images(diffs, win=3, opts={'title': 'Diffs'})
                    input("Key:")

                if key == 'q':
                    print('Quitting...')
                    break

##
class Input:
    def __init__(self, opt):
        img_size = (opt.batchsize, opt.nc, opt.isize, opt.isize)
        img_type = torch.float32
        gts_size = (opt.batchsize,)
        gts_type = torch.long
        device   = torch.device("cuda:0" if opt.gpu_ids != -1 else "cpu")

        self.img = torch.empty(size=img_size, dtype=img_type, device=device)
        self.gts = torch.empty(size=gts_size, dtype=gts_type, device=device)
        self.noi = torch.empty(size=img_size, dtype=img_type, device=device)
        self.fix = torch.empty(size=img_size, dtype=img_type, device=device)

        self.lbl = torch.empty(size=gts_size, dtype=img_type, device=device)
        self.fake_lbl = 0
        self.real_lbl = 1

##
class Output:
    def __init__(self, opt):
        img_size = (opt.batchsize, opt.nc, opt.isize, opt.isize)
        img_type = torch.float32      
        device   = torch.device("cuda:0" if opt.gpu_ids != -1 else "cpu")

        self.img = torch.empty(size=img_size, dtype=img_type, device=device)
        self.real_feats = None
        self.fake_feats = None
        self.real_score = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=device)
        self.fake_score = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=device)
