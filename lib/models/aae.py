"""GANomaly
"""
# pylint: disable=C0301,E0602,E1101,W0622,C0103,R0902,R0915

##
import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

# from lib.loss import Loss
from lib.evaluate import roc, evaluate
from lib.visualizer import Visualizer
from lib.models.networks import define_D, define_G, get_scheduler

##
class Aae:
    """GANomaly Class
    """

    def __init__(self, opt, dataloader=None):
        super(Aae, self).__init__()
        ##
        # Initalize variables.
        self.name = 'aae'
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if self.opt.gpus != -1 else "cpu")

        # Input, output and loss variables.
        self.input = Input(opt)
        self.output = Output(opt)
        self.loss = Loss()
        self.crit = Criterion()

        # -- Misc variables.
        self.epoch = 0
        self.times = []
        self.steps = 0

        ## Create and Load Models.
        self.netg = define_G(opt, net='dcgan')
        self.netd = define_D(opt)

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
        self.loss.d_real = self.crit.bce(self.output.real_score, self.input.lbl)

        # --
        # Train with fake
        self.input.lbl.data.resize_(self.opt.batchsize).fill_(self.input.fake_lbl)
        self.output.img, _ = self.netg(self.input.img + self.input.noi)

        self.output.fake_score, self.output.fake_feats = self.netd(self.output.img.detach())
        self.loss.d_fake = self.crit.bce(self.output.fake_score, self.input.lbl)

        # --
        # Compute the total loss based on the training type (feature maching | standard bce.)
        if self.opt.netD_training == 'fm': self.loss.d = self.crit.l2(self.output.fake_feats, self.output.real_feats)
        elif self.opt.netD_training == 'bce': self.loss.d = self.loss.d_real + self.loss.d_fake
        else: raise Exception('Supported netD trainings are: fm | bce')
        self.loss.d.backward(retain_graph=True)
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

        self.loss.g_adv   = self.crit.bce(self.output.fake_score, self.input.lbl)
        self.loss.g_rec   = self.crit.l1(self.output.img, self.input.img) * 50
        self.loss.g = self.loss.g_adv + self.loss.g_rec

        self.loss.g.backward()
        self.optimizer_g.step()

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

    ##
    def optimize(self):
        """ Optimize netD and netG  networks.
        """

        self.update_netd()
        self.update_netg()

        # If D loss is zero, then re-initialize netD
        if self.loss.d_real.item() < 1e-5 or self.loss.d_fake.item() < 1e-5:
            self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        return {i: getattr(self.loss, i).item()  for i in dir(self.loss) if not i.startswith('_')}

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """
        images = {
            'reals' : self.input.img.data,
            'fakes' : self.output.img.data,
            'fixed' : self.netg(self.input.fix)[0].data
        }

        return images

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
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.steps % self.opt.save_image_freq == 0:
                images = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, images)
                if self.opt.display:
                    self.visualizer.display_current_images(images)

        print(f">> Training {self.name}. Epoch {self.epoch + 1} / {self.opt.niter}")
        # self.visualizer.print_current_errors(self.epoch, errors)
    
    ##
    def train(self, epochs=None):
        """ Train the model
        """

        ##
        # TRAIN
        if epochs is not None:
            self.opt.niter = epochs
        self.steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(f">> Training {self.name}.")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_epoch()
            res = self.test()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
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
            size = (len(self.dataloader['test'].dataset),)
            self.an_scores = torch.zeros(size=size, dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=size, dtype=torch.long, device=self.device)

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
                self.output.img, _ = self.netg(self.input.img)

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
                error = lat
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
                    images= self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)
                if self.steps % self.opt.save_image_freq == 0:
                    images = self.get_current_images()
                    if self.opt.display:
                        self.visualizer.display_current_images(images, win=5, title='Test')

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                (torch.max(self.an_scores) - torch.min(self.an_scores))
            res = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            if len(res) == 2:
                performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('EER', res[1]), (f"{self.opt.metric}", res[0])])
            else:
                performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (f"{self.opt.metric}", res[0])])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

##
class Input:
    def __init__(self, opt):
        img_size = (opt.batchsize, opt.nc, opt.isize, opt.isize)
        img_type = torch.float32
        gts_size = (opt.batchsize,)
        gts_type = torch.long
        device   = torch.device("cuda:0" if opt.gpus != -1 else "cpu")

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
        device   = torch.device("cuda:0" if opt.gpus != -1 else "cpu")

        self.img = torch.empty(size=img_size, dtype=img_type, device=device)
        self.real_feats = None
        self.fake_feats = None
        self.real_score = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=device)
        self.fake_score = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=device)

##    
@dataclass
class Loss:
    d_real  : torch.Tensor = None
    d_fake  : torch.Tensor = None
    d       : torch.Tensor = None
    g_adv   : torch.Tensor = None
    g_rec   : torch.Tensor = None
    g       : torch.Tensor = None

##
class Criterion:
    def __init__(self):
        self.bce = torch.nn.BCELoss()
        self.l1  = torch.nn.L1Loss()
        self.l2  = torch.nn.MSELoss()