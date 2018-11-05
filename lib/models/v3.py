"""GANomaly
"""
# pylint: disable=C0301,E0602,E1101,W0622,C0103,R0902,R0915

##
import os
import time
import itertools
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from dataclasses import dataclass
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

# from lib.loss import Loss
from lib.evaluate import roc, evaluate
from lib.visualizer import Visualizer
from lib.models.networks import define_D, define_G, get_scheduler, GANLoss


##
class V3:
    """GANomaly Class
    """

    def __init__(self, opt, dataloader=None):
        super(V3, self).__init__()
        ##
        # Initalize variables.
        self.name = 'v3'
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if self.opt.gpus != -1 else "cpu")

        # Input, output and loss variables.
        self.inp = Input(opt)
        self.out = Output(opt)
        self.hid = Output(opt)
        self.loss = Loss()
        self.crit = Criterion()

        # -- Misc variables.
        self.epoch = 0
        self.times = []
        self.steps = 0

        ## Create and Load Models.
        self.netg_h = define_G(opt, net='dcgan')
        self.netg_o = define_G(opt, net='unet32')
        self.netd  = define_D(opt)
        self.netd_h = define_D(opt)
        self.netd_o = define_D(opt)

        ##
        if self.opt.resume != '': self.load_weights(path=self.opt.resume)
        print(self.netg_h)
        print(self.netg_o)
        print(self.netd)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg_h.train()
            self.netg_o.train()
            # self.netd.train()
            self.netd_h.train()
            self.netd_o.train()
            self.optimizers  = []
            # self.optimizer_d = optim.Adam(params=self.netd.parameters(),
            #                               lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_d = optim.Adam(params=itertools.chain(self.netd_h.parameters(), self.netd_o.parameters()),
                                          lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(params=itertools.chain(self.netg_h.parameters(), self.netg_o.parameters()),
                                          lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    ##
    def set_input(self, input):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        self.inp.img.data.resize_(input[0].size()).copy_(input[0])
        self.inp.gts.data.resize_(input[1].size()).copy_(input[1])

        # Add Gaussian Noise if requested.
        if self.opt.add_noise:
            self.inp.noi = self.inp.img.data.new(input[0].size()).normal_(self.opt.mean, self.opt.std)

        # Assign the first batch as fixed input.
        if self.steps == self.opt.batchsize:
            self.inp.fix.data.resize_(input[0].size()).copy_(input[0])

    def forward(self):
        self.hid.img, self.hid.vec = self.netg_h(self.inp.img + self.inp.noi)
        self.out.img, self.out.vec = self.netg_o(self.hid.img)

    def backward_d(self, netd, real, fake):
        # Real
        pred_real, _ = netd(real)
        loss_d_real = self.crit.adv(pred_real, True)
        # Fake
        pred_fake, _ = netd(fake.detach())
        loss_d_fake = self.crit.adv(pred_fake, False)
        # Combined loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        # Backward
        loss_d.backward()
        return loss_d
    
    ##
    def backward_d_h(self):
        self.loss.d_h = backward_d(self.netd_h, self.inp.img, self.hid.img)
    
    ##
    def backward_d_o(self):
        self.loss.d_o = backward_d(self.netd_o, self.inp.img, self.out.img)
    
    ##
    def backward_g(self):
        # self.netg_h.zero_grad()
        # self.netg_o.zero_grad()
        # self.inp.lbl.data.resize_(self.opt.batchsize).fill_(self.inp.real_lbl)
        # self.hid.fake_score, self.hid.fake_feats = self.netd(self.hid.img)
        # self.out.fake_score, self.out.fake_feats = self.netd(self.out.img)

        self.loss.g.enc = self.opt.w_enc * self.crit.enc(self.out.vec, self.hid.vec)
        self.loss.g.rec = self.opt.w_rec * self.crit.rec(self.inp.img, self.out.img)

        self.loss.g.g1.tot = self.opt.w_g1_adv * self.crit.adv(self.hid.fake_score, self.inp.lbl) \
                           + self.opt.w_g1_rec * self.crit.rec(self.hid.img, self.inp.img)
        self.loss.g.g2.tot = self.opt.w_g2_adv * self.crit.adv(self.out.fake_score, self.inp.lbl)

        self.loss.g.total  = self.loss.g.enc + self.loss.g.rec + self.loss.g.g1.tot + self.loss.g.g2.tot

        # self.loss.g.adv   = self.opt.w_adv  * self.loss.bce(self.out.fake_score, self.inp.lbl)
        # self.loss.g.rec   = self.opt.w_rec  * self.loss.l1(self.out.img, self.inp.img)
        # self.loss.g.total = self.loss.g.adv + self.loss.g.rec + self.loss.g.enc

        self.loss.g.total.backward()
        # self.optimizer_g.step()        

    # ##
    # def update_netd(self):
    #     """
    #     Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #     """
    #     # BCE
    #     self.netd.zero_grad()
    #     # --
    #     # Real
    #     self.hid.real_score, self.hid.real_feats = self.netd(self.inp.img + self.inp.noi)        
    #     self.inp.lbl.data.resize_(self.opt.batchsize).fill_(self.inp.real_lbl)
    #     self.loss.d.real = self.loss.bce(self.hid.real_score, self.inp.lbl)
    #     # TODO: ADD LOSS NETG2?
    #     # self.out.real_score, self.out.real_feats = self.netd(self.inp.img + self.inp.noi)
    #     # self.loss.d.real = self.loss.bce(self.out.real_score, self.inp.lbl)

    #     # --
    #     # Fake
    #     self.inp.lbl.data.resize_(self.opt.batchsize).fill_(self.inp.fake_lbl)
    #     self.hid.img, self.hid.vec = self.netg_h(self.inp.img + self.inp.noi)
    #     self.out.img, self.out.vec = self.netg_o(self.hid.img)

    #     self.hid.fake_score, self.hid.fake_feats = self.netd(self.hid.img.detach())
    #     self.out.fake_score, self.out.fake_feats = self.netd(self.out.img.detach())
    #     self.loss.d.fake = self.loss.bce(self.hid.fake_score, self.inp.lbl)
    #     # TODO: ADD LOSS NETG2?
    #     # self.loss.d.fake = self.loss.bce(self.out.fake_score, self.inp.lbl)

    #     # --
    #     # Compute the total loss based on the training type (feature maching | standard bce.)
    #     if self.opt.netD_training == 'fm':
    #         self.loss.d.total = self.loss.l2(self.hid.fake_feats, self.hid.real_feats)
    #         # TODO: ADD LOSS NETG2?
    #         # self.loss.d.total = self.loss.l2(self.out.fake_feats, self.out.real_feats)
    #     elif self.opt.netD_training == 'bce':
    #         self.loss.d.total = self.loss.d.real + self.loss.d.fake
    #         # TODO: ADD LOSS NETG2?
    #         # self.loss.d.total = self.loss.d.real + self.loss.d.fake
    #     else:
    #         raise Exception('Supported netD trainings are: fm | bce')

    #     self.loss.d.total.backward(retain_graph=True)
    #     self.optimizer_d.step()

    # ##
    # def update_netg(self):
    #     """
    #     # ============================================================ #
    #     # (2) Update G network: log(D(G(z)))  + ||G(z) - x||           #
    #     # ============================================================ #
    #     """
    #     self.netg_h.zero_grad()
    #     self.netg_o.zero_grad()
    #     self.inp.lbl.data.resize_(self.opt.batchsize).fill_(self.inp.real_lbl)
    #     self.hid.fake_score, self.hid.fake_feats = self.netd(self.hid.img)
    #     self.out.fake_score, self.out.fake_feats = self.netd(self.out.img)

    #     # TODO: UPdate netg.
    #     self.loss.g.enc = self.opt.w_enc * self.crit.enc(self.out.vec, self.hid.vec)
    #     self.loss.g.rec = self.opt.w_rec * self.crit.rec(self.inp.img, self.out.img)

    #     self.loss.g.g1.tot = self.opt.w_g1_adv * self.loss.bce(self.hid.fake_score, self.inp.lbl) \
    #                        + self.opt.w_g1_rec * self.loss.l1 (self.hid.img, self.inp.img)
    #     self.loss.g.g2.tot = self.opt.w_g2_adv * self.loss.bce(self.out.fake_score, self.inp.lbl)

    #     self.loss.g.total  = self.loss.g.enc + self.loss.g.rec + self.loss.g.g1.tot + self.loss.g.g2.tot

    #     # self.loss.g.adv   = self.opt.w_adv  * self.loss.bce(self.out.fake_score, self.inp.lbl)
    #     # self.loss.g.rec   = self.opt.w_rec  * self.loss.l1(self.out.img, self.inp.img)
    #     # self.loss.g.total = self.loss.g.adv + self.loss.g.rec + self.loss.g.enc

    #     self.loss.g.total.backward()
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
    # def optimize(self):
    #     """ Optimize netD and netG  networks.
    #     """

    #     self.update_netd()
    #     self.update_netg()

    #     # If D loss is zero, then re-initialize netD
    #     if self.loss.d.real.item() < 1e-5 or self.loss.d.fake.item() < 1e-5:
    #         self.reinitialize_netd()

    def optimize_parameters(self):
        # Forward
        self.forward()
        
        # Generator
        self.set_requires_grad([self.netd_h, self.netd_o], False)
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
        
        # Discriminator
        self.set_requires_grad([self.netd_h, self.netd_o], True)
        self.optimizer_d.zero_grad()
        self.backward_d_h()
        self.backward_d_o()
        self.optimizer_d.step()

        # If D loss is zero, then re-initialize netD
        if self.loss.d.real.item() < 1e-5 or self.loss.d.fake.item() < 1e-5:
            self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('loss.d.total', self.loss.d.total.item()),
            ('loss.g.total', self.loss.g.total.item()),
            ('loss.d.real', self.loss.d.real.item()),
            ('loss.d.fake', self.loss.d.fake.item()),
            ('loss.g.rec', self.loss.g.rec.item()),
            ('loss.g.enc', self.loss.g.enc.item()),
            ('loss.g.g1.tot', self.loss.g.g1.tot.item()),
            ('loss.g.g2.tot', self.loss.g.g2.tot.item())
        ])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """
        # reals = self.inp.img.data
        # fakes = self.out.img.data
        # fixed = self.netg(self.inp.fix).data

        # return reals, fakes, fixed
        images = {'input':  self.inp.img.data,
                  'hidden': self.hid.img.data,
                  'output': self.out.img.data}
        return images
        # return self.inp.img.data, self.hid.img.data, self.out.img.data

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
            torch.save({'epoch': epoch, 'state_dict': self.netg_h.state_dict()}, f'{weight_dir}/netG1_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netg_o.state_dict()}, f'{weight_dir}/netG2_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()},  f'{weight_dir}/netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netg_h.state_dict()}, f"{weight_dir}/netG1_{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg_o.state_dict()}, f'{weight_dir}/netG2_{epoch}.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()},  f"{weight_dir}/netD_{epoch}.pth")

    ##
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

        self.netg_h.train()
        self.netg_o.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize_parameters()

            if self.steps % self.opt.print_freq == 0 and self.opt.display:
                errors = self.get_errors()
                counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors, win=1)

            if self.steps % self.opt.save_image_freq == 0 and self.opt.display:
                self.visualizer.display_current_images(self.get_current_images(), win=2)

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
                self.hid.img, self.hid.vec = self.netg_h(self.inp.img)
                self.out.img, self.out.vec = self.netg_o(self.hid.img)

                # self.out.real_score, self.out.real_feats = self.netd(self.inp.img)
                # self.out.fake_score, self.out.fake_feats = self.netd(self.out.img)

                # Calculate the anomaly score.
                si = self.inp.img.size()
                sz = self.out.vec.size()
                rec = (self.inp.img - self.out.img).view(si[0], si[1] * si[2] * si[3])
                lat = (self.hid.vec - self.out.vec).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat
                error = lat
                # TODO: Anomaly score has been changed.
                # error = self.out.fake_score
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.inp.gts.reshape(error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, f'{dst}/real_{i+1:03}', normalize=True)
                    vutils.save_image(fake, f'{dst}/fake_{i+1:03}', normalize=True)
                if self.steps % self.opt.save_image_freq == 0:
                    if self.opt.display:
                        self.visualizer.display_current_images(self.get_current_images(), win=5, title='Test')

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
                self.out.img = self.netg(self.inp)

                _, self.feat_real = self.netd(self.inp)
                _, self.feat_fake = self.netd(self.out.img)

                while key != 'n':
                    # Show the images on visdom.
                    reals = self.visualizer.normalize(self.inp.cpu().numpy())
                    fakes = self.visualizer.normalize(self.out.imf.cpu().numpy())
                    diffs = self.visualizer.normalize((self.inp - self.out.img).cpu().numpy())
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
        vec_size = (opt.batchsize, opt.nz, 1, 1)
        img_type = torch.float32      
        vec_type = torch.float32
        device   = torch.device("cuda:0" if opt.gpus != -1 else "cpu")

        self.img = torch.empty(size=img_size, dtype=img_type, device=device)
        self.vec = torch.empty(size=vec_size, dtype=vec_type, device=device)
        self.real_feats = None
        self.fake_feats = None
        self.real_score = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=device)
        self.fake_score = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device=device)


# # # # # 
# LOSS  #
# # # # # 
##
@dataclass
class LossD:
    """ Loss val for netD
    """
    total: torch.Tensor = None
    real : torch.Tensor = None
    fake : torch.Tensor = None

@dataclass
class LossG1:
    """ Losses for G1 (DCGAN)
    """
    tot: torch.Tensor = None
    adv: torch.Tensor = None
    rec: torch.Tensor = None

@dataclass
class LossG2:
    """ Losses for G2 (UNET)
    """
    tot: torch.Tensor = None
    adv: torch.Tensor = None

@dataclass
class LossG:
    """ Loss val for netG
    """
    total: torch.Tensor = None
    rec  : torch.Tensor = None
    enc  : torch.Tensor = None
    g1   : type(LossG1) = LossG1
    g2   : type(LossG2) = LossG2

@dataclass
class Loss:
    # Loss Values
    d: type(LossD) = LossD()
    g: type(LossG) = LossG()
    d_h = torch.Tensor = None
    d_o = torch.Tensor = None

    # Loss Functions
    bce: type(torch.nn.BCELoss()) = torch.nn.BCELoss()
    l1 : type(torch.nn.L1Loss())  = torch.nn.L1Loss()
    l2 : type(torch.nn.MSELoss()) = torch.nn.MSELoss()

class Criterion:
    def __init__(self):
        self.adv = GANLoss(use_lsgan=False)
        self.rec = torch.nn.L1Loss()
        self.enc = torch.nn.MSELoss()