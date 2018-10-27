"""GANomaly
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

from lib.models.networks import NetD, NetDv2, weights_init, define_G, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.model = define_G(input_nc=opt.nc, output_nc=opt.nc, ngf=64,
                              which_model_netG='unet_32',
                              norm='batch', use_dropout=False, init_type='normal',
                              gpu_ids=opt.gpu_ids)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o

##


class Model:
    """GANomaly Class
    """

    def __init__(self, opt, dataloader=None):
        super(Model, self).__init__()
        ##
        # Initalize variables.
        self.name = 'model'
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device(
            "cuda:0" if self.opt.gpu_ids != -1 else "cpu")

        # -- Discriminator attributes.
        self.out_d_real = None
        self.feat_real = None
        self.err_d_real = None
        self.fake = None
        # self.latent_i = None
        # self.latent_o = None
        self.out_d_fake = None
        self.feat_fake = None
        self.err_d_fake = None
        self.err_d = None

        # -- Generator attributes.
        self.out_g = None
        self.err_g_bce = None
        self.err_g_l1l = None
        self.err_g_enc = None
        self.err_g = None

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        # self.netg = NetG(self.opt).to(self.device)
        self.netg = define_G(opt, input_nc=self.opt.nc, output_nc=self.opt.nc, ngf=self.opt.ngf,
                             which_model_netG='unet',
                             norm='batch', use_dropout=False, init_type='normal',
                             gpu_ids=opt.gpu_ids)
        self.netd = NetDv2(self.opt).to(self.device)
        # self.netd2 = NetDv2(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(
                self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        print(self.netg)
        print(self.netd)

        ##
        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.l2l_criterion = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize,
                                       self.opt.isize), dtype=torch.float32, device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize,
                                       self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(
            size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,),
                              dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(
            self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

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
        self.input.data.resize_(input[0].size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])

        # Add Gaussian Noise if requested.
        if self.opt.add_gaussian_noise:
            self.noise = self.input.data.new(input[0].size()).normal_(self.opt.mean,self.opt.std)

        # Copy the first batch as the fixed input.
        if self.total_steps == self.opt.batchsize:
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])

    ##
    def update_netd(self):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """
        # BCE
        self.netd.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        if self.opt.add_gaussian_noise: self.out_d_real, self.feat_real = self.netd(self.input + self.noise)
        else: self.out_d_real, self.feat_real = self.netd(self.input)
        self.err_d_real = self.bce_criterion(self.out_d_real, self.label)
        self.err_d_real.backward(retain_graph=True)
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.fake = self.netg(self.input + self.noise)

        self.out_d_fake, self.feat_fake = self.netd(self.fake.detach())
        self.err_d_fake = self.bce_criterion(self.out_d_fake, self.label)

        # Feature Loss btw real and fake images.
        self.err_g_enc = self.l2l_criterion(self.feat_fake, self.feat_real)
        self.err_g_enc.backward(retain_graph=True)

        # --
        self.err_d_fake.backward()
        self.err_d = self.err_d_real + self.err_d_fake + self.err_g_enc
        self.optimizer_d.step()

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def update_netg(self):
        """
        # ============================================================ #
        # (2) Update G network: log(D(G(z)))  + ||G(z) - x||           #
        # ============================================================ #

        """
        self.netg.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_g, _ = self.netd(self.fake)

        self.err_g_bce = self.bce_criterion(self.out_g, self.label)
        self.err_g_l1l = self.l1l_criterion(self.fake, self.input)  # constrain x' to look like x
        # self.err_g_enc = self.l2l_criterion(self.latent_o, self.latent_i)
        # self.err_g = self.err_g_bce + self.err_g_l1l * self.opt.alpha + self.err_g_enc
        self.err_g = self.err_g_bce + self.err_g_l1l * self.opt.w_rec

        self.err_g.backward(retain_graph=True)
        self.optimizer_g.step()

    def update_learning_rate(self):
        """ Update learning rate based on the rule provided in options.
        """

        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('   LR = %.7f' % lr)

    ##
    def optimize(self):
        """ Optimize netD and netG  networks.
        """

        self.update_netd()
        self.update_netg()

        # If D loss is zero, then re-initialize netD
        if self.err_d_real.item() < 1e-5 or self.err_d_fake.item() < 1e-5:
            self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([('err_d', self.err_d.item()),
                              ('err_g', self.err_g.item()),
                              ('err_d_real', self.err_d_real.item()),
                              ('err_d_fake', self.err_d_fake.item()),
                              ('err_g_bce', self.err_g_bce.item()),
                              ('err_g_l1l', self.err_g_l1l.item()),
                              ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input).data

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

        # Load the weights of netg and netd.
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        try:
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
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            self.optimize()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / \
                        len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(
                        self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
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
        self.total_steps = 0
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
            if self.opt.load_weights:
                self.load_weights(is_best=True)

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

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                
                # Forward - Pass
                self.set_input(data)
                self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)


                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9*rec + 0.1*lat

                time_o = time.time()

                # latent_i = self.feat_real.view(sizes[0], sizes[1] * sizes[2] * sizes[3])
                # latent_o = self.feat_fake.view(sizes[0], sizes[1] * sizes[2] * sizes[3])

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize +
                               error.size(0)] = self.gt.reshape(error.size(0))
                # self.latent_i[i*self.opt.batchsize: i*self.opt.batchsize +
                #               error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                # self.latent_o[i*self.opt.batchsize: i*self.opt.batchsize +
                #               error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(
                        self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' %
                                      (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' %
                                      (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc, eer = roc(self.gt_labels, self.an_scores)
            performance = OrderedDict(
                [('Avg Run Time (ms/batch)', self.times), ('EER', eer), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / \
                    len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(
                    self.epoch, counter_ratio, performance)
            return performance

    def demo(self):
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.opt.phase = 'test'

            self.total_steps = 0
            epoch_iter = 0
            key = 'a'
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                while key != 'n':
                    # Show the images on visdom.
                    reals = self.visualizer.normalize(self.input.cpu().numpy())
                    fakes = self.visualizer.normalize(self.fake.cpu().numpy())
                    diffs = self.visualizer.normalize((self.input - self.fake).cpu().numpy())
                    self.visualizer.vis.images(reals, win=1, opts={'title': 'Reals'})
                    self.visualizer.vis.images(fakes, win=2, opts={'title': 'Fakes'})
                    self.visualizer.vis.images(diffs, win=3, opts={'title': 'Diffs'})
                    input("Key:")

                if key == 'q':
                    print('Quitting...')
                    break


