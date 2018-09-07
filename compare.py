"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data.dataloader import load_data
# from lib.models.ganomaly import Ganomaly
from lib.models import load_model

##
# def main():
""" Training
"""

##
# ARGUMENTS
opt = Options().parse()

##
# LOAD DATA
dataloader = load_data(opt)

opt.model = 'nodisc'
m1 = load_model(opt, dataloader)

opt.model = 'ganomaly'
m2 = load_model(opt, dataloader)

opt.model = 'ganomaly2'
m3 = load_model(opt, dataloader)


m1.train()
m2.train()
m3.train()

##
inp = next(iter(dataloader['train']))[0].cuda()
i1 = inp
i2 = inp
f1 = m1.netg(i1)[0]
f2 = m2.netg(i2)[0]
real1 = i1.data
real2 = i2.data
fake1 = f1.data
fake2 = f2.data

self.visualizer.display_current_images(real1, fake1, fixed, win=9, title='nodisc')
self.visualizer.display_current_images(real2, fake2, fixed, win=6, title='ganomaly')




# ##
# # LOAD MODEL
# model = load_model(opt, dataloader)
# ##
# # TRAIN MODEL
# model.train()
