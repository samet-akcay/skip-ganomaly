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

##
# LOAD MODEL
model = load_model(opt, dataloader)
##
# DEMO MODEL
model.demo()
