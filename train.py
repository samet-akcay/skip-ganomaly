# """
# TRAIN GANOMALY

# . Example: Run the following command from the terminal.
#     run train.py                             \
#         --model ganomaly                        \
#         --dataset UCSD_Anomaly_Dataset/UCSDped1 \
#         --batchsize 32                          \
#         --isize 256                         \
#         --nz 512                                \
#         --ngf 64                               \
#         --ndf 64
# """


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data.dataloader import load_data
# from lib.models.ganomaly import Ganomaly
from lib.models import load_model
import copy

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

# # NOTE: this bit should go under the train method in v3
# # TODO: new bit
# opt_h = copy.copy(opt)
# opt_h.model = 'aae'
# model_h = load_model(opt_h, dataloader)
# model_h.train(epochs=5)
##
# LOAD MODEL
model = load_model(opt, dataloader)
# model.netg_h.load_state_dict(model_h.netg.state_dict())
##
# TRAIN MODEL
model.train()
