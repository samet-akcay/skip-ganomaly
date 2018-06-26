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
from lib.data import load_data
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
# model = Ganomaly(opt, dataloader)
model = load_model(opt, dataloader)
##
# TRAIN MODEL
model.train()

##

# def load_model(opt, dataloader):
#     import importlib
#     model_name = opt.model
#     model_path = f"lib.models.{model_name}"
#     model_lib  = importlib.import_module(model_path)
#     model = getattr(modellib, model_name.title())
#     return model(opt, dataloader)

# if __name__ == '__main__':
#     main()