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
model.train()


reals = model.input.data
noise = model.noise.data
noise = noise.new(reals.size()).normal_(0,0.2)
final = reals + noise
final = final.data

reals = model.visualizer.normalize(reals.cpu().numpy())
noise = model.visualizer.normalize(noise.cpu().numpy())
final = model.visualizer.normalize(final.cpu().numpy())

model.visualizer.vis.images(reals, win=4, opts={'title': 'Reals'})
model.visualizer.vis.images(noise, win=5, opts={'title': 'Noises'})
model.visualizer.vis.images(final, win=6, opts={'title': 'Output Image'})