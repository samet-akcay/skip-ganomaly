# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

import torch
from dataclasses import dataclass
from dataclasses import asdict

@dataclass
class LossD:
    """ Loss val for netD
    """
    total: torch.Tensor = torch.empty(1)
    real : torch.Tensor = torch.empty(1)
    fake : torch.Tensor = torch.empty(1)

@dataclass
class LossG:
    """ Loss val for netG
    """
    total: torch.Tensor = torch.empty(1)
    bce  : torch.Tensor = torch.empty(1)
    rec  : torch.Tensor = torch.empty(1)

@dataclass
class Loss:
    # Loss Values
    d: type(LossD) = LossD()
    g: type(LossG) = LossG()

    # Loss Functions
    bce: type(torch.nn.BCELoss()) = torch.nn.BCELoss()
    l1 : type(torch.nn.L1Loss()) = torch.nn.L1Loss()
    l2 : type(torch.nn.MSELoss()) = torch.nn.MSELoss()

loss = Loss()



class Input:
    def __init__(self, opt):
        img_size = (opt.batchsize, opt.nc, opt.isize, opt.isize)
        img_type = torch.float32
        gts_size = (opt.batchsize,)
        gts_type = torch.long
        device   = torch.device("cuda:0" if opt.gpus != -1 else "cpu")

        self.img = torch.empty(size=img_size, dtype=img_type, device=device)
        self.gts = torch.empty(size=gts_size, dtype=gts_type, device=device)
        self.lbl = torch.empty(size=gts_size, dtype=img_type, device=device)

class Output:
    def __init__(self, opt):
        img_size = (opt.batchsize, opt.nc, opt.isize, opt.isize)
        img_type = torch.float32      
        device   = torch.device("cuda:0" if opt.gpus != -1 else "cpu")

        self.img = torch.empty(size=img_size, dtype=img_type, device=device)
        self.real_feats = None
        self.fake_feats = None
        self.real_score = torch.empty(size=opt.batchsize, dtype=torch.float32, device=device)
        self.fake_score = torch.empty(size=opt.batchsize, dtype=torch.float32, device=device)
 

# input = Input(opt)