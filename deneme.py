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
