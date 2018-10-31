"""
Losses
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
# LIBRARIES
import torch
from dataclasses import dataclass
from dataclasses import asdict

##
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
    adv  : torch.Tensor = torch.empty(1)
    rec  : torch.Tensor = torch.empty(1)
    enc  : torch.Tensor = torch.empty(1)

@dataclass
class Loss:
    # Loss Values
    d: type(LossD) = LossD()
    g: type(LossG) = LossG()

    # Loss Functions
    bce: type(torch.nn.BCELoss()) = torch.nn.BCELoss()
    l1 : type(torch.nn.L1Loss()) = torch.nn.L1Loss()
    l2 : type(torch.nn.MSELoss()) = torch.nn.MSELoss()

##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)