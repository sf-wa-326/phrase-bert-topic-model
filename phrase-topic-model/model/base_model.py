"""

Define the base class of the model, which all other developed model inherit from.

"""

from abc import ABC, abstractmethod
from torch import nn, optim

# from utils.config import Config # config to be implemented


class BaseModel(nn.Module):
    """base class of all models. Defines the interface contracts"""

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def interpret_topics(self):
        pass 



