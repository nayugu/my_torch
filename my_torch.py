"""
MyTorch
"""

from abc import ABC, abstractmethod
from typing import OrderedDict
#import mlx.core as mx
import numpy as np


class Module(ABC):
    def __init__(self):
        """
        Create efficient interal model storage for modules.
        """
        self._modules = OrderedDict()

    def __setattr__(self, name:str, value):
        """
        Implement direct assignment of modules as a model attributes, 
        creating internal and external pointers to underlying modules.
        """
        if isinstance(value,Module):
            self._modules[name] = value
        object.__setattr__(self,name,value)

    @abstractmethod
    def forward(self):
        """Abstract method for forward propagation."""
        raise NotImplementedError("Must override abstract method in subclasses.")

class Linear(Module):
    pass

class ReLU(Module):
    pass

class Sigmoid(Module):
    pass

class Tanh(Module):
    pass

class Dropout(Module):
    pass

class GeLU(Module):
    pass