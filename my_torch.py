"""
MyTorch
"""

from abc import ABC, abstractmethod
from typing import OrderedDict, Type
#import mlx.core as mx
import numpy as np


class Tensor:
    def __init__(self, np_array, requires_grad = True):
        self.parameters = np_array
        self.grad = None

    def backward(self):
        """Compute gradients"""
        pass

    def zero_grad(self):
        """Clear gradients."""
        self.grad = np.zeros(shape=self.grad)

    def __array__(self):
        """Enable direct call by NumPy methods"""
        return self.parameters

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
    """Linear/FC module. Creates randomized weights and biases."""
    def __init__(self,in_dim:tuple,out_dim:int):
        flattened_in_dim = np.prod(in_dim) if isinstance(in_dim,tuple) else in_dim

        # Note to self: input_dim -> #rows, output_dim -> #columns
        self._w = np.random.randn(flattened_in_dim,out_dim)/np.sqrt(flattened_in_dim)
        self._b = np.zeros((out_dim,1))
    
    def forward(self,a_prev):
        """
        Input
        # Follow classical convention
        a_prev: shape = (in_dim, batch_size)
        self._w: shape = (in_dim, out_dim)
        self._b: shape = (out_dim)

        Output
        a: shape = (output_dim)
        """
        if len(a_prev.shape) >= 3:
            batch_size = a_prev.shape[-1]
            a_prev = a_prev.reshape(-1,batch_size)
        return np.dot(self._w.T,a_prev) + self._b

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