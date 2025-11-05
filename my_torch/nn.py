"""
MyTorch
"""

# region Imports
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import OrderedDict, Type, Callable, Optional
import numpy as np
from .tensor import Tensor, print_partial_d
# endregion

# region Settings
np.set_printoptions(precision=3)
# endregion

class Parameter(Tensor):
    def __init__(self, 
                 ndarray: Tensor, 
                 name: str | None = None):
        super().__init__(ndarray,
                         name,
                         requires_grad=True)

# region Module
class Module(ABC):
    def __init__(self):
        """
        Create efficient interal model storage for modules.
        """
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

    def __setattr__(self, name:str, value):
        """
        Implement direct assignment of modules as a model attributes, 
        creating internal and external pointers to underlying modules.
        """

        if isinstance(value,Parameter):
            self._modules[name] = value
            self._parameters[name] = value

        elif isinstance(value,Module):
            self._modules[name] = value

        object.__setattr__(self,name,value)
        

    def forward(self):
        """Abstract method for forward propagation."""
        raise NotImplementedError("Must override abstract method in subclasses.")
# endregion

# region Layer modules
class Linear(Module):
    """Linear/FC module. Creates randomized weights and biases."""
    def __init__(self,in_dim:tuple,out_dim:int):
        flattened_in_dim = np.prod(in_dim) if isinstance(in_dim,tuple) else in_dim

        # Note to self: input_dim -> #rows, output_dim -> #columns
        self._w = Parameter(np.random.randn(flattened_in_dim,out_dim)/np.sqrt(flattened_in_dim))
        self._b = Parameter(np.zeros((1,out_dim)))
    
    def __call__(self,
                 a_prev:Tensor):
        """
        Input
        # batch first because it is logical. E.g. x[0] gives the first batch
        a_prev: shape = (batch_size, in_dim)
        self._w: shape = (in_dim, out_dim)
        self._b: shape = (1, out_dim)

        Output
        a: shape = (batch_size, out_dim)
        """
        # Flatten a_prev it is of higher dims
        if len(a_prev.shape) >= 3:
            batch_size = a_prev.shape[0]
            a_prev = a_prev.reshape((batch_size, -1))
        return a_prev @ self._w + self._b

class ReLU(Module):
    def __call__(self,
                 a_prev:Tensor):
        return a_prev.relu()

class Sigmoid(Module):
    def __call__(self,
                 a_prev:Tensor):
        return a_prev.sigmoid()

class Tanh(Module):
    def __call__(self,
                 a_prev:Tensor):
        return a_prev.tanh()

class Dropout(Module):
    def __call__(self,
                 a_prev: Tensor,
                 p=0.5):
        return a_prev.dropout(p=p)

class Flatten(Module):
    def __call__(self,
                 a_prev:Tensor):
        return a_prev.flatten()
    
class Reshape(Module):
    def __call__(self,
                 a_prev:Tensor,
                 shape:tuple):
        return a_prev.reshape(shape=shape)

# endregion


# region Cost Functions
class CostFunction(ABC):
    def __init__(self):
        pass

    @ abstractmethod
    def __call__(self,outputs,targets):
        """
        outputs: shape = (batch_size, ...)
        targets: shape = (batch_size, ...)
        """
        pass

    def _compute_normalized_cost(self,outputs,targets,loss_calculation: Callable):
        """ 
        Apply loss function and compute average cost, taking into account the number of training examples

        Inputs
        outputs: shape = (batch_size, ...)
        targets: shape = (batch_size, ...)
        loss_calculation: a method that takes in y_hat and y to calculate 
                            loss for individual examples
        """
        if outputs.shape != targets.shape:
            raise ValueError(f"shape mismatch: outputs.shape {outputs.shape} not equal targets.shape {targets.shape}")

        batch_size = outputs.shape[0]
        loss = loss_calculation(outputs,targets)
        return np.sum(loss) / batch_size

class CrossEntropyLoss(CostFunction):
    def __call__(self,outputs,targets):
        self._compute_normalized_loss(
            outputs,
            targets,
            lambda y_hat, y: -np.log(y_hat)*y
            )
    
class BinaryCrossEntropyLoss(CostFunction):
    def __call__(self,outputs,targets):
        self._compute_normalized_loss(
            outputs,
            targets,
            lambda y_hat, y: -np.log(y_hat)*y -np.log(1-y_hat)*(1-y)
        )

class MeanSquaredError(CostFunction):
    def __call__(self,outputs,targets):
        self._compute_normalized_loss(
            outputs,
            targets,
            lambda y_hat, y: (y_hat-y)**2 / 2
        )
# endregion
