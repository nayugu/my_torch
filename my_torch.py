"""
MyTorch # test anonymity of nayugu 5
"""

from abc import ABC, abstractmethod
from typing import OrderedDict, Type, Callable
#import mlx.core as mx
import numpy as np

# Data Types
class Tensor:
    def __init__(self, np_array, requires_grad = False): # Set requires_grad to True by default for model parameters
        if not isinstance(np_array,np.ndarray):
            np_array = np.array(np_array)
        
        self.data = np_array
        self.grad = None
        self.requires_grad = requires_grad

    def backward(self):
        """Compute gradients"""
        pass

    def zero_grad(self):
        """Clear gradients."""
        self.grad = None

    def __array__(self):
        """Enable direct call by NumPy methods"""
        return self.data
    
    @ property
    def shape(self):
        return self.data.shape

# Module
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

# Activation Functions
class Linear(Module):
    """Linear/FC module. Creates randomized weights and biases."""
    def __init__(self,in_dim:tuple,out_dim:int):
        flattened_in_dim = np.prod(in_dim) if isinstance(in_dim,tuple) else in_dim

        # Note to self: input_dim -> #rows, output_dim -> #columns
        self._w = Tensor(np.random.randn(flattened_in_dim,out_dim)/np.sqrt(flattened_in_dim))
        self._b = Tensor(np.zeros((out_dim,1)))
    
    def forward(self,a_prev):
        """
        Input
        # Follow classical convention
        a_prev: shape = (in_dim, batch_size)
        self._w: shape = (in_dim, out_dim)
        self._b: shape = (out_dim)

        Output
        a: shape = (out_dim, batch_size)
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

# Loss Functions
class CostFunction(ABC):
    def __init__(self):
        pass

    @ abstractmethod
    def __call__(self,outputs,targets):
        """
        outputs: shape = (..., batch_size)
        targets: shape = (..., batch_size)
        """
        pass

    def _compute_normalized_cost(self,outputs,targets,loss_calculation: Callable):
        """ 
        Apply loss function and compute average cost, taking into account the number of training examples

        Inputs
        outputs: shape = (..., batch_size)
        targets: shape = (..., batch_size)
        loss_calculation: a method that takes in y_hat and y to calculate 
                            loss for individual examples
        """
        if outputs.shape != targets.shape:
            raise ValueError(f"shape mismatch: outputs.shape {outputs.shape} not equal targets.shape {targets.shape}")

        batch_size = outputs.shape[-1]
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
