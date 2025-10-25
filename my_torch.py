"""
MyTorch
"""

# region Imports
from abc import ABC, abstractmethod
from typing import OrderedDict, Type, Callable, Optional
#import mlx.core as mx
import numpy as np
# endregion


# region Tensor data type
class Tensor:
    # Core methods
    def __init__(self, np_array, requires_grad = False): # Set requires_grad to True by default for model parameters
        if isinstance(np_array,Tensor):
            self.data = np_array.data
            self.grad = np_array.grad
            self.requires_grad = np_array.requires_grad
        
        else:
            if not isinstance(np_array,np.ndarray):
                np_array = np.array(np_array)
            
            self.data = np_array
            self.grad = None
            self.requires_grad = requires_grad

        # Keep track of forward prop inputs and outputs for computation graph
        # format: OrderedDict[Tensor, [Callable derivatives]]
        self.f_inputs = OrderedDict() 
        self.f_outputs = OrderedDict()

    def backward(self) -> None:
        """Compute gradients"""
        pass

    def zero_grad(self):
        """Clear gradients."""
        self.grad = None

    def __array__(self):
        """Enable direct call by NumPy methods"""
        return self.data
    
    def receive_grad(self,grad):
        """Method called to receive gradient"""
        if self.grad is None:
            self.grad = grad
        else:
            if grad.shape == self.grad.shape:
                self.grad += grad
            else:
                raise ValueError("Shape of incoming gradient does not matching existing gradient.")
    
    @ property
    def shape(self):
        return self.data.shape
    
    # Methods to create automatic computation graphs
    
    # TODO: Create module objects when arithmatic operations are called for back propagation
    # region Arithmatic operations
    def __add__(self, other):
        if isinstance(other, Tensor): 
            return Tensor(self.data + other.data) 
        else: 
            return Tensor(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, Tensor): 
            return Tensor(self.data - other.data) 
        else: 
            return Tensor(self.data - other)
        
    def __mul__(self, other):
        if isinstance(other, Tensor): 
            return Tensor(self.data * other.data) 
        else: 
            return Tensor(self.data * other)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor): 
            return Tensor(self.data / other.data) 
        else: 
            return Tensor(self.data / other)
        
    def __pow__(self, other):
        derivative = lambda x: other*(x**(other-1))
        if isinstance(other, Tensor):
            self.append_derivative(other,derivative)
            return Tensor(self.data ** other.data)
        else:
            return Tensor(self.data ** other)
        
    def __matmul__(self, other):
        self.outputs[other] = other
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data,other.data))
        else:
            return Tensor(np.dot(self.data,other))
    # endregion
    
    # region Reversed order arithmatic operations. E.g. a + b vs. b + a
    # Only activates if other is NOT a Tensor, because then other.__<operation>__() fails, 
    # so then Python checks self.__r<operation>__()
    def __radd__(self, other):
        return Tensor(other + self.data)
    
    def __rsub__(self, other):
        return Tensor(other - self.data)
        
    def __rmul__(self, other):
        return Tensor(other * self.data)

    def __rtruediv__(self, other):
        return Tensor(other / self.data) 
    
    def __rpow__(self, other):
        return Tensor(other ** self.data)
    
    def __rmatmul__(self,other):
        return Tensor(np.dot(other,self.data))
    # endregion
    
    # region Unitary operations
    def __neg__(self):
        return Tensor(-self.data)
    
    def __abs__(self):
        return Tensor(np.abs(self.data))
    # endregion
    
    # region Helper methods
    def append_derivative(self, other, derivative):
        """Helper method to match derivatives to inputs and store them"""
        if other in self.f_inputs:
            self.f_inputs[other].append(derivative)
        else:
            self.f_inputs[other] = [derivative]
    # endregion
# endregion


# region Module
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
# endregion
# region Layer modules
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
        # batch first because it is logical. E.g. x[0] gives the first batch
        a_prev: shape = (batch_size, in_dim)
        self._w: shape = (in_dim, out_dim)
        self._b: shape = (out_dim)

        Output
        a: shape = (batch_size, out_dim)
        """
        # Flatten a_prev it is of higher dims
        if len(a_prev.shape) >= 3:
            batch_size = a_prev.shape[0]
            a_prev = a_prev.reshape(batch_size, -1)
        return np.dot(a_prev,self._w) + self._b

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

class Flatten(Module):
    pass

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
