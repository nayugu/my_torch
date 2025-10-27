"""
MyTorch
"""

# region Imports
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import OrderedDict, Type, Callable, Optional
#import mlx.core as mx
import numpy as np
# endregion


# region Tensor data type
class Tensor:
    # Core methods
    def __init__(self, 
                 ndarray,  
                 grad_to_parents = OrderedDict(), # Ordered dict storing gradients to send to parent modules
                 requires_grad = False): # Remember to set requires_grad to True by default for model parameters
        
        if isinstance(ndarray,Tensor):
            self.data = ndarray.data
            self.grad = ndarray.grad # Gradient used to update self
            self.grad_to_parents = ndarray.grad_to_parents # d/dparents (current module), to be sent to parent modules
            self.requires_grad = ndarray.requires_grad
        
        else:
            if not isinstance(ndarray,np.ndarray):
                ndarray = np.asarray(ndarray, dtype=np.float64)
            
            self.data = ndarray
            self.grad = None
            self.grad_to_parents = grad_to_parents
            self.requires_grad = requires_grad

    # TODO: Use slicing and views to enable converging outputs to slice the gradient in back prop
    def backward(self) -> None:
        """Compute the sum of gradients of given tensors with respect to graph leaves."""
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

    # Helper method to reduce boilerplate code
    def calc_output_and_grad(
            self: Tensor,
            other: Tensor,
            operation: Callable, 
            dself: Callable,
            dother: Callable
            ) -> Tensor:
        """Takes in inputs for an operation, then calculates the partial derivatives of the 
        operation output with respect to its inputs. Checks if other is a Tensor. 
        If so, then the operation uses self.data and other.data,
        and both self.grad_to_parents and other.grad_to_parents are updated. 
        If other is not a Tensor, the operation uses other directly in the operation
        and only self.grad_to_parents is updated.

        Args:
            self: Tensor
            other: Tensor
            operation: Callable - func(self, other) -> operation_output
            dself: Callable - func(self, other) -> d_operation_output / d_self
            dother: Callable - func(self, other) -> d_operation_output / d_other

        Example:
        calc_output_and_grad(
            self = Tensor[...]\n
            other = Tensor[...]\n
            operation = lambda s,o: s * o\n
            dself = lambda s,o: o\n
            dother = lambda s,o: s\n
            )
        """

        if isinstance(other, Tensor): # If other is a Tensor
            output = Tensor(operation(self.data,other.data)) # Operation output

            # Calculate partial derivatives. E.g. z = w * x
            output.grad_to_parents[self]  = dself(self.data,other.data)  # dzdw
            output.grad_to_parents[other] = dother(self.data,other.data) # dzdx 
        
        else: # If other is NOT a Tensor
            output = Tensor(operation(self.data,other)) # Operation output (use other directly)

            # Calculate partial derivatives.
            output.grad_to_parents[self] = dself(self.data,other)
            # Do not update dother, since it is not a Tensor with parameters to be updated

        return output
    
    
    # TODO: Create module objects when arithmatic operations are called for back propagation
    # region Arithmatic operations
    def __add__(self, other):
        # output = self + other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s + o,
            dself =     lambda s,o: 1,
            dother =    lambda s,o: 1
        )
    
    def __sub__(self, other):
        # output = self - other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s - o,
            dself =     lambda s,o: 1,
            dother =    lambda s,o: -1
        )
        
    def __mul__(self, other):
        # output = self * other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s * o,
            dself =     lambda s,o: o,
            dother =    lambda s,o: s
        )

    def __truediv__(self, other):
        # output = self / other = self * (other ** -1)
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s / o,
            dself =     lambda s,o: 1/o,
            dother =    lambda s,o: s * -(o ** -2)
        )
    
    def __pow__(self, other):
        # output = self ** other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s ** o,
            dself =     lambda s,o: o * (s ** (o-1)),
            dother =    lambda s,o: np.log(s) * s ** o
        )
    
    # TODO: Learn Matrix Calculus
    def __matmul__(self, other):
        # output = self @ other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s @ o,
            dself =     lambda s,o: o.T,
            dother =    lambda s,o: s.T
        )
    # endregion
    
    # region Reversed order arithmatic operations. E.g. a + b vs. b + a
    # Only activates if other is NOT a Tensor, because then other.__<operation>__() fails, 
    # so then Python checks self.__r<operation>__()
    def __radd__(self, other):
        # output = other + self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o + s,
            dself =     lambda s,o: 1,
            dother =    lambda s,o: 1
        )
    
    def __rsub__(self, other):
        # output = other - self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o - s,
            dself =     lambda s,o: -1,
            dother =    lambda s,o: 1
        )
        
    def __rmul__(self, other):
        # output = other * self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o * s,
            dself =     lambda s,o: o,
            dother =    lambda s,o: s
        )

    def __rtruediv__(self, other):
        # output = other / self = other * (self ** -1)
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o / s,
            dself =     lambda s,o: o * -(s ** -2),
            dother =    lambda s,o: 1/s,
        )
    
    def __rpow__(self, other):
        # output = other ** self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o ** s,
            dself =     lambda s,o: np.log(o) * o ** s,
            dother =    lambda s,o: s * (o ** (s-1)),
        )
    
    def __rmatmul__(self, other):
        # output = other @ self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o @ s,
            dself =     lambda s,o: o.T,
            dother =    lambda s,o: s.T
        )
    # endregion
    
    # region Unitary operations
    def __neg__(self):
        return Tensor(-self.data)
    
    def __abs__(self):
        return Tensor(np.abs(self.data))
    # endregion
    
    # region Helper methods
    # TODO: Consider removing this
    def append_derivative(self, other, derivative):
        """Helper method to store derivatives. Done because appending, and then summing with a vectorized approach
         because this is faster than summing when encountering every derivative"""
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
