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

# region Global Helper Methods
def print_partial_d(partial_d_dict):
        """
        Prints the partial derivatives stored within a dictionary
        """
        output = "\nPartial Derivatives\n"
        for tensor,partial_d in partial_d_dict.items():
            if tensor: # Not none
                if tensor.name != None:
                    name = tensor.name
                else:
                    name = type(tensor)
                output += (f"{name}{tensor.data} : {partial_d}\n")
        print(output)
        return output
# endregion

# region Tensor data type
class Tensor:
    # Core methods
    def __init__(self, 
                 ndarray, 
                 name:Optional[str] = None,
                 requires_grad = False): # Remember to set requires_grad to True by default for model parameters
        
        if isinstance(ndarray,Tensor):
            self.data = ndarray.data
        
        else:
            if not isinstance(ndarray,np.ndarray):
                ndarray = np.asarray(ndarray, dtype=np.float64)
            self.data = ndarray
        
        self.name = name
        self.grad: Optional[np.ndarray] = None
        self.partial_d: dict[Tensor,np.ndarray] = {} # With respect to __: partial derivative of self
        self.requires_grad = requires_grad

    # TODO: Use slicing and views to enable converging outputs to slice the gradient in back prop
    
    # region Calculus
    def recursive_chain_rule(node: Tensor,
                            leaves: dict[Tensor,np.ndarray] = {}, accumulated_grad=1):
        if leaves is None:
            leaves: dict[Tensor,np.ndarray] = {}
            
        if node.partial_d == {}:
            return node
        else:
            for sub_node, d_sub_node in node.partial_d.items():
                new_accumulated_grad = accumulated_grad * d_sub_node
                leaf = sub_node.recursive_chain_rule(leaves=leaves,
                                                     accumulated_grad=new_accumulated_grad)
                if leaf in leaves:
                    leaves[leaf] += new_accumulated_grad
                else:
                    leaves[leaf] = new_accumulated_grad
        
    def backward(self):
        leaves: dict[Tensor,np.ndarray] = {}
        self.recursive_chain_rule(leaves=leaves)
        del leaves[None]

        result = {leaf: grad.copy() if isinstance(grad, np.ndarray)
                  else grad for leaf, grad in leaves.items()}

        for leaf,leaf_grad in leaves.items():
            leaf.receive_grad(leaf_grad)

        return result
    
    def receive_grad(self,grad):
        """Method called to receive gradient"""
        if self.requires_grad:
            if self.grad is None:
                self.grad = grad
            else:
                if (isinstance(grad,np.float64) or
                    isinstance(grad,float) or
                    grad.shape == self.grad.shape):
                    self.grad += grad
                else:
                    raise ValueError("Shape of incoming gradient does not matching existing gradient.")
    

    def zero_grad(self):
        """Clear gradients."""
        self.grad = None
    # endregion

    # region Helper Methods
    def __array__(self):
        """Enable direct call by NumPy methods"""
        return self.data
    
    def __str__(self):
        return f"Tensor:\n{str(self.data)}\nGradients:{str(self.grad)}"
    

    @ property
    def shape(self):
        return self.data.shape
    
    # Methods to create automatic computation graphs

    # Helper method to reduce boilerplate code
    def calc_output_and_grad(
            self: Tensor,
            other: Optional[Tensor],
            operation: Callable, 
            dself: Callable,
            dother: Callable
            ) -> Tensor:
        """Takes in inputs for an operation, then calculates the partial derivatives of the 
        operation output with respect to its inputs. Checks if other is a Tensor. 
        If so, then the operation uses self.data and other.data,
        and both self.partial_d and other.partial_d are updated. 
        If other is not a Tensor, the operation uses other directly in the operation
        and only self.partial_d is updated.

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
            output.partial_d[self] = dself(self.data,other.data)  # dzdw, np.ndarray
            output.partial_d[other] = dother(self.data,other.data) # dzdx, np.ndarray
        
        else: # If other is NOT a Tensor

            if other is None: # If it is a unitary operation
                output = Tensor(operation(self.data))

                # Calculate partial derivatives.
                output.partial_d[self] = dself(self.data) # np.ndarray
            

            else: # Non-unitary operation
                output = Tensor(operation(self.data,other)) # Operation output (use other directly)

                # Calculate partial derivatives.
                output.partial_d[self] = dself(self.data,other) # np.ndarray

        return output
    # endregion
    
    
    # TODO: Create module objects when arithmatic operations are called for back propagation
    # region Arithmatic operations
    def __add__(self, other):
        # output = self + other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s + o,
            dself =     lambda s,o: 1.,
            dother =    lambda s,o: 1.
        )
    
    def __sub__(self, other):
        # output = self - other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s - o,
            dself =     lambda s,o: 1.,
            dother =    lambda s,o: -1.
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
            dself =     lambda s,o: 1.,
            dother =    lambda s,o: 1.
        )
    
    def __rsub__(self, other):
        # output = other - self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o - s,
            dself =     lambda s,o: -1.,
            dother =    lambda s,o: 1.
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
        # output = -self
        return self.calc_output_and_grad(
            other=None,
            operation = lambda s: -s,
            dself =     lambda s: -1.,
            dother =    None
        )
    
    def __abs__(self):
        # output = |self| = np.abs(self)
        return self.calc_output_and_grad(
            other=None,
            operation = lambda s: np.abs(s),
            dself =     lambda s: np.sign(s),
            dother =    None
        )
    
    def __sum__(self, axis=None):
        # output = np.sum(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.sum(s,axis=axis),
            dself = lambda s: np.ones_like(s),
            dother = None
        )
    
    # Trig functions
    def sin(self):
        # output = sin(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.sin(s),
            dself = lambda s: np.cos(s),
            dother = None
        )

    def cos(self):
        # output = cos(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.cos(s),
            dself = lambda s: -np.sin(s),
            dother = None
        )

    def tan(self):
        # output = tan(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.tan(s),
            dself = lambda s: 1 / (np.cos(s)**2),
            dother = None
        )
    
    # Common activation functions
    def relu(self):
        # output = ReLU(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.maximum(0,s),
            dself = lambda s: (s > 0) * 1.0,
            dother = None
        )
    
    def leaky_relu(self, alpha=0.01):
        # output = LeakyReLU(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.maximum(alpha*s,s),
            dself = lambda s: (s > 0) * 1.0 + alpha * (s <= 0),
            dother = None
        )

    def sigmoid(self):
        # output = Sigmoid(self)
        sigmoid = 1 / (1 + np.exp(-self.data))
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: sigmoid,
            dself = lambda s: sigmoid * (1-sigmoid),
            dother = None
        )

    def softmax(self):
        # output = Softmax(self)
        if self.data.ndims > 2:
            raise ValueError("Softmax not supported for ndarrays currently. Please use a 1D or 2D input.")
        
        def forward(s):
            s = np.squeeze(s)  # Remove singleton dimensions
            
            if s.ndim == 1:
                s_shifted = s - np.max(s)
                e = np.exp(s_shifted)
                return e / np.sum(e)
            else:  # ndim >= 2, treat first dim as batch
                s_shifted = s - np.max(s, axis=-1, keepdims=True)
                e = np.exp(s_shifted)
                return e / np.sum(e, axis=-1, keepdims=True)
            
        def jacobian(s):
            s = np.squeeze(s)
            softmax_out = forward(s)
            
            if s.ndim == 1:
                return np.diag(softmax_out) - np.outer(softmax_out, softmax_out)
            else:
                batch_size, n = softmax_out.shape
                jac = np.zeros((batch_size, n, n))
                for i in range(batch_size):
                    jac[i] = np.diag(softmax_out[i]) - np.outer(softmax_out[i], softmax_out[i])
                return jac

        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: forward(s),
            dself = lambda s: jacobian(s),
            dother = None
        )

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
