from __future__ import annotations
from typing import Callable, Optional
import numpy as np

# region Settings
np.set_printoptions(precision=3)
# end region

# region Tensor data type
class Tensor:
    # Class attributes
    auto_name = False # By default, do not auto name Tensors
    equation_precision = 2

    # Core methods
    def __init__(self, 
                 ndarray, 
                 name:Optional[str] = None,
                 requires_grad = False # Do not update gradients by default to save memory
                 ): # Remember to set requires_grad to True by default for model parameters
        
        if isinstance(ndarray,Tensor):
            self.data = ndarray.data
        
        else:
            if not isinstance(ndarray,np.ndarray):
                ndarray = np.asarray(ndarray, dtype=np.float64)
            self.data = ndarray
        
        self._name = name
        self.grad: Optional[np.ndarray] = None
        self.partial_d: dict[Tensor,np.ndarray] = {} # With respect to __: partial derivative of self
        self.requires_grad = requires_grad

        # Used to check when to apply matrix multiplication for chain rule
        # and then store the dimension to broadcast across
        self._left_matmul:Optional[int] = None 
        self._right_matmul:Optional[int] = None

    # TODO: Use slicing and views to enable converging outputs to slice the gradient in back prop
    
    # region Calculus
    def recursive_chain_rule(node: Tensor,
                            leaves: dict[Tensor,np.ndarray] = {}, accumulated_grad=1.0):
        if leaves is None:
            leaves: dict[Tensor,np.ndarray] = {}
            
        if node.partial_d == {}:
            return node
        else:
            for sub_node, d_sub_node in node.partial_d.items():
                # Current node was created through matrix multiplication
                if sub_node._left_matmul:
                    if isinstance(accumulated_grad,float) or \
                        isinstance(accumulated_grad,np.float64): # =1.0
                        
                        ones_shape = list(sub_node.shape) # [..., rows, cols]
                        ones_shape[-1] = sub_node._left_matmul  # Last dim matches the contracted dimension, i.e. cols
                        accumulated_grad = np.ones(shape=tuple(ones_shape))
                    new_accumulated_grad =accumulated_grad @ d_sub_node

                elif sub_node._right_matmul:
                    if isinstance(accumulated_grad,float) or \
                        isinstance(accumulated_grad,np.float64): # =1.0
                        
                        ones_shape = list(sub_node.shape) # [..., rows, cols]
                        ones_shape[-2] = sub_node._right_matmul  # Second-to-last dim matches contracted dimension, i.e. rows
                        accumulated_grad = np.ones(shape=tuple(ones_shape))
                    new_accumulated_grad = d_sub_node @ accumulated_grad

                else:
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
        if None in leaves: del leaves[None]

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
        data_str = np.array2string(self.data, precision=3, suppress_small=True)
        grad_str = np.array2string(self.grad, precision=3, suppress_small=True) if self.grad is not None else "None"
        return f"\n\n{self.name}\n-----\ndata:\n{data_str}\n-----\ngrad:\n{grad_str}\n"
    
    def __repr__(self):
        if self._name == None:
            name = ""
        else:
            name = self._name
        return f"Tensor '{name}' of shape {self.shape}"
    

    @ property
    def shape(self):
        return self.data.shape
    
    @ property
    def name(self):
        if self._name:
            return self._name
        else:
            return f"<T{self.data.shape}>"
        
    @ name.setter
    def name(self,name:str):
        self._name = str(name)
    
    # Methods to create automatic computation graphs

    # Helper method to reduce boilerplate code
    def calc_output_and_grad(
            self: Tensor,
            other: Optional[Tensor],
            operation: Callable, 
            dself: Callable,
            dother: Callable,
            op_name: str # Name of operation. E.g. A+B
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

        if Tensor.auto_name and output._name == None:
            output._name = concise(op_name)
        
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
            dother =    lambda s,o: 1.,
            op_name = f'({self.name} + {other.name})'
        )
    
    def __sub__(self, other):
        # output = self - other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s - o,
            dself =     lambda s,o: 1.,
            dother =    lambda s,o: -1.,
            op_name = f'({self.name} - {other.name})'
        )
        
    def __mul__(self, other):
        # output = self * other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s * o,
            dself =     lambda s,o: o,
            dother =    lambda s,o: s,
            op_name = f'({self.name} * {other.name})'
        )

    def __truediv__(self, other):
        # output = self / other = self * (other ** -1)
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s / o,
            dself =     lambda s,o: 1/o,
            dother =    lambda s,o: s * -(o ** -2),
            op_name = f'({self.name} / {other.name})'
        )
    
    def __pow__(self, other):
        # output = self ** other
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s ** o,
            dself =     lambda s,o: o * (s ** (o-1)),
            dother =    lambda s,o: np.log(s) * s ** o,
            op_name = f'({self.name} ** {other.name})'
        )
    
    # TODO: Learn Matrix Calculus
    def __matmul__(self, other):
        # output = self @ other
        self._left_matmul = other.shape[1]
        other._right_matmul = self.shape[0]
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: s @ o,
            dself =     lambda s,o: o.T,
            dother =    lambda s,o: s.T,
            op_name = f'({self.name} @ {other.name})'
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
            dother =    lambda s,o: 1.,
            op_name = f'({other.name} + {self.name}'
        )
    
    def __rsub__(self, other):
        # output = other - self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o - s,
            dself =     lambda s,o: -1.,
            dother =    lambda s,o: 1.,
            op_name = f'({other.name} - {self.name})'
        )
        
    def __rmul__(self, other):
        # output = other * self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o * s,
            dself =     lambda s,o: o,
            dother =    lambda s,o: s,
            op_name = f'({other.name} * {self.name})'
        )

    def __rtruediv__(self, other):
        # output = other / self = other * (self ** -1)
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o / s,
            dself =     lambda s,o: o * -(s ** -2),
            dother =    lambda s,o: 1/s,
            op_name = f'({other.name} / {self.name})'
        )
    
    def __rpow__(self, other):
        # output = other ** self
        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o ** s,
            dself =     lambda s,o: np.log(o) * o ** s,
            dother =    lambda s,o: s * (o ** (s-1)),
            op_name = f'({other.name} ** {self.name})'
        )
    
    def __rmatmul__(self, other):
        # output = other @ self
        other._left_matmul = self.shape[1]
        self._right_matmul = other.shape[0]

        return self.calc_output_and_grad(
            other,
            operation = lambda s,o: o @ s,
            dself =     lambda s,o: o.T,
            dother =    lambda s,o: s.T,
            op_name = f'({other.name} @ {self.name})'
        )
    
    # endregion
    
    # region Unitary operations
    def __neg__(self):
        # output = -self
        return self.calc_output_and_grad(
            other=None,
            operation = lambda s: -s,
            dself =     lambda s: -1.,
            dother =    None,
            op_name = f'(-{self.name})'
        )
    
    def __abs__(self):
        # output = |self| = np.abs(self)
        return self.calc_output_and_grad(
            other=None,
            operation = lambda s: np.abs(s),
            dself =     lambda s: np.sign(s),
            dother =    None,
            op_name = f'abs({self.name})'
        )
    
    def __sum__(self, axis=None):
        # output = np.sum(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.sum(s,axis=axis),
            dself = lambda s: np.ones_like(s),
            dother = None,
            op_name = f'sum({self.name},axis={axis})'
        )
    
    # Trig functions
    def sin(self):
        # output = sin(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.sin(s),
            dself = lambda s: np.cos(s),
            dother = None,
            op_name = f'{self.name}.sin'
        )

    def cos(self):
        # output = cos(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.cos(s),
            dself = lambda s: -np.sin(s),
            dother = None,
            op_name = f'{self.name}.cos'
        )

    def tan(self):
        # output = tan(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.tan(s),
            dself = lambda s: 1 / (np.cos(s)**2),
            dother = None,
            op_name = f'{self.name}.tan'
        )
    
    # region Activation Functions
    def relu(self):
        # output = ReLU(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.maximum(0,s),
            dself = lambda s: (s > 0) * 1.0,
            dother = None,
            op_name = f'{self.name}.relu'
        )
    
    def leaky_relu(self, alpha=0.01):
        # output = LeakyReLU(self)
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: np.maximum(alpha*s,s),
            dself = lambda s: (s > 0) * 1.0 + alpha * (s <= 0),
            dother = None,
            op_name = f'{self.name}.leaky_relu'
        )

    def sigmoid(self):
        # output = Sigmoid(self)
        sigmoid = 1 / (1 + np.exp(-self.data))
        return self.calc_output_and_grad(
            other=None,
            operation=lambda s: sigmoid,
            dself = lambda s: sigmoid * (1-sigmoid),
            dother = None,
            op_name = f'{self.name}.sigmoid'
        )

    def softmax(self):
        # output = Softmax(self)
        if self.data.ndim > 2:
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
            dother = None,
            op_name = f'{self.name}.softmax'
        )
    # endregion
    
    # region Loss Functions
    def mean_squared_error_loss(predictions,targets):
        n = predictions.data.size
        return predictions.calc_output_and_grad(
            other=targets,
            operation = lambda p,t: np.sum((p - t) ** 2) / (2 * n),
            dself =     lambda p,t: (p - t) / n,
            dother =    lambda p,t: -(p - t) / n,
            op_name = f'MSE_Loss({predictions.name},{targets.name})'
        )

    def cross_entropy_loss(predictions,targets,residual = 1e-7):
        n = predictions.data.size
        return predictions.calc_output_and_grad(
            other=targets,
            operation = lambda p,t: np.sum(t*-np.log(p+residual)) / n,
            dself =     lambda p,t: -t/(p+residual) / n,
            dother =    lambda p,t: -np.log(p+residual) / n,
            op_name = f'CE_Loss({predictions.name},{targets.name})'
        )

    def binary_cross_entropy_loss(predictions,targets,residual = 1e-7):
        n = predictions.data.size
        return predictions.calc_output_and_grad(
            other=targets,
            operation = lambda p,t: np.sum(t*-np.log(p+residual) + (1-t)*-np.log(1-p+residual)) / n,
            dself =     lambda p,t: (-t/(p+residual) + (1-t)/(1-p+residual)) / n,
            dother =    lambda p,t: (-np.log(p+residual) + np.log(1-p+residual)) / n,
            op_name = f'BCE_Loss({predictions.name},{targets.name})'
        )
    # endregion
# endregion


# region Global Helper Methods
def print_partial_d(partial_d_dict):
        """
        Prints the partial derivatives stored within a dictionary
        """
        output = "\nPartial Derivatives\n"
        for tensor,partial_d in partial_d_dict.items():
            if tensor: # Not none
                if isinstance(partial_d,np.ndarray) and len(partial_d) > 1:
                    next_line = "\n"
                else:
                    next_line = ""
                output += (f"{tensor.name}{tensor.data} : {next_line}{partial_d}\n\n")

        print(output)
        return output

def concise(str:str,
            replace_symbol:str="..."):
    """
    Returns a more concise version of an equation with nested parenthesis
    by eliminating all parenthesis after a certain depth.
    """
    max_depth = Tensor.equation_precision # E.g. (5 + (4 - ?))) is ok, but (5 + (4 - (?) * ?)))) is not
    depth = 0
    content = ""
    #stack = []
    for i in range(len(str)):
        if str[i] == "(":
            depth += 1
            #stack.append(depth)

            if depth == max_depth + 1:
                content += replace_symbol

        if depth <= max_depth:
            content += str[i]

        if str[i] == ")":
            depth -= 1
            #stack.pop()

        #print(stack,10*" ",str[:i+1],10*" ",content)
    return content

# endregion