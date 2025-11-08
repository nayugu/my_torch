# region Imports:
from abc import ABC, abstractmethod
# endregion

# region Optimizers
class Optimizer(ABC): # Define common optimizer interface
    def __init__(self,model_params,lr):
        self.model_params = list(model_params)
        self.lr = lr

    @ abstractmethod
    def step(self) -> None:
        # Update .data in place with gradients
        pass

    def zero_grad(self) -> None:
        for p in self.model_params:
            p.grad = None

    def __str__(self):
        output = ""
        for p in self.model_params:
            output += f"{p.name}\n"
        return output


class SGD(Optimizer):
    # Standard Gradient Descent
    def step(self) -> None:
        for param in self.model_params:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad

class Adam(Optimizer):
    # Adaptive Momemtum Estimation
    def __init__(self, model_params, lr, b1=0.9, b2=0.999):
        super().__init__(model_params, lr)
        self.b1 = b1 # for Momentum
        self.b2 = b2 # for Squared gradients
        self.t = 0 # time step

        self.v = {param:0 for param in self.model_params}
        self.s = {param:0 for param in self.model_params}

    def step(self) -> None:
        self.t += 1
        for param in self.model_params:
            if param.grad is None:
                continue
            self.v[param] = self.b1 * self.v[param] + (1-self.b1) * param.grad
            self.s[param] = self.b2 * self.s[param] + (1-self.b2) * (param.grad ** 2)

            # Correct bias for initial time steps
            _v = self.v[param] / (1-self.b1**self.t)
            _s = self.s[param] / (1-self.b2**self.t)
            weighted_grad = _v/((_s**0.5)+1e-16)
            
            param.data -= self.lr * weighted_grad

        
# end region