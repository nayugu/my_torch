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
    # Vanilla/base optimizer
    def step(self) -> None:
        for param in self.model_params:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad

class Adam(Optimizer):
    # Adaptive Momemtum Estimation
    def step(self) -> None:
        pass
# end region