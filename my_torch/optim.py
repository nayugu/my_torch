# region Imports:
from abc import ABC, abstractmethod
# endregion

# region Optimizers
class Optimizer(ABC): # Define common optimizer interface
    def __init__(self,model_params,lr):
        self.model_params = model_params
        self.lr = lr

    @ abstractmethod
    def step(self) -> None:
        # Update .data in place with gradients
        pass

class SGD(Optimizer):
    # Vanilla/base optimizer
    def step(self) -> None:
        pass

class Adam(Optimizer):
    # Adaptive Momemtum Estimation
    def step(self) -> None:
        pass
# end region