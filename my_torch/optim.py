# region Imports:
from abc import ABC, abstractmethod

# region Optimizers
class Optimizer(ABC): # Define common optimizer interface
    @ abstractmethod
    def __init__(self,model_params,lr):
        pass

    @ abstractmethod
    def step(self) -> None:
        # Update .data in place with gradients
        pass

class SGD(Optimizer):
    # Vanilla/base optimizer
    pass

class Adam(Optimizer):
    # Adaptive Momemtum Estimation
    pass
# end region