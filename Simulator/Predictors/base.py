#Abstract Predictor Interface
from abc import ABC, abstractmethod

class BranchPredictor(ABC):
    
    @abstractmethod
    def predict(self, pc: int) -> bool:
        #return True if branch is predicted taken.
        ...

    @abstractmethod
    def update(self, pc: int, taken: bool, target: int) -> None:
        #update predictor state after branch resolves
        ...

    @abstractmethod
    def storage_bits(self) -> int:
        #return total storage cost in bits
        ...

    