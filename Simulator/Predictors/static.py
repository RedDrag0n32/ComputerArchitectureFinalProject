
from .base import BranchPredictor

class StaticAlwaysTakenPredictor(BranchPredictor):

    #Static always taken predictor
    #predicts every branch taken
    #does not learn or store history

    def predict(self, pc: int) -> bool:
        return True
    
    def update(self, pc: int, taken: bool, target: int) -> None:
        pass

    def storage_bits(self) -> int:
        return 0
    
    