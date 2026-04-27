
from .base import BranchPredictor

SN, WN, WT, ST = 0, 1, 2, 3  # Strong/Weak Not Taken, Weak/Strong Taken


class TwoBitPredictor(BranchPredictor):
    def __init__(self, table_bits: int = 10):
        self.k = table_bits                          
        self.size = 1 << self.k     
        self.table = [WT] * (1 << self.k)

    def _index(self, pc: int) -> int:
        return (pc >> 2) & ((1 << self.k) - 1)

    def predict(self, pc: int) -> bool:
        state = self.table[self._index(pc)]
        return state >= WT  # WT or ST → taken

    def update(self, pc: int, taken: bool, target: int) -> None:
        idx = self._index(pc)
        s = self.table[idx]
        if taken:
            self.table[idx] = min(s + 1, ST)
        else:
            self.table[idx] = max(s - 1, SN)


    def storage_bits(self) -> int:
        return 2 * (1 << self.k)
