from .base import BranchPredictor


class GsharePredictor(BranchPredictor):
    """
    Gshare predictor:
    - Global History Register (GHR) of H bits
    - Pattern History Table (PHT) of 2-bit saturating counters, size 2^H
    - index = (PC >> 2) low H bits XOR GHR
    """

    def __init__(self, history_bits: int = 10):
        if history_bits <= 0:
            raise ValueError("history_bits must be >= 1")

        self.H = history_bits
        self.ghr = 0
        self._mask = (1 << self.H) - 1
        # 2-bit counters: 0,1 => predict not taken; 2,3 => predict taken
        self.pht = [2] * (1 << self.H)  # weakly taken initial state

    def _index(self, pc: int) -> int:
        pc_bits = (pc >> 2) & self._mask
        return (pc_bits ^ self.ghr) & self._mask

    def predict(self, pc: int) -> bool:
        return self.pht[self._index(pc)] >= 2

    def update(self, pc: int, taken: bool, target: int) -> None:
        idx = self._index(pc)
        ctr = self.pht[idx]

        if taken:
            if ctr < 3:
                ctr += 1
        else:
            if ctr > 0:
                ctr -= 1

        self.pht[idx] = ctr

        self.ghr = ((self.ghr << 1) | (1 if taken else 0)) & self._mask

    def storage_bits(self) -> int:
        return (2 * len(self.pht)) + self.H
