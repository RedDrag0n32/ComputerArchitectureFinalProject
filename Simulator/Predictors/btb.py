
class BTBEntry:
    def __init__(self):
        self.valid = False
        self.tag = 0
        self.target = 0


class BTB:
    def __init__(self, num_entries: int = 256):
        self.size = num_entries
        self.entries = [BTBEntry() for _ in range(num_entries)]
        self.idx_bits = (num_entries - 1).bit_length()

    def _index(self, pc: int) -> int:
        return (pc >> 2) & (self.size - 1)
    
    def _tag(self, pc: int) -> int:
        return pc >> (2 + self.idx_bits)
    
    def lookup(self, pc: int):
        idx = self._index(pc)
        entry = self.entries[idx]

        if entry.valid and entry.tag == self._tag(pc):
            return entry.target
        
        return None
    
    def update(self, pc: int, target: int) -> None:
        idx = self._index(pc)
        entry = self.entries[idx]

        entry.valid = True
        entry.tag = self._tag(pc)
        entry.target = target & 0xFFFFFFFF

    def storage_bits(self) -> int:
        tag_bits = 32 - 2 - self.idx_bits
        return self.size * (1+ tag_bits + 32)


    # def access(self, pc, update=False, target=None):
    #     index = (pc >> 2) & (self.num_entries - 1)
    #     tag = pc >> (2 + (self.num_entries - 1).bit_length())



    #     entry = self.entries[index]

    #     if update:
    #         entry["valid"] = True
    #         entry["tag"] = tag
    #         entry["target"] = target
    #         return None



    #     if entry["valid"] and entry["tag"] == tag:
    #         return entry["target"]



    #     return None


