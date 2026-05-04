
class BranchTargetBuffer:
    def __init__(self, num_entries=16):
        self.num_entries = num_entries
        self.entries = [
            {"valid": False, "tag": 0, "target": 0}
              for _ in range(num_entries)
              ]


    def access(self, pc, update=False, target=None):
        index = (pc >> 2) & (self.num_entries - 1)
        tag = pc >> (2 + (self.num_entries - 1).bit_length())



        entry = self.entries[index]

        if update:
            entry["valid"] = True
            entry["tag"] = tag
            entry["target"] = target
            return None



        if entry["valid"] and entry["tag"] == tag:
            return entry["target"]



        return None


