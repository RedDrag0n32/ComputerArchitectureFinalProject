
class BTBEntry:
    def __init__(self):
        self.valid = False
        self.tag = 0
        self.target = 0


class BTB:
    def __init__(self, num_entries: int = 256):
        self.size = num_entries
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


