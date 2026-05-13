
class StatsCollector:

    def __init__(self):

        self.TOTAL_BRANCHES = 0
        self.CORRECT_PREDICTIONS = 0
        self.MISPREDICTIONS = 0
        self.PENALTY_CYCLES = 0

        self.TOTAL_JUMPS = 0
        self.JUMP_MISPREDICTIONS = 0

        # self.CYCLE_COUNT = 0
        # self.INSTRUCTION_COUNT = 0

        self.BTB_HITS = 0
        self.BTB_MISSES = 0

    def record_branch(self, predicted_pc, correct_pc):
        self.TOTAL_BRANCHES += 1

        if predicted_pc == correct_pc:
            self.CORRECT_PREDICTIONS += 1
        else:
            self.MISPREDICTIONS += 1
            self.PENALTY_CYCLES += 2

    def record_jump(self, predicted_pc, correct_pc):
        self.TOTAL_JUMPS += 1

        if predicted_pc != correct_pc:
            self.JUMP_MISPREDICTIONS += 1
            self.PENALTY_CYCLES += 2

    def record_btb_access(self, hit):
        if hit:
            self.BTB_HITS += 1
        else:
            self.BTB_MISSES += 1

    def prediction_accuracy(self):
        if self.TOTAL_BRANCHES == 0:
            return 0
        return self.CORRECT_PREDICTIONS / self.TOTAL_BRANCHES
    

    def misprediction_rate(self):
        if self.TOTAL_BRANCHES == 0:
            return 0
        return self.MISPREDICTIONS / self.TOTAL_BRANCHES
    
    def btb_hit_rate(self):
        total = self.BTB_HITS + self.BTB_MISSES
        if total == 0:
            return 0
        return self.BTB_HITS / total
    
    def cpi(self, cycle_count, instruction_count):
        if instruction_count == 0:
            return 0
        return cycle_count / instruction_count
    
    def actual_cpi(self, cycle_count, instruction_count):
        if instruction_count == 0:
            return 0
        return (cycle_count + self.PENALTY_CYCLES) / instruction_count
    
    def print_summary(self, predictor, btb, cycle_count, instruction_count):
        print("--------- Simulator Statistics ---------")
        print(f"Total Branches: {self.TOTAL_BRANCHES}")
        print(f"Correct Predictions: {self.CORRECT_PREDICTIONS}")
        print(f"Mispredictions: {self.MISPREDICTIONS}")
        print(f"Accuracy: {self.prediction_accuracy() * 100:.2f}%")
        print(f"Misprediction Rate: {self.misprediction_rate()}")
        print(f"BTB Hits: {self.BTB_HITS}")
        print(f"BTB Misses: {self.BTB_MISSES}")
        print(f"BTB Hit Rate: {self.btb_hit_rate() * 100:.2f}%")
        print(f"CPI: {self.cpi(cycle_count, instruction_count):.4f}")
        print(f"Actual CPI: {self.actual_cpi(cycle_count, instruction_count):.4f}")
        print(f"Storage Bits: {predictor.storage_bits() + btb.storage_bits()}")
        

    



