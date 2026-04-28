def always_taken_predictor(opcode, pc, target_address):
    BRANCH_OPS = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}

    if opcode in BRANCH_OPS:
        predict_taken = True
        predicted_pc = target_address
    else:
        predict_taken = False
        predicted_pc = pc + 4

    return predict_taken, predicted_pc