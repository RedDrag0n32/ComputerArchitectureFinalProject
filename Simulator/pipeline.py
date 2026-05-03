# 5 - stage RISC-V Pipeline Core
#!/usr/bin/env python3
"""
Python conversion of the MU-RISCV simulator C file.
This keeps the original structure: memory helpers, command loop, pipeline stages,
and RISC-V instruction printing.

Run:
    python mu_riscv_sim.py input_program.txt
"""

import sys
from dataclasses import dataclass, field
from typing import List
from Predictors.static import StaticAlwaysTakenPredictor
from Predictors.gshare import GsharePredictor


# -----------------------------
# Constants / opcode definitions
# -----------------------------
TRUE = True
FALSE = False

RISCV_REGS = 32
NUM_MEM_REGION = 3

MEM_TEXT_BEGIN = 0x00400000
MEM_TEXT_END   = 0x004FFFFF
MEM_DATA_BEGIN = 0x10000000
MEM_DATA_END   = 0x1000FFFF
MEM_STACK_BEGIN = 0x7FF00000
MEM_STACK_END   = 0x7FFFFFFF

R_OPCODE       = 0x33
IMM_ALU_OPCODE = 0x13
LOAD_OPCODE    = 0x03
STORE_OPCODE   = 0x23
BRANCH_OPCODE  = 0x63
JUMP_OPCODE    = 0x6F

BIT_MASK_3  = 0x7
BIT_MASK_5  = 0x1F
BIT_MASK_7  = 0x7F
BIT_MASK_12 = 0xFFF
BIT_MASK_20 = 0xFFFFF

BRANCH_PREDICTOR = GsharePredictor(history_bits=10)
#BRANCH_PREDICTOR = StaticAlwaysTakenPredictor()


def u32(value: int) -> int:
    """Force a value into unsigned 32-bit range."""
    return value & 0xFFFFFFFF


def to_i32(value: int) -> int:
    """Interpret unsigned 32-bit value as signed int32."""
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def get_opcode(instruction: int) -> int:
    return instruction & 0x7F


@dataclass
class CPUState:
    PC: int = 0
    REGS: List[int] = field(default_factory=lambda: [0] * RISCV_REGS)
    HI: int = 0
    LO: int = 0

    def copy(self) -> "CPUState":
        return CPUState(
            PC=self.PC,
            REGS=self.REGS.copy(),
            HI=self.HI,
            LO=self.LO,
        )


@dataclass
class PipelineRegister:
    IR: int = 0
    PC: int = 0
    A: int = 0
    B: int = 0
    imm: int = 0
    ALUOutput: int = 0
    LMD: int = 0

    predicted_taken: bool = False
    predicted_pc: int = 0


@dataclass
class MemoryRegion:
    begin: int
    end: int
    mem: bytearray = field(init=False)

    def __post_init__(self):
        self.mem = bytearray(self.end - self.begin + 1)


# -----------------------------
# Global simulator state
# -----------------------------
MEM_REGIONS = [
    MemoryRegion(MEM_TEXT_BEGIN, MEM_TEXT_END),
    MemoryRegion(MEM_DATA_BEGIN, MEM_DATA_END),
    MemoryRegion(MEM_STACK_BEGIN, MEM_STACK_END),
]

CURRENT_STATE = CPUState(PC=MEM_TEXT_BEGIN)
NEXT_STATE = CURRENT_STATE.copy()

IF_ID = PipelineRegister()
ID_EX = PipelineRegister()
EX_MEM = PipelineRegister()
MEM_WB = PipelineRegister()

RUN_FLAG = TRUE
CYCLE_COUNT = 0
INSTRUCTION_COUNT = 0
PROGRAM_SIZE = 0
prog_file = ""

ENABLE_FORWARDING = 0
bubble = False
CONTROL_STALL = 0
CONTROL_FLUSH = 0
BRANCH_TARGET = 0


# -----------------------------
# Utility functions
# -----------------------------
def help_menu() -> None:
    print("------------------------------------------------------------------\n")
    print("\t**********MU-RISCV Help MENU**********\n")
    print("sim\t-- simulate program to completion")
    print("run <n>\t-- simulate program for <n> instructions")
    print("rdump\t-- dump register values")
    print("reset\t-- clears all registers/memory and re-loads the program")
    print("input <reg> <val>\t-- set GPR <reg> to <val>")
    print("mdump <start> <stop>\t-- dump memory from <start> to <stop> address")
    print("high <val>\t-- set the HI register to <val>")
    print("low <val>\t-- set the LO register to <val>")
    print("print\t-- print the program loaded into memory")
    print("show\t-- print the current content of the pipeline registers")
    print("f [0 | 1]\t-- Enable/disable forwarding.")
    print("?\t-- display help menu")
    print("quit\t-- exit the simulator\n")
    print("------------------------------------------------------------------\n")


def sign_extend(value: int, bits: int) -> int:
    mask = 1 << (bits - 1)
    return (value ^ mask) - mask


def mem_read_32(address: int) -> int:
    for region in MEM_REGIONS:
        if region.begin <= address <= region.end:
            offset = address - region.begin
            return u32(
                (region.mem[offset + 3] << 24) |
                (region.mem[offset + 2] << 16) |
                (region.mem[offset + 1] << 8) |
                (region.mem[offset + 0])
            )
    return 0


def mem_write_32(address: int, value: int) -> None:
    value = u32(value)
    for region in MEM_REGIONS:
        if region.begin <= address <= region.end:
            offset = address - region.begin
            region.mem[offset + 3] = (value >> 24) & 0xFF
            region.mem[offset + 2] = (value >> 16) & 0xFF
            region.mem[offset + 1] = (value >> 8) & 0xFF
            region.mem[offset + 0] = value & 0xFF
            return


def writes_to_reg(ir: int) -> int:
    opcode = ir & 0x7F
    return 1 if opcode in (R_OPCODE, IMM_ALU_OPCODE, LOAD_OPCODE) else 0


# -----------------------------
# Simulator control
# -----------------------------
def cycle() -> None:
    global CURRENT_STATE, NEXT_STATE, CYCLE_COUNT
    NEXT_STATE = CURRENT_STATE.copy()
    handle_pipeline()
    CURRENT_STATE = NEXT_STATE.copy()
    CYCLE_COUNT += 1


def run(num_cycles: int) -> None:
    if not RUN_FLAG:
        print("Simulation Stopped\n")
        return

    print(f"Running simulator for {num_cycles} cycles...\n")
    for _ in range(num_cycles):
        if not RUN_FLAG:
            print("Simulation Stopped.\n")
            break
        cycle()


def run_all() -> None:
    if not RUN_FLAG:
        print("Simulation Stopped.\n")
        return

    print("Simulation Started...\n")
    while RUN_FLAG:
        cycle()
    print("Simulation Finished.\n")


def mdump(start: int, stop: int) -> None:
    print("-------------------------------------------------------------")
    print(f"Memory content [0x{start:08x}..0x{stop:08x}] :")
    print("-------------------------------------------------------------")
    print("\t[Address in Hex (Dec) ]\t[Value]")
    address = start
    while address <= stop:
        print(f"\t0x{address:08x} ({address}) :\t0x{mem_read_32(address):08x}")
        address += 4
    print()


def rdump() -> None:
    print("-------------------------------------")
    print("Dumping Register Content")
    print("-------------------------------------")
    print(f"# Instructions Executed\t: {INSTRUCTION_COUNT}")
    print(f"PC\t: 0x{CURRENT_STATE.PC:08x}")
    print("-------------------------------------")
    print("[Register]\t[Value]")
    print("-------------------------------------")
    for i in range(RISCV_REGS):
        print(f"[R{i}]\t: 0x{CURRENT_STATE.REGS[i] & 0xFFFFFFFF:08x}")
    print("-------------------------------------")
    print(f"[HI]\t: 0x{CURRENT_STATE.HI & 0xFFFFFFFF:08x}")
    print(f"[LO]\t: 0x{CURRENT_STATE.LO & 0xFFFFFFFF:08x}")
    print("-------------------------------------")


def reset() -> None:
    global CURRENT_STATE, NEXT_STATE, RUN_FLAG, INSTRUCTION_COUNT
    global IF_ID, ID_EX, EX_MEM, MEM_WB
    global bubble, CONTROL_STALL, CONTROL_FLUSH, BRANCH_TARGET

    CURRENT_STATE = CPUState(PC=MEM_TEXT_BEGIN)
    NEXT_STATE = CURRENT_STATE.copy()

    for region in MEM_REGIONS:
        region.mem[:] = b"\x00" * len(region.mem)

    IF_ID = PipelineRegister()
    ID_EX = PipelineRegister()
    EX_MEM = PipelineRegister()
    MEM_WB = PipelineRegister()

    bubble = False
    CONTROL_STALL = 0
    CONTROL_FLUSH = 0
    BRANCH_TARGET = 0

    load_program()
    INSTRUCTION_COUNT = 0
    CURRENT_STATE.PC = MEM_TEXT_BEGIN
    NEXT_STATE = CURRENT_STATE.copy()
    RUN_FLAG = TRUE


def initialize() -> None:
    global CURRENT_STATE, NEXT_STATE, RUN_FLAG
    CURRENT_STATE.PC = MEM_TEXT_BEGIN
    NEXT_STATE = CURRENT_STATE.copy()
    RUN_FLAG = TRUE


def load_program() -> None:
    global PROGRAM_SIZE
    try:
        with open(prog_file, "r", encoding="utf-8") as fp:
            i = 0
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                word = int(line, 16)
                address = MEM_TEXT_BEGIN + i
                mem_write_32(address, word)
                print(f"writing 0x{word:08x} into address 0x{address:08x} ({address})")
                i += 4
            PROGRAM_SIZE = i // 4
            print(f"Program loaded into memory.\n{PROGRAM_SIZE} words written into memory.\n")
    except FileNotFoundError:
        print(f"Error: Can't open program file {prog_file}")
        sys.exit(-1)


# -----------------------------
# Pipeline
# -----------------------------
def handle_pipeline() -> None:
    global bubble
    WB()
    MEM()
    EX()
    ID()
    # On load/use hazards we bubble decode; do not refetch/advance in the same
    # cycle (this model keeps IF/ID stable while the hazard resolves).
    if not bubble:
        IF()
    bubble = False


def WB() -> None:
    global INSTRUCTION_COUNT
    opcode = MEM_WB.IR & 0x7F
    rd = (MEM_WB.IR >> 7) & 0x1F

    # Nothing in WB this cycle.
    if MEM_WB.IR == 0:
        return

    if opcode in (R_OPCODE, IMM_ALU_OPCODE):
        if rd != 0:
            NEXT_STATE.REGS[rd] = u32(MEM_WB.ALUOutput)
    elif opcode == LOAD_OPCODE:
        if rd != 0:
            NEXT_STATE.REGS[rd] = u32(MEM_WB.LMD)
    elif opcode == STORE_OPCODE:
        pass

    INSTRUCTION_COUNT += 1
    # Don't clear MEM_WB here: ID() runs later in the same cycle and
    # relies on MEM_WB.IR for hazard detection in this simulator ordering.


def MEM() -> None:
    opcode = EX_MEM.IR & 0x7F
    funct3 = (EX_MEM.IR >> 12) & 0x7

    # Nothing in MEM this cycle.
    if EX_MEM.IR == 0:
        MEM_WB.IR = 0
        return

    address = EX_MEM.ALUOutput
    write_data = EX_MEM.B
    read_data = 0

    if opcode == LOAD_OPCODE:
        if funct3 == 0x2:
            read_data = mem_read_32(address)
    elif opcode == STORE_OPCODE:
        if funct3 == 0x2:
            mem_write_32(address, write_data)

    MEM_WB.ALUOutput = EX_MEM.ALUOutput
    MEM_WB.LMD = read_data
    MEM_WB.IR = EX_MEM.IR
    MEM_WB.PC = EX_MEM.PC
    # Don't clear EX_MEM here: ID() runs later in the same cycle and
    # relies on EX_MEM.IR for hazard detection in this simulator ordering.


def EX() -> None:
    global BRANCH_TARGET, CONTROL_FLUSH, CONTROL_STALL

    opcode = ID_EX.IR & 0x7F
    funct3 = (ID_EX.IR >> 12) & 0x7
    funct7 = (ID_EX.IR >> 25) & 0x7F

    # Nothing in EX this cycle.
    if ID_EX.IR == 0:
        EX_MEM.IR = 0
        return

    A = ID_EX.A
    B = ID_EX.B
    imm = ID_EX.imm
    result = 0

    if opcode == R_OPCODE:
        if funct3 == 0x0:
            if funct7 == 0x00:
                result = A + B
            elif funct7 == 0x20:
                result = A - B
        elif funct3 == 0x1:
            result = A << (B & 0x1F)
        elif funct3 == 0x5:
            if funct7 == 0x00:
                result = A >> (B & 0x1F)
            elif funct7 == 0x20:
                result = to_i32(A) >> (B & 0x1F)
        elif funct3 == 0x4:
            result = A ^ B
        elif funct3 == 0x6:
            result = A | B
        elif funct3 == 0x7:
            result = A & B

    elif opcode == IMM_ALU_OPCODE:
        shamt = imm & 0x1F
        if funct3 == 0x0:
            result = A + imm
        elif funct3 == 0x1:
            result = A << shamt
        elif funct3 == 0x5:
            if funct7 == 0x00:
                result = A >> shamt
            elif funct7 == 0x20:
                result = to_i32(A) >> shamt

    elif opcode in (LOAD_OPCODE, STORE_OPCODE):
        result = A + imm

    elif opcode == BRANCH_OPCODE:
        #take_branch = 0
        actual_taken = False
        if funct3 == 0x0:      # BEQ
            #take_branch = A == B
            actual_taken = A == B
        elif funct3 == 0x1:    # BNE
            #take_branch = A != B
            actual_taken = A != B
        elif funct3 == 0x4:    # BLT
            #take_branch = to_i32(A) < to_i32(B)
            actual_taken = to_i32(A) < to_i32(B)
        elif funct3 == 0x5:    # BGE
            #take_branch = to_i32(A) >= to_i32(B)
            actual_taken = to_i32(A) >= to_i32(B)

        actual_target = ID_EX.PC + imm

        if(actual_taken):
            correct_PC = actual_target
        else:
            correct_PC = ID_EX.PC + 4

        BRANCH_PREDICTOR.update(ID_EX.PC, actual_taken, actual_target)

        # if(ID_EX.predicted_pc != correct_PC):
        #     print(f"Branch misprediction at PC = 0x{ID_EX.PC:08x}")
        #     NEXT_STATE.PC = correct_PC

        if ID_EX.predicted_pc != correct_PC:
            print(f"[MISPREDICT] PC=0x{ID_EX.PC:08x} | "
            f"predicted=0x{ID_EX.predicted_pc:08x}, "
            f"actual=0x{correct_PC:08x}")

            IF_ID.IR = 0
            ID_EX.IR = 0

        print(f"[RESOLVE] PC=0x{ID_EX.PC:08x} -> "
        f"{'TAKEN' if actual_taken else 'NOT TAKEN'} "
        f"(correct PC=0x{correct_PC:08x})")

        # if take_branch:
        #     BRANCH_TARGET = ID_EX.PC + imm
        #     CONTROL_FLUSH = 1

        #CONTROL_STALL = 0

    elif opcode == JUMP_OPCODE:
        # BRANCH_TARGET = ID_EX.PC + imm
        # CONTROL_FLUSH = 1
        # CONTROL_STALL = 0
        actual_target = ID_EX.PC + imm
        correct_PC = actual_target

        if(ID_EX.predicted_pc != correct_PC):
            NEXT_STATE.PC = correct_PC
            IF_ID.IR = 0
            ID_EX.IR = 0


    EX_MEM.ALUOutput = u32(result)
    EX_MEM.B = B
    EX_MEM.IR = ID_EX.IR
    EX_MEM.PC = ID_EX.PC
    # Leave ID_EX as-is; it will be overwritten next cycle (or cleared on bubble/flush).


def ID() -> None:
    global bubble, CONTROL_STALL

    if IF_ID.IR == 0x00000000:
        return

    rd = (IF_ID.IR >> 7) & 0x1F
    rs1 = (IF_ID.IR >> 15) & 0x1F
    rs2 = (IF_ID.IR >> 20) & 0x1F
    opcode = IF_ID.IR & 0x7F

    A = CURRENT_STATE.REGS[rs1]
    B = CURRENT_STATE.REGS[rs2]

    hazard_detected = 0
    idex_rd = (ID_EX.IR >> 7) & 0x1F
    exmem_rd = (EX_MEM.IR >> 7) & 0x1F
    memwb_rd = (MEM_WB.IR >> 7) & 0x1F

    # Without forwarding, we must also stall on values still in ID/EX.
    if writes_to_reg(ID_EX.IR) and idex_rd != 0 and (idex_rd == rs1 or idex_rd == rs2):
        hazard_detected = 1

    if writes_to_reg(EX_MEM.IR) and exmem_rd != 0 and (exmem_rd == rs1 or exmem_rd == rs2):
        hazard_detected = 1

    if writes_to_reg(MEM_WB.IR) and memwb_rd != 0 and (memwb_rd == rs1 or memwb_rd == rs2):
        hazard_detected = 1

    if hazard_detected:
        print(f"Hazard detected at PC = 0x{IF_ID.PC:08x}")
        bubble = True
        ID_EX.IR = 0
        ID_EX.A = 0
        ID_EX.B = 0
        ID_EX.imm = 0
        ID_EX.PC = 0
        return

    # if opcode in (BRANCH_OPCODE, JUMP_OPCODE):
    #     if not CONTROL_STALL:
    #         CONTROL_STALL = 1
    #         print(f"Branch or Jump detected at PC = 0x{IF_ID.PC:08x}")

    imm = 0
    if opcode == R_OPCODE:
        imm = 0
    elif opcode in (IMM_ALU_OPCODE, LOAD_OPCODE):
        imm = sign_extend((IF_ID.IR >> 20) & 0xFFF, 12)
    elif opcode == STORE_OPCODE:
        imm_s = (((IF_ID.IR >> 25) & 0x7F) << 5) | ((IF_ID.IR >> 7) & 0x1F)
        imm = sign_extend(imm_s, 12)
    elif opcode == BRANCH_OPCODE:
        imm_b = (
            (((IF_ID.IR >> 31) & 0x1) << 12) |
            (((IF_ID.IR >> 7) & 0x1) << 11) |
            (((IF_ID.IR >> 25) & 0x3F) << 5) |
            (((IF_ID.IR >> 8) & 0xF) << 1)
        )
        imm = sign_extend(imm_b, 13)
    elif opcode == JUMP_OPCODE:
        imm_j = (
            (((IF_ID.IR >> 31) & 0x1) << 20) |
            (((IF_ID.IR >> 12) & 0xFF) << 12) |
            (((IF_ID.IR >> 20) & 0x1) << 11) |
            (((IF_ID.IR >> 21) & 0x3FF) << 1)
        )
        imm = sign_extend(imm_j, 21)

    ID_EX.A = A
    ID_EX.B = B
    ID_EX.imm = imm
    ID_EX.IR = IF_ID.IR
    ID_EX.PC = IF_ID.PC
    ID_EX.predicted_pc = IF_ID.predicted_pc
    ID_EX.predicted_taken = IF_ID.predicted_taken


# def IF() -> None:
#     global CONTROL_FLUSH, CONTROL_STALL

#     if CONTROL_FLUSH:
#         NEXT_STATE.PC = BRANCH_TARGET
#         CONTROL_FLUSH = 0
#         CONTROL_STALL = 0
#         IF_ID.IR = 0x00000000
#         return

#     if CONTROL_STALL:
#         return

#     instruction = mem_read_32(CURRENT_STATE.PC)
#     IF_ID.IR = instruction
#     IF_ID.PC = CURRENT_STATE.PC
#     NEXT_STATE.PC = CURRENT_STATE.PC + 4

def IF() -> None:
    instruction = mem_read_32(CURRENT_STATE.PC)
    opcode = instruction & 0x7F

    IF_ID.IR = instruction
    IF_ID.PC = CURRENT_STATE.PC

    predicted_taken = False
    predicted_pc = CURRENT_STATE.PC + 4

    if opcode == BRANCH_OPCODE:
        predicted_taken = BRANCH_PREDICTOR.predict(CURRENT_STATE.PC)

        imm_b = (
            (((instruction >> 31) & 0x1) << 12) |
            (((instruction >> 7) & 0x1) << 11) |
            (((instruction >> 25) & 0x3F) << 5) |
            (((instruction >> 8) & 0xF) << 1)
        )
        imm = sign_extend(imm_b, 13)

        if predicted_taken:
            predicted_pc = CURRENT_STATE.PC + imm

    elif opcode == JUMP_OPCODE:
        predicted_taken = True

        imm_j = (
            (((instruction >> 31) & 0x1) << 20) |
            (((instruction >> 12) & 0xFF) << 12) |
            (((instruction >> 20) & 0x1) << 11) |
            (((instruction >> 21) & 0x3FF) << 1) 
        )

        imm = sign_extend(imm_j, 21)
        predicted_pc = CURRENT_STATE.PC + imm

    IF_ID.predicted_taken = predicted_taken
    IF_ID.predicted_pc = predicted_pc

    NEXT_STATE.PC = predicted_pc

    if opcode == BRANCH_OPCODE:
        print(f"[PREDICT] PC=0x{CURRENT_STATE.PC:08x} -> "
          f"{'TAKEN' if predicted_taken else 'NOT TAKEN'} "
          f"(next PC=0x{predicted_pc:08x})")
    
    elif opcode == JUMP_OPCODE:
        print(f"[PREDICT] PC=0x{CURRENT_STATE.PC:08x} -> TAKEN (JUMP) "
          f"(next PC=0x{predicted_pc:08x})")


# -----------------------------
# Instruction printing
# -----------------------------
def print_program() -> None:
    mem_tracer = MEM_TEXT_BEGIN
    while mem_tracer < MEM_TEXT_BEGIN + PROGRAM_SIZE * 4:
        cmd = mem_read_32(mem_tracer)
        print_command(cmd)
        print()
        mem_tracer += 4


def print_command(bincmd: int) -> None:
    if bincmd:
        opcode = get_opcode(bincmd)
        if opcode == R_OPCODE:
            handle_r_print(bincmd)
        elif opcode == STORE_OPCODE:
            handle_s_print(bincmd)
        elif opcode in (IMM_ALU_OPCODE, LOAD_OPCODE):
            handle_i_print(bincmd)
        elif opcode == BRANCH_OPCODE:
            handle_b_print(bincmd)
        elif opcode == JUMP_OPCODE:
            handle_j_print(bincmd)
        else:
            print("Unknown command!", end="")


def handle_r_print(bincmd: int) -> None:
    rd = (bincmd >> 7) & BIT_MASK_5
    funct3 = (bincmd >> 12) & BIT_MASK_3
    rs1 = (bincmd >> 15) & BIT_MASK_5
    rs2 = (bincmd >> 20) & BIT_MASK_5
    funct7 = (bincmd >> 25) & BIT_MASK_7

    if funct3 == 0x0:
        if funct7 == 0x0:
            print_r_cmd("add", rd, rs1, rs2)
        elif funct7 == 0x20:
            print_r_cmd("sub", rd, rs1, rs2)
        else:
            print(f"No funct7({funct7}) for funct3({funct3}) found for R-type.", end="")
    elif funct3 == 0x1:
        print_r_cmd("sll", rd, rs1, rs2)
    elif funct3 == 0x2:
        print_r_cmd("slt", rd, rs1, rs2)
    elif funct3 == 0x3:
        print_r_cmd("sltu", rd, rs1, rs2)
    elif funct3 == 0x4:
        print_r_cmd("xor", rd, rs1, rs2)
    elif funct3 == 0x5:
        if funct7 == 0x0:
            print_r_cmd("srl", rd, rs1, rs2)
        elif funct7 == 0x20:
            print_r_cmd("sra", rd, rs1, rs2)
        else:
            print(f"No funct7({funct7}) for funct3({funct3}) found for R-type.", end="")
    elif funct3 == 0x6:
        print_r_cmd("or", rd, rs1, rs2)
    elif funct3 == 0x7:
        print_r_cmd("and", rd, rs1, rs2)
    else:
        print(f"Unknown funct3({funct3}) in R-type", end="")


def handle_s_print(bincmd: int) -> None:
    imm4 = (bincmd >> 7) & BIT_MASK_5
    f3 = (bincmd >> 12) & BIT_MASK_3
    rs1 = (bincmd >> 15) & BIT_MASK_5
    rs2 = (bincmd >> 20) & BIT_MASK_5
    imm11 = (bincmd >> 25) & BIT_MASK_7
    imm = (imm11 << 5) | imm4

    if f3 == 0x0:
        print_s_cmd("sb", rs2, imm, rs1)
    elif f3 == 0x1:
        print_s_cmd("sh", rs2, imm, rs1)
    elif f3 == 0x2:
        print_s_cmd("sw", rs2, imm, rs1)
    else:
        print(f"Unknown funct3({f3}) in S type", end="")


def handle_i_print(bincmd: int) -> None:
    opcode = get_opcode(bincmd)
    rd = (bincmd >> 7) & BIT_MASK_5
    funct3 = (bincmd >> 12) & BIT_MASK_3
    rs1 = (bincmd >> 15) & BIT_MASK_5
    imm = (bincmd >> 20) & BIT_MASK_12

    if opcode == IMM_ALU_OPCODE:
        if funct3 == 0x0:
            print_i_type1_cmd("addi", rd, rs1, imm)
        elif funct3 == 0x1:
            print_i_type1_cmd("slli", rd, rs1, imm)
        elif funct3 == 0x2:
            print_i_type1_cmd("slti", rd, rs1, imm)
        elif funct3 == 0x3:
            print_i_type1_cmd("sltiu", rd, rs1, imm)
        elif funct3 == 0x4:
            print_i_type1_cmd("xori", rd, rs1, imm)
        elif funct3 == 0x5:
            imm5 = imm >> 5
            if imm5 == 0:
                print_i_type1_cmd("srli", rd, rs1, imm)
            elif imm5 == 0x20:
                print_i_type1_cmd("srai", rd, rs1, imm)
            else:
                print(f"Invalid imm[11:5]({imm5}) for I-Type opcode({opcode}) funct3({funct3})", end="")
        elif funct3 == 0x6:
            print_i_type1_cmd("ori", rd, rs1, imm)
        elif funct3 == 0x7:
            print_i_type1_cmd("andi", rd, rs1, imm)
        else:
            print(f"Invalid funct3({funct3}) for I-type opcode({opcode})", end="")

    elif opcode == LOAD_OPCODE:
        if funct3 == 0x0:
            print_i_type2_cmd("lb", rd, rs1, imm)
        elif funct3 == 0x1:
            print_i_type2_cmd("lh", rd, rs1, imm)
        elif funct3 == 0x2:
            print_i_type2_cmd("lw", rd, rs1, imm)
        elif funct3 == 0x4:
            print_i_type2_cmd("lbu", rd, rs1, imm)
        elif funct3 == 0x5:
            print_i_type2_cmd("lhu", rd, rs1, imm)
        else:
            print(f"Unknown funct3({funct3}) for I-type opcode({opcode}).", end="")
    else:
        print(f"Unknown opcode({opcode}) for I-Type.", end="")


def handle_b_print(bincmd: int) -> None:
    imm_b = (
        (((bincmd >> 31) & 0x1) << 12) |
        (((bincmd >> 7) & 0x1) << 11) |
        (((bincmd >> 25) & 0x3F) << 5) |
        (((bincmd >> 8) & 0xF) << 1)
    )
    imm = sign_extend(imm_b, 13)
    f3 = (bincmd >> 12) & BIT_MASK_3
    rs1 = (bincmd >> 15) & BIT_MASK_5
    rs2 = (bincmd >> 20) & BIT_MASK_5

    if f3 == 0x0:
        print_b_cmd("beq", rs1, rs2, imm)
    elif f3 == 0x1:
        print_b_cmd("bne", rs1, rs2, imm)
    elif f3 == 0x4:
        print_b_cmd("blt", rs1, rs2, imm)
    elif f3 == 0x5:
        print_b_cmd("bge", rs1, rs2, imm)
    elif f3 == 0x6:
        print_b_cmd("bltu", rs1, rs2, imm)
    elif f3 == 0x7:
        print_b_cmd("bgeu", rs1, rs2, imm)
    else:
        print(f"Unknown funct3({f3}) in B type", end="")


def handle_j_print(bincmd: int) -> None:
    rd = (bincmd >> 7) & BIT_MASK_5
    imm_j = (
        (((bincmd >> 31) & 0x1) << 20) |
        (((bincmd >> 12) & 0xFF) << 12) |
        (((bincmd >> 20) & 0x1) << 11) |
        (((bincmd >> 21) & 0x3FF) << 1)
    )
    offset = sign_extend(imm_j, 21)
    print(f"jal x{rd}, {offset}", end="")


def print_r_cmd(cmd_name: str, rd: int, rs1: int, rs2: int) -> None:
    print(f"{cmd_name} x{rd}, x{rs1}, x{rs2}", end="")


def print_s_cmd(cmd_name: str, rs2: int, offset: int, rs1: int) -> None:
    print(f"{cmd_name} x{rs2}, {offset}(x{rs1})", end="")


def print_i_type1_cmd(cmd_name: str, rd: int, rs1: int, imm: int) -> None:
    print(f"{cmd_name} x{rd}, x{rs1}, {imm}", end="")


def print_i_type2_cmd(cmd_name: str, rd: int, rs1: int, imm: int) -> None:
    print(f"{cmd_name} x{rd}, {imm}(x{rs1})", end="")


def print_b_cmd(cmd_name: str, rs1: int, rs2: int, imm: int) -> None:
    print(f"{cmd_name} x{rs1}, x{rs2}, {imm}", end="")


def show_pipeline() -> None:
    print("Current pipeline registers:")
    print(f"IF/ID : IR=0x{IF_ID.IR:08x}, PC=0x{IF_ID.PC:08x}")
    print(f"ID/EX : IR=0x{ID_EX.IR:08x}, PC=0x{ID_EX.PC:08x}, A=0x{ID_EX.A & 0xFFFFFFFF:08x}, B=0x{ID_EX.B & 0xFFFFFFFF:08x}, imm={ID_EX.imm}")
    print(f"EX/MEM: IR=0x{EX_MEM.IR:08x}, PC=0x{EX_MEM.PC:08x}, ALUOutput=0x{EX_MEM.ALUOutput & 0xFFFFFFFF:08x}, B=0x{EX_MEM.B & 0xFFFFFFFF:08x}")
    print(f"MEM/WB: IR=0x{MEM_WB.IR:08x}, PC=0x{MEM_WB.PC:08x}, ALUOutput=0x{MEM_WB.ALUOutput & 0xFFFFFFFF:08x}, LMD=0x{MEM_WB.LMD & 0xFFFFFFFF:08x}")


# -----------------------------
# Command loop
# -----------------------------
def handle_command() -> None:
    global ENABLE_FORWARDING, CURRENT_STATE, NEXT_STATE

    try:
        raw = input("MU-RISCV SIM:> ").strip()
    except EOFError:
        sys.exit(0)

    if not raw:
        return

    parts = raw.split()
    command = parts[0].lower()

    try:
        if command in ("sim", "s"):
            run_all()
        elif command in ("show", "sh"):
            show_pipeline()
        elif command in ("mdump", "m"):
            start = int(parts[1], 16)
            stop = int(parts[2], 16)
            mdump(start, stop)
        elif command == "?":
            help_menu()
        elif command in ("quit", "q"):
            print("**************************")
            print("Exiting MU-RISCV! Good Bye...")
            print("**************************")
            sys.exit(0)
        elif command in ("rdump", "rd"):
            rdump()
        elif command in ("reset", "re"):
            reset()
        elif command in ("run", "r"):
            cycles = int(parts[1])
            run(cycles)
        elif command in ("input", "i"):
            register_no = int(parts[1])
            register_value = int(parts[2], 0)
            CURRENT_STATE.REGS[register_no] = u32(register_value)
            NEXT_STATE.REGS[register_no] = u32(register_value)
        elif command in ("high", "h"):
            value = int(parts[1], 0)
            CURRENT_STATE.HI = u32(value)
            NEXT_STATE.HI = u32(value)
        elif command in ("low", "l"):
            value = int(parts[1], 0)
            CURRENT_STATE.LO = u32(value)
            NEXT_STATE.LO = u32(value)
        elif command in ("print", "p"):
            print_program()
        elif command in ("f", "forwarding"):
            ENABLE_FORWARDING = int(parts[1])
            print("Forwarding OFF" if ENABLE_FORWARDING == 0 else "Forwarding ON")
        else:
            print("Invalid Command.")
    except (IndexError, ValueError):
        print("Invalid command format.")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    global prog_file

    print(f"MEM_TEXT_BEGIN = 0x{MEM_TEXT_BEGIN:08x}")
    print("\n**************************")
    print("Welcome to MU-RISCV SIM...")
    print("**************************\n")

    if len(sys.argv) < 2:
        print(f"Error: You should provide input file.\nUsage: {sys.argv[0]} <input program> \n")
        return 1

    prog_file = sys.argv[1]
    initialize()
    reset()
    help_menu()

    while True:
        handle_command()


if __name__ == "__main__":
    raise SystemExit(main())
