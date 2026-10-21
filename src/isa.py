from __future__ import annotations
from enum import IntEnum

WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
INSTR_BITS = 48
INSTR_MASK = (1 << INSTR_BITS) - 1
SIGN_BIT = 1 << (WORD_BITS - 1)

class Opcode(IntEnum):
    LD = 0x01
    ST = 0x02

    ADD = 0x03
    SUB = 0x04
    MUL = 0x05
    DIV = 0x06
    REM = 0x07

    JMP = 0x08
    JE = 0x09   # ZF = 1
    JG = 0x0A   # PF = 1
    JGE = 0x0B  # ZF = 1 or PF = 1
    JL = 0x0C   # NF = 1
    JLE = 0x0D  # ZF = 1 or NF = 1
    JNE = 0x0E  # ZF = 0

    PUSH = 0x0F
    POP = 0x10
    CALL = 0x11
    RET = 0x12
    HLT = 0x13

    IN = 0x20
    OUT = 0x21

    EI = 0x30
    DI = 0x31
    IRET = 0x32

class OperandType(IntEnum):
    NONE = 0
    IMMEDIATE = 1
    ADDRESS = 2
    STACK_OFFSET = 3
    STACK_POINTER = 4
    PORT = 5

# Допустимые типы операнда для каждой инструкции
ALLOWED_OPERAND_TYPES: dict[Opcode, frozenset[OperandType]] = {
    Opcode.LD:   frozenset({OperandType.IMMEDIATE, OperandType.ADDRESS,
                          OperandType.STACK_OFFSET, OperandType.STACK_POINTER}),
    Opcode.ST:   frozenset({OperandType.ADDRESS, OperandType.STACK_OFFSET,
                          OperandType.STACK_POINTER}),
    Opcode.ADD:  frozenset({OperandType.IMMEDIATE, OperandType.ADDRESS,
                           OperandType.STACK_OFFSET, OperandType.STACK_POINTER}),
    Opcode.SUB:  frozenset({OperandType.IMMEDIATE, OperandType.ADDRESS,
                           OperandType.STACK_OFFSET, OperandType.STACK_POINTER}),
    Opcode.MUL:  frozenset({OperandType.IMMEDIATE, OperandType.ADDRESS,
                           OperandType.STACK_OFFSET, OperandType.STACK_POINTER}),
    Opcode.DIV:  frozenset({OperandType.IMMEDIATE, OperandType.ADDRESS,
                           OperandType.STACK_OFFSET, OperandType.STACK_POINTER}),
    Opcode.REM:  frozenset({OperandType.IMMEDIATE, OperandType.ADDRESS,
                           OperandType.STACK_OFFSET, OperandType.STACK_POINTER}),
    Opcode.JMP:  frozenset({OperandType.ADDRESS}),
    Opcode.JE:   frozenset({OperandType.ADDRESS}),
    Opcode.JG:   frozenset({OperandType.ADDRESS}),
    Opcode.JGE:  frozenset({OperandType.ADDRESS}),
    Opcode.JL:   frozenset({OperandType.ADDRESS}),
    Opcode.JLE:  frozenset({OperandType.ADDRESS}),
    Opcode.JNE:  frozenset({OperandType.ADDRESS}),
    Opcode.PUSH: frozenset({OperandType.NONE}),
    Opcode.POP:  frozenset({OperandType.NONE}),
    Opcode.CALL: frozenset({OperandType.ADDRESS}),
    Opcode.RET:  frozenset({OperandType.NONE}),
    Opcode.HLT:  frozenset({OperandType.NONE}),
    Opcode.IN:   frozenset({OperandType.PORT}),
    Opcode.OUT:  frozenset({OperandType.PORT}),
    Opcode.EI:   frozenset({OperandType.NONE}),
    Opcode.DI:   frozenset({OperandType.NONE}),
    Opcode.IRET: frozenset({OperandType.NONE}),
}

missing = set(Opcode) - set(ALLOWED_OPERAND_TYPES)
if missing:
    raise RuntimeError(
        f"ALLOWED_OPERAND_TYPES не покрывает опкоды: {sorted(o.name for o in missing)}"
    )

def sign_extend32(raw: int) -> int:
    if raw & SIGN_BIT:
        return raw - (1 << WORD_BITS)
    else:
        return raw

def to_unsigned32(val: int) -> int:
    return val & WORD_MASK

class Instruction:
    """Логическое представление инструкции до и после кодирования"""
    opcode: Opcode
    operand_type: OperandType
    operand: int
    line: int | None

    def __init__(
            self,
            opcode: Opcode,
            operand_type: OperandType,
            operand: int = 0,
            line: int | None = None,
    )-> None:
        self.opcode = opcode
        self.operand_type = operand_type
        self.operand = operand
        self.line = line


    def encode(self) -> int:
        return (
            ((int(self.opcode) & 0xFF) << 40)
            | ((int(self.operand_type) & 0xFF) << 32)
            | (to_unsigned32(self.operand) & WORD_MASK)
        )

    @classmethod
    def decode(cls, raw: int) -> "Instruction":
        opcode = Opcode((raw >> 40) & 0xFF)
        operand_type = OperandType((raw >> 32) & 0xFF)
        operand = sign_extend32(raw & WORD_MASK)
        return cls(opcode=opcode, operand_type=operand_type, operand=operand)

    def __repr__(self) -> str:
        if self.operand_type == OperandType.NONE:
            return self.opcode.name
        return f"{self.opcode.name} {self.operand_type.name}:{self.operand}"


