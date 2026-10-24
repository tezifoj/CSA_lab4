from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
SIGN_BIT = 1 << (WORD_BITS - 1)
INT32_MIN = -(1 << (WORD_BITS - 1))
INT32_MAX = (1 << (WORD_BITS - 1)) - 1

CODE_MEM_WORDS = 1 << 16  # 65536 инструкций
DATA_MEM_WORDS = 1 << 16  # 65536 слов данных
STACK_WORDS = 1 << 12  # 4096 слов стека (растёт вниз)


class Opcode(IntEnum):
    LD = 0x01
    ST = 0x02

    ADD = 0x03
    SUB = 0x04
    MUL = 0x05
    DIV = 0x06
    REM = 0x07

    JMP = 0x08
    JE = 0x09  # ZF = 1
    JG = 0x0A  # PF = 1
    JGE = 0x0B  # ZF = 1 or PF = 1
    JL = 0x0C  # NF = 1
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


_ARITH_OPERANDS = frozenset(
    {
        OperandType.IMMEDIATE,
        OperandType.ADDRESS,
        OperandType.STACK_OFFSET,
        OperandType.STACK_POINTER,
    }
)
_ADDR_ONLY = frozenset({OperandType.ADDRESS})
_NO_OPERAND = frozenset({OperandType.NONE})
_PORT_ONLY = frozenset({OperandType.PORT})

# Допустимые типы операнда для каждой инструкции
ALLOWED_OPERAND_TYPES: dict[Opcode, frozenset[OperandType]] = {
    Opcode.LD: _ARITH_OPERANDS,
    Opcode.ST: _ARITH_OPERANDS - {OperandType.IMMEDIATE},
    Opcode.ADD: _ARITH_OPERANDS,
    Opcode.SUB: _ARITH_OPERANDS,
    Opcode.MUL: _ARITH_OPERANDS,
    Opcode.DIV: _ARITH_OPERANDS,
    Opcode.REM: _ARITH_OPERANDS,
    Opcode.JMP: _ADDR_ONLY,
    Opcode.JE: _ADDR_ONLY,
    Opcode.JG: _ADDR_ONLY,
    Opcode.JGE: _ADDR_ONLY,
    Opcode.JL: _ADDR_ONLY,
    Opcode.JLE: _ADDR_ONLY,
    Opcode.JNE: _ADDR_ONLY,
    Opcode.PUSH: _NO_OPERAND,
    Opcode.POP: _NO_OPERAND,
    Opcode.CALL: _ADDR_ONLY,
    Opcode.RET: _NO_OPERAND,
    Opcode.HLT: _NO_OPERAND,
    Opcode.IN: _PORT_ONLY,
    Opcode.OUT: _PORT_ONLY,
    Opcode.EI: _NO_OPERAND,
    Opcode.DI: _NO_OPERAND,
    Opcode.IRET: _NO_OPERAND,
}

JUMP_OPCODES: frozenset[Opcode] = frozenset(
    {
        Opcode.JMP,
        Opcode.JE,
        Opcode.JG,
        Opcode.JGE,
        Opcode.JL,
        Opcode.JLE,
        Opcode.JNE,
        Opcode.CALL,
    }
)


def sign_extend32(raw: int) -> int:
    return raw - (1 << WORD_BITS) if raw & SIGN_BIT else raw


def to_unsigned32(val: int) -> int:
    return val & WORD_MASK


@dataclass
class Instruction:
    opcode: Opcode
    operand_type: OperandType
    operand: int = 0
    line: int | None = None


def instruction_to_string(instr: Instruction) -> str:
    name = instr.opcode.name
    if instr.operand_type == OperandType.NONE:
        return name
    if instr.operand_type in (OperandType.IMMEDIATE, OperandType.PORT):
        return f"{name} {instr.operand}"
    if instr.operand_type == OperandType.ADDRESS:
        if instr.opcode in JUMP_OPCODES:
            return f"{name} {instr.operand}"
        return f"{name} [{instr.operand}]"
    if instr.operand_type == OperandType.STACK_OFFSET:
        return f"{name} [SP{instr.operand:+d}]"
    if instr.operand_type == OperandType.STACK_POINTER:
        return f"{name} [[SP{instr.operand:+d}]]"
    raise ValueError(f"неизвестный тип операнда: {instr.operand_type!r}")

# Формат в памяти: opcode (1 байт), тип операнда (1 байт), операнд (4 байта), little-endian
_INSTR_STRUCT = struct.Struct("<BBI")
_WORD_STRUCT = struct.Struct("<I")

INSTR_BYTES = _INSTR_STRUCT.size
WORD_BYTES = _WORD_STRUCT.size


def encode_instruction(instr: Instruction) -> bytes:
    return _INSTR_STRUCT.pack(
        int(instr.opcode),
        int(instr.operand_type),
        to_unsigned32(instr.operand),
    )


def decode_instruction(data: bytes) -> Instruction:
    opcode, op_type, raw_operand = _INSTR_STRUCT.unpack(data)
    return Instruction(
        opcode=Opcode(opcode),
        operand_type=OperandType(op_type),
        operand=sign_extend32(raw_operand),
    )


def encode_word(value: int) -> bytes:
    """Упаковать 32-битное слово данных"""
    return _WORD_STRUCT.pack(to_unsigned32(value))


def decode_word(data: bytes) -> int:
    """Распаковать 32-битное знаковое слово данных"""
    (raw,) = _WORD_STRUCT.unpack(data)
    return sign_extend32(raw)