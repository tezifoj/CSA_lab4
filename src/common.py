from __future__ import annotations
import struct

from isa import (
    INSTR_BITS,
    INSTR_MASK,
    SIGN_BIT,
    WORD_BITS,
    WORD_MASK,
    Instruction,
    OperandType,
    Opcode,
    sign_extend32,
    to_unsigned32,
)

CODE_MEM_WORDS = 1 << 16     # 65536 инструкций
DATA_MEM_WORDS = 1 << 16     # 65536 слов данных
STACK_WORDS = 1 << 12        # 4096 слов стека (растёт вниз)

INT32_MIN = -(1 << (WORD_BITS - 1))
INT32_MAX = (1 << (WORD_BITS - 1)) - 1



JUMP_OPCODES: frozenset[Opcode] = frozenset(
    {
        Opcode.JMP, Opcode.JE, Opcode.JG, Opcode.JGE,
        Opcode.JL, Opcode.JLE, Opcode.JNE, Opcode.CALL,
    }
)

def is_jump_instruction_opcode(op: Opcode) -> bool:
    return op in JUMP_OPCODES

missing_jumps = JUMP_OPCODES - set(Opcode)
if missing_jumps:
    raise RuntimeError(
        f"JUMP_OPCODES содержит неизвестные коды операции: "
        f"{sorted(o.name for o in missing_jumps)}"
    )

def instruction_to_string(instr: Instruction) -> str:
    """Приводим к ассерблерному виду для тестов"""
    if instr.operand_type == OperandType.NONE:
        return instr.opcode.name
    if instr.operand_type == OperandType.IMMEDIATE:
        return f"{instr.opcode.name} {instr.operand}"
    if instr.operand_type == OperandType.ADDRESS:
        if is_jump_instruction_opcode(instr.opcode):
            return f"{instr.opcode.name} {instr.operand}"
        return f"{instr.opcode.name} [{instr.operand}]"
    if instr.operand_type == OperandType.STACK_OFFSET:
        return f"{instr.opcode.name} [SP{instr.operand:+d}]"
    if instr.operand_type == OperandType.STACK_POINTER:
        return f"{instr.opcode.name} [[SP{instr.operand:+d}]]"
    if instr.operand_type == OperandType.PORT:
        return f"{instr.opcode.name} {instr.operand}"
    return instr.opcode.name

def instruction_to_string(instr: Instruction) -> str:
    if instr.operand_type == OperandType.NONE:
        return instr.opcode.name
    return f"{instr.opcode.name} {instr.operand_type.name}:{instr.operand}"

_INSTR_STRUCT = struct.Struct("<BBI")  # 1 + 1 + 4 = 6 byte
_WORD_STRUCT = struct.Struct("<I")     # 4 byte

INSTR_BYTES = _INSTR_STRUCT.size
WORD_BYTES = _WORD_STRUCT.size

assert INSTR_BYTES * 8 >= INSTR_BITS, "инструкция не помещается в отведённые байты"
assert WORD_BYTES * 8 >= WORD_BITS, "слово не помешается в отведённые байты"

def encode_instruction(instr: Instruction) -> bytes:
    return _INSTR_STRUCT.pack(
        int(instr.opcode) & 0xFF,
        int(instr.operand_type) & 0xFF,
        to_unsigned32(instr.operand),
    )

def decode_instruction(data: bytes) -> Instruction:
    if len(data) != INSTR_BYTES:
        raise ValueError(
            f"decode_instruction: ожидалось {INSTR_BYTES} байт, получено {len(data)}"
        )
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
    if len(data) != WORD_BYTES:
        raise ValueError(
            f"decode_word: ожидалось {WORD_BYTES} байт, получено {len(data)}"
        )
    (raw,) = _WORD_STRUCT.unpack(data)
    return sign_extend32(raw)




