from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from isa import (
    ALLOWED_OPERAND_TYPES,
    CODE_MEM_WORDS,
    DATA_MEM_WORDS,
    INSTR_BYTES,
    JUMP_OPCODES,
    STACK_WORDS,
    WORD_BYTES,
    WORD_MASK,
    Instruction,
    Opcode,
    OperandType,
    decode_instruction,
    decode_word,
    instruction_to_string,
    sign_extend32,
)

INTERRUPT_VECTOR = 1  # адрес в памяти команд, с которого начинается обработка прерывания
PORT_IN = 0
PORT_OUT = 1

STACK_BOTTOM = DATA_MEM_WORDS - STACK_WORDS  # ниже этого адреса стек расти не может


class MachineError(Exception):
    pass  # ошибка времени выполнения модели

class PsSel(Enum):
    INC = auto()  # PS + 1
    OPERAND = auto()  # операнд команды (адрес перехода)
    VECTOR = auto()  # вектор прерывания
    DATA = auto()  # данные, прочитанные из памяти (адрес возврата)


class SpSel(Enum):
    DEC = auto()
    INC = auto()


class DaSel(Enum):
    OPERAND = auto()  # операнд -> DA
    SP_OFFSET = auto()  # SP + операнд -> DA
    SP = auto()  # SP -> DA
    DATA = auto()  # прочитанные данные (косвенная адресация) -> DA


class BSel(Enum):
    OPERAND = auto()  # непосредственный операнд
    DATA = auto()  # данные из кеша
    PORT = auto()  # данные из порта ввода


class AluOp(Enum):
    PASS_B = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    REM = auto()


class WSel(Enum):
    ACC = auto()
    PS = auto()
    FLAGS = auto()


class FlagsSel(Enum):
    ALU = auto()  # флаги по результату ALU
    DATA = auto()  # флаги восстанавливаются из памяти (iret)


@dataclass(frozen=True)
class Signals:
    """Набор управляющих сигналов одного такта"""

    latch_ps: bool = False
    ps_sel: PsSel = PsSel.INC
    latch_dr: bool = False
    latch_sp: bool = False
    sp_sel: SpSel = SpSel.DEC
    latch_da: bool = False
    da_sel: DaSel = DaSel.OPERAND
    latch_acc: bool = False
    b_sel: BSel = BSel.OPERAND
    alu_op: AluOp = AluOp.PASS_B
    latch_flags: bool = False
    flags_sel: FlagsSel = FlagsSel.ALU
    mem_rd: bool = False
    mem_wr: bool = False
    w_sel: WSel = WSel.ACC
    io_wr: bool = False
    set_ie: bool = False
    clr_ie: bool = False
    set_in_irq: bool = False
    clr_in_irq: bool = False
    ack_irq: bool = False
    halt: bool = False


@dataclass(frozen=True)
class Flags:
    z: bool = False
    n: bool = False
    p: bool = False

    @classmethod
    def of(cls, result: int) -> Flags:
        return cls(z=result == 0, n=result < 0, p=result > 0)

    @classmethod
    def from_word(cls, word: int) -> Flags:
        return cls(z=bool(word & 1), n=bool(word & 2), p=bool(word & 4))

    def to_word(self) -> int:
        return int(self.z) | (int(self.n) << 1) | (int(self.p) << 2)

    def __str__(self) -> str:
        return f"Z{int(self.z)}N{int(self.n)}P{int(self.p)}"


class InputDevice:
    """Устройство ввода (в заданные такты в port появляется символ и выставляется irq)"""

    def __init__(self, schedule: list[tuple[int, str]]) -> None:
        self.schedule = sorted(schedule, key=lambda item: item[0])
        self.next_index = 0
        self.data = 0
        self.irq = False

    def deliver(self, tick: int) -> list[str]:
        notes: list[str] = []
        while self.next_index < len(self.schedule) and self.schedule[self.next_index][0] <= tick:
            pair = self.schedule[self.next_index]
            char = pair[1]
            self.next_index += 1
            if self.irq:
                notes.append(f"INPUT OVERRUN: символ {self.data} потерян")
            self.data = ord(char) # кладём новый символ в порт
            self.irq = True
            notes.append(f"INPUT {char!r} -> порт {PORT_IN}, irq=1")
        return notes

    def read(self, port: int) -> int:
        if port != PORT_IN:
            raise MachineError(f"чтение из порта {port}: порт не является устройством ввода")
        return self.data

    def ack(self) -> None:
        self.irq = False


class OutputDevice:
    """Устройство вывода: принимает байты в порт, превращает в символы, копит их"""
    def __init__(self) -> None:
        self.chars: list[str] = []

    def write(self, port: int, value: int) -> str:
        if port != PORT_OUT:
            raise MachineError(f"запись в порт {port}: порт не является устройством вывода")
        char = chr(value & 0xFF)
        self.chars.append(char)
        return char

