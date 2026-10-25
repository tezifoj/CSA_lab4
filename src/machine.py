from __future__ import annotations

import struct
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


class DataPath:

    def __init__(
            self,
            code: bytes,
            data: bytes,
            schedule: list[tuple[int, str]],

    ) -> None:
        if len(code) % INSTR_BYTES:
            raise MachineError("размер файла кода не кратен размеру команды")
        if len(data) % WORD_BYTES:
            raise MachineError("размер файла данных не кратен размеру слова")
        # память команд
        fmt = f"{INSTR_BYTES}s"
        self.imem = [item[0] for item in struct.iter_unpack(fmt, code)]
        if len(self.imem) > CODE_MEM_WORDS:
            raise MachineError("программа не помещается в память команд")
        # память данных
        fmt = f">{WORD_BYTES}s"
        words = [decode_word(item[0]) for item in struct.iter_unpack(fmt, data)]
        if len(words) > STACK_BOTTOM:
            raise MachineError("данные не помещаются в память данных")
        self.dmem = words + [0] * (DATA_MEM_WORDS - len(words))

        self.input_dev = InputDevice(schedule)
        self.output_dev = OutputDevice()

        self.ps = 0
        self.ac = 0
        self.sp = DATA_MEM_WORDS  # указатель стека
        self.da = 0  # регистр адреса данных
        self.dr: Instruction | None = None  # регистр команды
        self.flags = Flags()
        self.ie = False  # разрешение прерываний
        self.in_irq = False  # выполняется обработчик прерывания
        self.halted = False

    # состояния, которые видит ControlUnit
    @property
    def pend(self) -> bool:
        """Запрос прерывания от устройства ввода"""
        return self.input_dev.irq

    @property
    def operand(self) -> int:
        if self.dr is None:
            raise MachineError("в регистре команды нет команды")
        return self.dr.operand

    # устройства

    def deliver_input(self, tick: int) -> list[str]:
        return self.input_dev.deliver(tick)

    # мультиплексоры

    def mux_ps(self, sel: PsSel) -> int:
        if sel == PsSel.INC:
            return self.ps + 1
        if sel == PsSel.OPERAND:
            return self.operand
        if sel == PsSel.VECTOR:
            return INTERRUPT_VECTOR
        return self.cache.rdata

    def mux_da(self, sel: DaSel) -> int:
        if sel == DaSel.OPERAND:
            return self.operand
        if sel == DaSel.SP_OFFSET:
            return self.sp + self.operand
        if sel == DaSel.SP:
            return self.sp
        return self.cache.rdata

    def mux_b(self, sel: BSel, notes: list[str]) -> int:
        if sel == BSel.OPERAND:
            return self.operand
        if sel == BSel.DATA:
            return self.cache.rdata
        value = self.input_dev.read(self.operand)
        notes.append(f"IN port {self.operand} -> {value}")
        return value

    def mux_w(self, sel: WSel) -> int:
        if sel == WSel.ACC:
            return self.ac
        if sel == WSel.PS:
            return self.ps
        return self.flags.to_word()

    # функциональные блоки

    def alu(self, op: AluOp, b: int) -> int:
        """ALU: левый вход AC, правый вход B; результат 32 бита со знаком"""
        a = self.ac
        if op == AluOp.PASS_B:
            result = b
        elif op == AluOp.ADD:
            result = a + b
        elif op == AluOp.SUB:
            result = a - b
        elif op == AluOp.MUL:
            result = a * b
        else:
            if b == 0:
                raise MachineError("деление на ноль")
            quotient = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                quotient = -quotient
            result = quotient if op == AluOp.DIV else a - b * quotient
        return sign_extend32(result & WORD_MASK)

    def sp_adder(self, sel: SpSel) -> int:
        new_sp = self.sp - 1 if sel == SpSel.DEC else self.sp + 1
        if new_sp < STACK_BOTTOM:
            raise MachineError("переполнение стека")
        if new_sp > DATA_MEM_WORDS:
            raise MachineError("выход за основание стека")
        return new_sp

    def fetch_instruction(self) -> Instruction:
        """Чтение памяти команд по PS и декодирование слова в регистр команды"""
        if not 0 <= self.ps < len(self.imem):
            raise MachineError(f"PS={self.ps} вне программы")
        try:
            return decode_instruction(self.imem[self.ps])
        except ValueError as exc:
            raise MachineError(f"неверная команда по адресу {self.ps}: {exc}") from exc

    def access_memory(self, sig: Signals, notes: list[str]) -> bool:
        """Такт обращения к памяти данных через кеш; вернуть ready."""
        if not self.cache.busy:
            if sig.mem_rd:
                self.cache.start_read(self.da)
            else:
                self.cache.start_write(self.da, self.mux_w(sig.w_sel))
        ready = self.cache.tick()
        notes.append(self.cache.describe())
        return ready
