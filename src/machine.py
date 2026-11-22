from __future__ import annotations

import argparse
import ast
import contextlib
import struct
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TextIO

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

HIT_TICKS = 1  # обращение к кешу
MEM_TICKS = 10  # обращение к памяти данных
CACHE_LINES = 4
DEFAULT_TICK_LIMIT = 1_000_000


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
            self.data = ord(char)  # кладём новый символ в порт
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


class DataCache:
    def __init__(self, memory: list[int], lines: int = CACHE_LINES) -> None:
        self.memory = memory
        self.lines = lines
        self.tags: list[int | None] = [None] * lines
        self.words: list[int] = [0] * lines
        self.rdata = 0  # выходной регистр кеша
        self.busy = False
        self.is_write = False
        self.addr = 0
        self.value = 0
        self.hit = False
        self.remaining = 0
        self.total = 0
        self.read_hits = 0
        self.read_misses = 0
        self.writes = 0

    def _locate(self, addr: int) -> tuple[int, int, bool]:
        if not 0 <= addr < DATA_MEM_WORDS:
            raise MachineError(f"адрес данных вне памяти: {addr}")
        if self.lines == 0:
            return 0, addr, False
        index, tag = addr % self.lines, addr // self.lines
        return index, tag, self.tags[index] == tag

    def start_read(self, addr: int) -> None:
        _, _, self.hit = self._locate(addr)
        self._start(addr, is_write=False, value=0, ticks=HIT_TICKS if self.hit else MEM_TICKS)

    def start_write(self, addr: int, value: int) -> None:
        _, _, self.hit = self._locate(addr)
        self._start(addr, is_write=True, value=value, ticks=MEM_TICKS)

    def _start(self, addr: int, *, is_write: bool, value: int, ticks: int) -> None:
        self.busy = True
        self.is_write = is_write
        self.addr = addr
        self.value = value
        self.remaining = ticks
        self.total = ticks

    def tick(self) -> bool:
        """Один такт работы, return ready"""
        self.remaining -= 1
        if self.remaining > 0:
            return False
        index, tag, _ = self._locate(self.addr)
        if self.is_write:
            self._finish_write(index)
        else:
            self._finish_read(index, tag)
        self.busy = False
        return True

    def _finish_write(self, index: int) -> None:
        self.writes += 1
        self.memory[self.addr] = self.value
        if self.hit:
            self.words[index] = self.value

    def _finish_read(self, index: int, tag: int) -> None:
        if self.hit:
            self.read_hits += 1
            self.rdata = self.words[index]
            return
        self.read_misses += 1
        self.rdata = self.memory[self.addr]
        if self.lines > 0:  # заполнение строки
            self.tags[index] = tag
            self.words[index] = self.rdata

    def describe(self) -> str:
        done = self.total - self.remaining
        kind = "WR" if self.is_write else "RD"
        status = "HIT" if self.hit else "MISS"
        wait = "" if self.remaining == 0 else " wait"
        return f"CACHE {kind} [{self.addr}] {status} {done}/{self.total}{wait}"


class DataPath:
    def __init__(
        self,
        code: bytes,
        data: bytes,
        schedule: list[tuple[int, str]],
        cache_lines: int = CACHE_LINES,
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
        self.cache = DataCache(self.dmem, cache_lines)

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

    # такт

    def execute(self, sig: Signals, notes: list[str]) -> bool:
        """Выполнить такт по сигналам, вернуть ready"""
        if sig.mem_rd or sig.mem_wr:
            return self.access_memory(sig, notes)

        new_ps, new_dr, new_sp, new_da = self.ps, self.dr, self.sp, self.da
        new_ac, new_flags = self.ac, self.flags

        if sig.latch_dr:
            new_dr = self.fetch_instruction()
        if sig.latch_ps:
            new_ps = self.mux_ps(sig.ps_sel)
        if sig.latch_sp:
            new_sp = self.sp_adder(sig.sp_sel)
        if sig.latch_da:
            new_da = self.mux_da(sig.da_sel)
        if sig.latch_acc or (sig.latch_flags and sig.flags_sel == FlagsSel.ALU):
            result = self.alu(sig.alu_op, self.mux_b(sig.b_sel, notes))
            if sig.latch_acc:
                new_ac = result
            if sig.latch_flags:
                new_flags = Flags.of(result)
        if sig.latch_flags and sig.flags_sel == FlagsSel.DATA:
            new_flags = Flags.from_word(self.cache.rdata)
        if sig.io_wr:
            char = self.output_dev.write(self.operand, self.ac)
            notes.append(f"OUT port {self.operand} <- {char!r}")

        self.ps, self.dr, self.sp, self.da = new_ps, new_dr, new_sp, new_da
        self.ac, self.flags = new_ac, new_flags
        self._apply_control_flags(sig)
        return True

    def _apply_control_flags(self, sig: Signals) -> None:
        if sig.set_ie:
            self.ie = True
        if sig.clr_ie:
            self.ie = False
        if sig.set_in_irq:
            self.in_irq = True
        if sig.clr_in_irq:
            self.in_irq = False
        if sig.ack_irq:
            self.input_dev.ack()
        if sig.halt:
            self.halted = True


# ControlUnit

FETCH = Signals(latch_dr=True, latch_ps=True, ps_sel=PsSel.INC)
_RD = Signals(mem_rd=True)
_DA_FROM_DATA = Signals(latch_da=True, da_sel=DaSel.DATA)
_DA_FROM_SP = Signals(latch_da=True, da_sel=DaSel.SP)
_SP_DEC = Signals(latch_sp=True, sp_sel=SpSel.DEC)
_SP_INC = Signals(latch_sp=True, sp_sel=SpSel.INC)

_ALU_OF: dict[Opcode, AluOp] = {
    Opcode.LD: AluOp.PASS_B,
    Opcode.ADD: AluOp.ADD,
    Opcode.SUB: AluOp.SUB,
    Opcode.MUL: AluOp.MUL,
    Opcode.DIV: AluOp.DIV,
    Opcode.REM: AluOp.REM,
}
_COND_JUMPS = JUMP_OPCODES - {Opcode.CALL}

_INTERRUPT_ENTRY: list[Signals] = [
    _SP_DEC,
    _DA_FROM_SP,
    Signals(mem_wr=True, w_sel=WSel.PS),
    _SP_DEC,
    _DA_FROM_SP,
    Signals(mem_wr=True, w_sel=WSel.FLAGS),
    Signals(latch_ps=True, ps_sel=PsSel.VECTOR, set_in_irq=True, ack_irq=True),
]


class ControlUnit:
    """Блок управления: decoder, step counter, branch logic, interrupt condition, signal logic"""

    def __init__(self, datapath: DataPath) -> None:
        self.datapath = datapath
        self.step = 0  # счётчик тактов внутри текущей инструкции
        self.interrupt_entry = False
        self.instructions = 0
        self.interrupts = 0

    def instruction_decoder(self) -> Instruction:
        """Код операции и способ адресации выполняемой команды"""
        instr = self.datapath.dr
        if instr is None:
            raise MachineError("в регистре команды нет команды")
        if instr.operand_type not in ALLOWED_OPERAND_TYPES[instr.opcode]:
            raise MachineError(f"недопустимый тип операнда {instr.operand_type.name} для {instr.opcode.name}")
        return instr

    def branch_logic(self, op: Opcode) -> bool:
        """Условия перехода по флагам N, Z, P"""
        flags = self.datapath.flags
        if op == Opcode.JMP:
            return True
        if op == Opcode.JE:
            return flags.z
        if op == Opcode.JG:
            return flags.p
        if op == Opcode.JGE:
            return flags.z or flags.p
        if op == Opcode.JL:
            return flags.n
        if op == Opcode.JLE:
            return flags.z or flags.n
        if op == Opcode.JNE:
            return not flags.z
        raise MachineError(f"не условный переход: {op.name}")

    def interrupt_condition(self) -> bool:
        """PEND & IE & ¬IN_IRQ & step0"""
        return self.step == 0 and self.datapath.pend and self.datapath.ie and not self.datapath.in_irq

    @staticmethod
    def _da_from_operand(operand_type: OperandType) -> Signals:
        sel = DaSel.OPERAND if operand_type == OperandType.ADDRESS else DaSel.SP_OFFSET
        return Signals(latch_da=True, da_sel=sel)

    def _load_like_signals(self, instr: Instruction) -> list[Signals]:
        """ld, add, sub, mul, div, rem"""
        op, ot = instr.opcode, instr.operand_type
        immediate = ot == OperandType.IMMEDIATE
        execute = Signals(
            latch_acc=True,
            b_sel=BSel.OPERAND if ot == OperandType.IMMEDIATE else BSel.DATA,
            alu_op=_ALU_OF[op],
            latch_flags=op != Opcode.LD,
        )
        if immediate:
            return [execute]
        if ot == OperandType.STACK_POINTER:
            return [self._da_from_operand(ot), _RD, _DA_FROM_DATA, _RD, execute]
        return [self._da_from_operand(ot), _RD, execute]

    def _store_signals(self, instr: Instruction) -> list[Signals]:
        ot = instr.operand_type
        write = Signals(mem_wr=True, w_sel=WSel.ACC)
        if ot == OperandType.STACK_POINTER:
            return [self._da_from_operand(ot), _RD, _DA_FROM_DATA, write]
        return [self._da_from_operand(ot), write]

    def _execution_signals(self, instr: Instruction) -> list[Signals]:
        op = instr.opcode
        if op == Opcode.LD or op in _ALU_OF:
            return self._load_like_signals(instr)
        if op == Opcode.ST:
            return self._store_signals(instr)
        if op in _COND_JUMPS:
            if self.branch_logic(op):
                return [Signals(latch_ps=True, ps_sel=PsSel.OPERAND)]
            return [Signals()]
        if op == Opcode.PUSH:
            return [_SP_DEC, _DA_FROM_SP, Signals(mem_wr=True, w_sel=WSel.ACC)]
        if op == Opcode.POP:
            return [_SP_INC]
        if op == Opcode.CALL:
            return [
                _SP_DEC,
                _DA_FROM_SP,
                Signals(mem_wr=True, w_sel=WSel.PS),
                Signals(latch_ps=True, ps_sel=PsSel.OPERAND),
            ]
        if op == Opcode.RET:
            return [
                _DA_FROM_SP,
                _RD,
                Signals(latch_ps=True, ps_sel=PsSel.DATA, latch_sp=True, sp_sel=SpSel.INC),
            ]
        if op == Opcode.IN:
            return [Signals(latch_acc=True, b_sel=BSel.PORT, alu_op=AluOp.PASS_B)]
        if op == Opcode.OUT:
            return [Signals(io_wr=True)]
        if op == Opcode.EI:
            return [Signals(set_ie=True)]
        if op == Opcode.DI:
            return [Signals(clr_ie=True)]
        if op == Opcode.IRET:
            return [
                _DA_FROM_SP,
                _RD,
                Signals(
                    latch_flags=True,
                    flags_sel=FlagsSel.DATA,
                    latch_sp=True,
                    sp_sel=SpSel.INC,
                ),
                _DA_FROM_SP,
                _RD,
                Signals(
                    latch_ps=True,
                    ps_sel=PsSel.DATA,
                    latch_sp=True,
                    sp_sel=SpSel.INC,
                    clr_in_irq=True,
                ),
            ]
        if op == Opcode.HLT:
            return [Signals(halt=True)]
        raise MachineError(f"неизвестная команда: {op.name}")

    def signal_logic(self) -> tuple[Signals, bool]:
        """Выбрать Signals текущего такта и признак последнего такта"""
        if self.interrupt_entry:
            return _INTERRUPT_ENTRY[self.step - 1], self.step == len(_INTERRUPT_ENTRY)
        if self.step == 0:
            return FETCH, False
        sequence = self._execution_signals(self.instruction_decoder())
        return sequence[self.step - 1], self.step == len(sequence)

    def step_counter(self, *, ready: bool, last: bool) -> None:
        """Номер такта растёт, пока тракт данных готов; после последнего такта сбрасывается"""
        if not ready:
            return
        if last:
            self.step = 0
            self.interrupt_entry = False
        else:
            self.step += 1

    def tick(self, notes: list[str]) -> tuple[int, bool]:
        if self.interrupt_condition():
            self.interrupt_entry = True
            self.step = 1
            self.interrupts += 1
            notes.append("INTERRUPT: вход в обработчик")

        executed_step = self.step
        was_interrupt = self.interrupt_entry
        signals, last = self.signal_logic()
        ready = self.datapath.execute(signals, notes)
        if executed_step == 0 and not was_interrupt:
            self.instructions += 1
        self.step_counter(ready=ready, last=last)
        return executed_step, was_interrupt

    def label(self, step: int, *, was_interrupt: bool) -> str:
        """Подпись выполняемой команды для лога"""
        if was_interrupt:
            return "INTERRUPT ENTRY"
        instr = self.datapath.dr
        mnemonic = instruction_to_string(instr) if instr is not None else "-"
        return f"FETCH -> {mnemonic}" if step == 0 else mnemonic


@dataclass
class Result:
    output: str
    ticks: int
    instructions: int
    interrupts: int
    halted: bool
    reason: str
    read_hits: int
    read_misses: int


class Machine:
    """Процессор: ControlUnit + DataPath, счётчик тактов и логи"""

    def __init__(
        self,
        code: bytes,
        data: bytes,
        schedule: list[tuple[int, str]],
        *,
        log: TextIO | None = None,
        cache_lines: int = CACHE_LINES,
    ) -> None:
        self.datapath = DataPath(code, data, schedule, cache_lines)
        self.control_unit = ControlUnit(self.datapath)
        self.log = log
        self.ticks = 0

    def tick(self) -> None:
        self.ticks += 1
        notes = self.datapath.deliver_input(self.ticks)
        step, was_interrupt = self.control_unit.tick(notes)
        label = self.control_unit.label(step, was_interrupt=was_interrupt)
        self._write_log(label, step, notes)

    def _write_log(self, label: str, step: int, notes: list[str]) -> None:
        if self.log is None:
            return
        dp = self.datapath
        mode = "INT " if dp.in_irq else "MAIN"
        line = (
            f"{self.ticks:7d} | {mode} | {label:<26} s{step} | "
            f"PS={dp.ps:<5d} AC={dp.ac:<11d} SP={dp.sp:<5d} DA={dp.da:<5d} | "
            f"{dp.flags} IE{int(dp.ie)} PEND{int(dp.pend)}"
        )
        if notes:
            line += " | " + "; ".join(notes)
        self.log.write(line + "\n")

    def run(self, limit: int = DEFAULT_TICK_LIMIT) -> Result:
        reason = "halt"
        try:
            while not self.datapath.halted:
                if self.ticks >= limit:
                    reason = f"лимит тактов ({limit})"
                    break
                self.tick()
        except MachineError as exc:
            reason = f"ошибка: {exc}"
            if self.log is not None:
                self.log.write(f"ERROR at tick {self.ticks}: {exc}\n")
        cache = self.datapath.cache
        return Result(
            output="".join(self.datapath.output_dev.chars),
            ticks=self.ticks,
            instructions=self.control_unit.instructions,
            interrupts=self.control_unit.interrupts,
            halted=self.datapath.halted,
            reason=reason,
            read_hits=cache.read_hits,
            read_misses=cache.read_misses,
        )


def parse_schedule(text: str) -> list[tuple[int, str]]:
    """Преобразование текста в список"""
    stripped = text.strip()
    if not stripped:
        return []
    try:
        raw = ast.literal_eval(stripped)
    except (ValueError, SyntaxError) as exc:
        raise MachineError(f"неверный формат ввода: {exc}") from exc
    schedule: list[tuple[int, str]] = []
    for item in raw:
        tick, char = item
        if not isinstance(tick, int) or not isinstance(char, str) or len(char) != 1:
            raise MachineError(f"неверный элемент: {item!r}")
        schedule.append((tick, char))
    return schedule


def run_files(  # noqa: PLR0913 - три файла и три настройки модели, все именованные
    code_path: str,
    data_path: str,
    input_path: str,
    *,
    log: TextIO | None,
    limit: int,
    cache_lines: int,
) -> Result:
    code = Path(code_path).read_bytes()
    data = Path(data_path).read_bytes()
    schedule = parse_schedule(Path(input_path).read_text(encoding="utf-8"))
    machine = Machine(code, data, schedule, log=log, cache_lines=cache_lines)
    return machine.run(limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Модель процессора (acc, harv, trap, port, cache)")
    parser.add_argument("code", help="бинарный файл команд")
    parser.add_argument("data", help="бинарный файл данных")
    parser.add_argument("input", help="файл с расписанием ввода")
    parser.add_argument("--log", help="файл журнала работы процессора")
    parser.add_argument("--limit", type=int, default=DEFAULT_TICK_LIMIT, help="максимум тактов")
    parser.add_argument("--cache-lines", type=int, default=CACHE_LINES, help="число строк кеша (0 - без кеша)")
    args = parser.parse_args(argv)

    try:
        with contextlib.ExitStack() as stack:
            log_file = stack.enter_context(Path(args.log).open("w", encoding="utf-8")) if args.log else None
            result = run_files(
                args.code,
                args.data,
                args.input,
                log=log_file,
                limit=args.limit,
                cache_lines=args.cache_lines,
            )
    except (MachineError, OSError) as exc:
        sys.stderr.write(f"ошибка: {exc}\n")
        return 1

    sys.stdout.write(result.output + "\n")
    sys.stderr.write(
        f"тактов: {result.ticks}, команд: {result.instructions}, прерываний: {result.interrupts}, "
        f"кеш: попаданий {result.read_hits}, промахов {result.read_misses}, "
        f"причина остановки: {result.reason}\n"
    )
    return 0 if result.halted else 1


if __name__ == "__main__":
    sys.exit(main())
