from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isa import (
    ALLOWED_OPERAND_TYPES,
    DATA_MEM_WORDS,
    INT32_MAX,
    INT32_MIN,
    STACK_WORDS,
    Instruction,
    Opcode,
    OperandType,
    encode_instruction,
    encode_word,
    instruction_to_string,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

DEFAULT_STDLIB = Path(__file__).with_name("stdlib.lsp")
STDLIB_INPUT_TRIGGER = "readchar"
STATIC_LIMIT = DATA_MEM_WORDS - STACK_WORDS
PORT_LIMIT = 255


class TranslatorError(Exception):
    """Ошибка трансляции"""


OPERATORS = frozenset({"+", "-", "*", "/", "%", "=", "!=", "<", ">", "<=", ">="})
ARITH_OPS: dict[str, Opcode] = {
    "+": Opcode.ADD,
    "-": Opcode.SUB,
    "*": Opcode.MUL,
    "/": Opcode.DIV,
    "%": Opcode.REM,
}
COMPARE_TRUE: dict[str, Opcode] = {
    "=": Opcode.JE,
    "!=": Opcode.JNE,
    "<": Opcode.JL,
    ">": Opcode.JG,
    "<=": Opcode.JLE,
    ">=": Opcode.JGE,
}
COMPARE_FALSE: dict[str, Opcode] = {
    "=": Opcode.JNE,
    "!=": Opcode.JE,
    "<": Opcode.JGE,
    ">": Opcode.JLE,
    "<=": Opcode.JG,
    ">=": Opcode.JL,
}
RESERVED = frozenset(
    {"define", "defun", "defint", "if", "progn", "let", "setq", "loop", "for", "from", "to", "do", "not"}
)
BUILTIN_ARITY: dict[str, int] = {
    "port-in": 1,  # (port-in 0)
    "port-out": 2,  # (port-out 1 65)
    "getchar": 2,  # (getchar buffer index)
    "setchar": 3,  # (setchar buffer index c)
    "makestring": 1,  # (makestring length)
    "ei": 0,
    "di": 0,
    "halt": 0,
}
CHAR_NAMES = {"newline": 10, "space": 32, "tab": 9, "nul": 0}
ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
NUMBER_RE = re.compile(r"-?\d+")
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
DELIMITERS = set(' \t\r\n()";')


@dataclass(frozen=True)
class Token:
    kind: str  # "(", ")", "num", "char", "str", "sym"
    value: str | int
    line: int


def tokenize(text: str, origin: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, n = 0, 1, len(text)

    def fail(message: str) -> TranslatorError:
        return TranslatorError(f"{origin}:{line}: {message}")

    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
        elif c in " \t\r":
            i += 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c in "()":
            tokens.append(Token(c, c, line))
            i += 1
        elif c == '"':
            i, value = _read_string(text, i, line, origin)
            tokens.append(Token("str", value, line))
        elif text.startswith("#\\", i):
            i, code = _read_char(text, i, line, origin)
            tokens.append(Token("char", code, line))
        else:
            j = i
            while j < n and text[j] not in DELIMITERS:
                j += 1
            word = text[i:j]
            i = j
            if NUMBER_RE.fullmatch(word):
                tokens.append(Token("num", int(word), line))
            elif word in OPERATORS or NAME_RE.fullmatch(word):
                tokens.append(Token("sym", word, line))
            else:
                raise fail(f"недопустимое слово {word!r}")
    return tokens


def _read_string(text: str, start: int, line: int, origin: str) -> tuple[int, str]:
    i = start + 1
    chars: list[str] = []
    while i < len(text):
        c = text[i]
        if c == '"':
            return i + 1, "".join(chars)
        if c == "\\":
            i += 1
            if i >= len(text) or text[i] not in ESCAPES:
                raise TranslatorError(f"{origin}:{line}: неизвестная escape-последовательность")
            chars.append(ESCAPES[text[i]])
        else:
            chars.append(c)
        i += 1
    raise TranslatorError(f"{origin}:{line}: незакрытая строка")


def _read_char(text: str, start: int, line: int, origin: str) -> tuple[int, int]:
    i = start + 2
    if i >= len(text):
        raise TranslatorError(f"{origin}:{line}: пустой литерал символа")
    j = i + 1
    if text[i].isalpha():
        while j < len(text) and text[j].isalnum():
            j += 1
    word = text[i:j]
    if len(word) == 1:
        code = ord(word)
        if code > 0xFF:
            raise TranslatorError(f"{origin}:{line}: символ {word!r} не помещается в байт")
        return j, code
    if word in CHAR_NAMES:
        return j, CHAR_NAMES[word]
    raise TranslatorError(f"{origin}:{line}: неизвестный литерал символа #\\{word}")

# Чтение S-выражений

@dataclass(frozen=True)
class SAtom:
    token: Token


@dataclass(frozen=True)
class SList:
    items: tuple[SExpr, ...]
    line: int


SExpr = SAtom | SList


def read_sexprs(tokens: list[Token], origin: str) -> list[SExpr]:
    result: list[SExpr] = []
    pos = 0

    def read() -> SExpr:
        nonlocal pos
        token = tokens[pos]
        pos += 1
        if token.kind == ")":
            raise TranslatorError(f"{origin}:{token.line}: лишняя закрывающая скобка")
        if token.kind != "(":
            return SAtom(token)
        items: list[SExpr] = []
        while True:
            if pos >= len(tokens):
                raise TranslatorError(f"{origin}:{token.line}: не закрыта скобка")
            if tokens[pos].kind == ")":
                pos += 1
                return SList(tuple(items), token.line)
            items.append(read())

    while pos < len(tokens):
        result.append(read())
    return result

# AST
@dataclass(frozen=True)
class Num:
    value: int
    line: int


@dataclass(frozen=True)
class Str:
    value: str
    line: int


@dataclass(frozen=True)
class Var:
    name: str
    line: int


@dataclass(frozen=True)
class BinOp:
    op: str
    left: Expr
    right: Expr
    line: int


@dataclass(frozen=True)
class Not:
    arg: Expr
    line: int


@dataclass(frozen=True)
class If:
    cond: Expr
    then: Expr
    otherwise: Expr
    line: int


@dataclass(frozen=True)
class Progn:
    body: tuple[Expr, ...]
    line: int


@dataclass(frozen=True)
class Let:
    bindings: tuple[tuple[str, Expr], ...]
    body: tuple[Expr, ...]
    line: int


@dataclass(frozen=True)
class Setq:
    name: str
    value: Expr
    line: int


@dataclass(frozen=True)
class Loop:
    var: str
    start: Expr
    end: Expr
    body: tuple[Expr, ...]
    line: int


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Expr, ...]
    line: int


Expr = Num | Str | Var | BinOp | Not | If | Progn | Let | Setq | Loop | Call


@dataclass(frozen=True)
class Define:
    name: str
    value: Expr
    line: int


@dataclass(frozen=True)
class Defun:
    name: str
    params: tuple[str, ...]
    body: tuple[Expr, ...]
    line: int


@dataclass(frozen=True)
class Defint:
    name: str
    body: tuple[Expr, ...]
    line: int


Toplevel = Define | Defun | Defint | Expr


class Parser:
    """Строит AST из s-выражений, проверка синтаксиса форм"""

    def __init__(self, origin: str) -> None:
        self.origin = origin

    def fail(self, line: int, message: str) -> TranslatorError:
        return TranslatorError(f"{self.origin}:{line}: {message}")

    def parse_program(self, text: str) -> list[Toplevel]:
        sexprs = read_sexprs(tokenize(text, self.origin), self.origin)
        return [self.parse_toplevel(s) for s in sexprs]

    def name(self, s: SExpr, what: str) -> str:
        if isinstance(s, SAtom) and s.token.kind == "sym" and NAME_RE.fullmatch(str(s.token.value)):
            value = str(s.token.value)
            if value in RESERVED:
                raise self.fail(s.token.line, f"{value!r} зарезервировано и не может быть именем {what}")
            return value
        line = s.token.line if isinstance(s, SAtom) else s.line
        raise self.fail(line, f"ожидалось имя ({what})")

    def parse_toplevel(self, s: SExpr) -> Toplevel:
        if isinstance(s, SList) and s.items and isinstance(s.items[0], SAtom):
            head = s.items[0].token.value
            if head == "define":
                if len(s.items) != 3:
                    raise self.fail(s.line, "define: ожидалось (define имя выражение)")
                return Define(self.name(s.items[1], "переменной"), self.parse_expr(s.items[2]), s.line)
            if head == "defun":
                return self.parse_defun(s)
            if head == "defint":
                return self.parse_defint(s)
        return self.parse_expr(s)

    def parse_defun(self, s: SList) -> Defun:
        if len(s.items) < 4 or not isinstance(s.items[2], SList):
            raise self.fail(s.line, "defun: ожидалось (defun имя (параметры) выражение+)")
        params = tuple(self.name(p, "параметра") for p in s.items[2].items)
        if len(set(params)) != len(params):
            raise self.fail(s.line, "defun: повторяющиеся имена параметров")
        name = self.name(s.items[1], "функции")
        if name in BUILTIN_ARITY:
            raise self.fail(s.line, f"defun: {name!r} это встроенная форма")
        return Defun(name, params, tuple(self.parse_expr(e) for e in s.items[3:]), s.line)

    def parse_defint(self, s: SList) -> Defint:
        if len(s.items) < 4 or not isinstance(s.items[2], SList) or s.items[2].items:
            raise self.fail(s.line, "defint: ожидалось (defint имя () выражение+)")
        return Defint(
            self.name(s.items[1], "обработчика"),
            tuple(self.parse_expr(e) for e in s.items[3:]),
            s.line,
        )

    def parse_expr(self, s: SExpr) -> Expr:
        if isinstance(s, SAtom):
            token = s.token
            if token.kind in ("num", "char"):
                value = int(token.value)
                if not INT32_MIN <= value <= INT32_MAX:
                    raise self.fail(token.line, f"число {value} не помещается в 32 бита")
                return Num(value, token.line)
            if token.kind == "str":
                return Str(str(token.value), token.line)
            return Var(self.name(s, "переменной"), token.line)
        if not s.items:
            raise self.fail(s.line, "пустой список ()")
        head = s.items[0]
        if not isinstance(head, SAtom) or head.token.kind != "sym":
            raise self.fail(s.line, "первым элементом формы должно быть имя или операция")
        word = str(head.token.value)
        args = s.items[1:]
        if word in OPERATORS:
            if len(args) != 2:
                raise self.fail(s.line, f"({word} a b): операция принимает ровно два аргумента")
            return BinOp(word, self.parse_expr(args[0]), self.parse_expr(args[1]), s.line)
        builders: dict[str, Callable[[SList, tuple[SExpr, ...]], Expr]] = {
            "not": self.parse_not,
            "if": self.parse_if,
            "progn": self.parse_progn,
            "let": self.parse_let,
            "setq": self.parse_setq,
            "loop": self.parse_loop,
        }
        builder = builders.get(word)
        if builder is not None:
            return builder(s, args)
        if word in ("define", "defun", "defint"):
            raise self.fail(s.line, f"{word} допустим только на верхнем уровне")
        if word in RESERVED:
            raise self.fail(s.line, f"неожиданное ключевое слово {word!r}")
        if not NAME_RE.fullmatch(word):
            raise self.fail(s.line, f"{word!r} не является именем функции")
        return Call(word, tuple(self.parse_expr(a) for a in args), s.line)

    def parse_not(self, s: SList, args: tuple[SExpr, ...]) -> Not:
        if len(args) != 1:
            raise self.fail(s.line, "(not a): ожидался один аргумент")
        return Not(self.parse_expr(args[0]), s.line)

    def parse_if(self, s: SList, args: tuple[SExpr, ...]) -> If:
        if len(args) != 3:
            raise self.fail(s.line, "(if условие то иначе): ожидалось три аргумента")
        return If(self.parse_expr(args[0]), self.parse_expr(args[1]), self.parse_expr(args[2]), s.line)

    def parse_progn(self, s: SList, args: tuple[SExpr, ...]) -> Progn:
        if not args:
            raise self.fail(s.line, "(progn выражение+): нужно хотя бы одно выражение")
        return Progn(tuple(self.parse_expr(a) for a in args), s.line)

    def parse_let(self, s: SList, args: tuple[SExpr, ...]) -> Let:
        if len(args) < 2 or not isinstance(args[0], SList) or not args[0].items:
            raise self.fail(s.line, "(let ((имя значение)+) выражение+)")
        bindings: list[tuple[str, Expr]] = []
        for b in args[0].items:
            if not isinstance(b, SList) or len(b.items) != 2:
                raise self.fail(s.line, "let: привязка должна иметь вид (имя значение)")
            bindings.append((self.name(b.items[0], "переменной"), self.parse_expr(b.items[1])))
        return Let(tuple(bindings), tuple(self.parse_expr(a) for a in args[1:]), s.line)

    def parse_setq(self, s: SList, args: tuple[SExpr, ...]) -> Setq:
        if len(args) != 2:
            raise self.fail(s.line, "(setq имя выражение)")
        return Setq(self.name(args[0], "переменной"), self.parse_expr(args[1]), s.line)

    def parse_loop(self, s: SList, args: tuple[SExpr, ...]) -> Loop:
        def keyword(index: int, expected: str) -> None:
            item = args[index] if index < len(args) else None
            if not (isinstance(item, SAtom) and item.token.value == expected):
                raise self.fail(s.line, f"loop: ожидалось ключевое слово {expected!r}")

        if len(args) < 8:
            raise self.fail(s.line, "(loop for i from a to b do выражение+)")
        keyword(0, "for")
        keyword(2, "from")
        keyword(4, "to")
        keyword(6, "do")
        return Loop(
            self.name(args[1], "счётчика цикла"),
            self.parse_expr(args[3]),
            self.parse_expr(args[5]),
            tuple(self.parse_expr(a) for a in args[7:]),
            s.line,
        )


# вывод AST
def _describe(node: object) -> tuple[str, Sequence[tuple[str, object]]]:
    if isinstance(node, Num):
        return f"Num {node.value}", []
    if isinstance(node, Str):
        return f"Str {node.value!r}", []
    if isinstance(node, Var):
        return f"Var {node.name}", []
    if isinstance(node, BinOp):
        return f"BinOp {node.op}", [("left", node.left), ("right", node.right)]
    if isinstance(node, Not):
        return "Not", [("arg", node.arg)]
    if isinstance(node, If):
        return "If", [("cond", node.cond), ("then", node.then), ("else", node.otherwise)]
    if isinstance(node, Progn):
        return "Progn", [("", e) for e in node.body]
    if isinstance(node, Let):
        return "Let", [(f"bind {n}", e) for n, e in node.bindings] + [("body", e) for e in node.body]
    if isinstance(node, Setq):
        return f"Setq {node.name}", [("value", node.value)]
    if isinstance(node, Loop):
        return f"Loop {node.var}", [("from", node.start), ("to", node.end)] + [("do", e) for e in node.body]
    if isinstance(node, Call):
        return f"Call {node.name}", [("arg", a) for a in node.args]
    if isinstance(node, Define):
        return f"Define {node.name}", [("value", node.value)]
    if isinstance(node, Defun):
        return f"Defun {node.name}({' '.join(node.params)})", [("body", e) for e in node.body]
    if isinstance(node, Defint):
        return f"Defint {node.name}", [("body", e) for e in node.body]
    raise TypeError(node)


def ast_to_lines(node: object, indent: int = 0, label: str = "") -> list[str]:
    head, children = _describe(node)
    prefix = "  " * indent + (f"{label}: " if label else "")
    lines = [prefix + head]
    for child_label, child in children:
        lines += ast_to_lines(child, indent + 1, child_label)
    return lines


def ast_to_string(items: list[Toplevel]) -> str:
    lines = ["Program"]
    for item in items:
        lines += ast_to_lines(item, 1)
    return "\n".join(lines) + "\n"


def iter_nodes(node: object) -> Iterator[object]:
    """Все узлы поддерева в прямом порядке."""
    yield node
    for _, child in _describe(node)[1]:
        yield from iter_nodes(child)

# Размещение данных


@dataclass
class Layout:
    strings: dict[str, int]
    globals: dict[str, int]
    buffers: dict[int, int]  # id(makestring) -> адрес
    words: list[int]


def build_layout(items: list[tuple[Toplevel, str]]) -> Layout:
    """Строки с адреса 0, затем глобальные переменные, затем буферы makestring"""
    strings: dict[str, int] = {}
    words: list[int] = []
    for item, _ in items:
        for node in iter_nodes(item):
            if isinstance(node, Str) and node.value not in strings:
                strings[node.value] = len(words)
                words += [*node.value.encode("utf-8"), 0]
    globals_: dict[str, int] = {}
    for item, origin in items:
        if isinstance(item, Define):
            if item.name in globals_:
                raise TranslatorError(f"{origin}:{item.line}: повторное определение переменной {item.name!r}")
            globals_[item.name] = len(words)
            words.append(0)
    buffers: dict[int, int] = {}
    for item, origin in items:
        for node in iter_nodes(item):
            if isinstance(node, Call) and node.name == "makestring":
                size = node.args[0] if len(node.args) == 1 else None
                if not isinstance(size, Num) or size.value <= 0:
                    raise TranslatorError(f"{origin}:{node.line}: makestring: размер должен быть положительным числом")
                buffers[id(node)] = len(words)
                words += [0] * size.value
    if len(words) > STATIC_LIMIT:
        raise TranslatorError("статические данные не помещаются в память данных")
    return Layout(strings, globals_, buffers, words)


# Генерация команд


def base_line(e: Expr) -> int:
    return e.line


@dataclass
class Asm:
    opcode: Opcode
    operand_type: OperandType
    operand: int | str
    origin: str
    line: int


@dataclass
class Ctx:
    name: str | None
    depth: int = 0  # сколько слов на стеке


Env = tuple[tuple[str, int, bool], ...]


class CodeGen:
    def __init__(self, functions: dict[str, Defun], layout: Layout) -> None:
        self.functions = functions
        self.layout = layout
        self.items: list[Asm | str] = []
        self.ctx = Ctx(None)
        self.origin = "user"
        self.line = 0
        self.labels_count = 0

    def error(self, message: str) -> TranslatorError:
        return TranslatorError(f"{self.origin}:{self.line}: {message}")

    def new_label(self, hint: str) -> str:
        self.labels_count += 1
        return f".{hint}{self.labels_count}"

    def label(self, name: str) -> None:
        self.items.append(name)

    def emit(self, opcode: Opcode, operand_type: OperandType = OperandType.NONE, operand: int | str = 0) -> None:
        if operand_type not in ALLOWED_OPERAND_TYPES[opcode]:
            raise self.error(f"внутренняя ошибка: {opcode.name} не допускает {operand_type.name}")
        self.items.append(Asm(opcode, operand_type, operand, self.origin, self.line))

    def push(self) -> None:
        self.emit(Opcode.PUSH)
        self.ctx.depth += 1

    def pop(self) -> None:
        self.emit(Opcode.POP)
        self.ctx.depth -= 1

    def lookup(self, name: str, env: Env) -> tuple[OperandType, int, bool]:
        for var, slot, readonly in reversed(env):
            if var == name:
                return OperandType.STACK_OFFSET, self.ctx.depth + slot, readonly
        if name in self.layout.globals:
            return OperandType.ADDRESS, self.layout.globals[name], False
        raise self.error(f"необъявленное имя {name!r}")

    def simple_operand(self, e: Expr, env: Env) -> tuple[OperandType, int] | None:
        """Операнд (можно использовать прямо в команде без вычисления)"""
        if isinstance(e, Num):
            return OperandType.IMMEDIATE, e.value
        if isinstance(e, Str):
            return OperandType.IMMEDIATE, self.layout.strings[e.value]
        if isinstance(e, Var):
            self.line = e.line
            operand_type, operand, _ = self.lookup(e.name, env)
            return operand_type, operand
        return None

    def gen_function(self, f: Defun, origin: str) -> None:
        self.origin, self.line = origin, f.line
        self.ctx = Ctx(f.name)
        count = len(f.params)
        env: Env = tuple((p, count - i, False) for i, p in enumerate(f.params))
        self.label(f"fn_{f.name}")
        self.gen_body(f.body, env, tail=True)
        self.emit(Opcode.RET)

    def gen_interrupt(self, h: Defint, origin: str) -> None:
        self.origin, self.line = origin, h.line
        self.ctx = Ctx(None)
        self.label("int_handler")
        self.push()  # сохранить AC, PS и флаги уже сохранены аппаратно
        self.gen_body(h.body, (), tail=False)
        self.emit(Opcode.LD, OperandType.STACK_OFFSET, self.ctx.depth - 1)
        self.pop()
        self.emit(Opcode.IRET)

    def gen_toplevel(self, items: list[tuple[Toplevel, str]]) -> None:
        self.ctx = Ctx(None)
        self.label("start")
        for item, origin in items:
            self.origin = origin
            self.line = item.line
            if isinstance(item, Define):
                self.gen_expr(item.value, ())
                self.emit(Opcode.ST, OperandType.ADDRESS, self.layout.globals[item.name])
            elif isinstance(item, (Defun, Defint)):
                continue
            else:
                self.gen_expr(item, ())
        self.emit(Opcode.HLT)

    def gen_body(self, body: tuple[Expr, ...], env: Env, *, tail: bool) -> None:
        for index, e in enumerate(body):
            self.gen_expr(e, env, tail=tail and index == len(body) - 1)

    # выражения

    def gen_expr(self, e: Expr, env: Env, *, tail: bool = False) -> None:
        self.line = e.line
        if isinstance(e, (Num, Str, Var)):
            operand = self.simple_operand(e, env)
            if operand is None:
                raise self.error("внутренняя ошибка: операнд не вычислен")
            self.emit(Opcode.LD, *operand)
        elif isinstance(e, BinOp):
            self.gen_binop(e, env)
        elif isinstance(e, Not):
            self.gen_not(e, env)
        elif isinstance(e, If):
            self.gen_if(e, env, tail=tail)
        elif isinstance(e, Progn):
            self.gen_body(e.body, env, tail=tail)
        elif isinstance(e, Let):
            self.gen_let(e, env, tail=tail)
        elif isinstance(e, Setq):
            self.gen_setq(e, env)
        elif isinstance(e, Loop):
            self.gen_loop(e, env)
        else:
            self.gen_call(e, env, tail=tail)

    def gen_flags(self, e: BinOp, env: Env) -> None:
        """AC = left op right"""
        opcode = ARITH_OPS.get(e.op, Opcode.SUB)
        self.gen_expr(e.left, env)
        operand = self.simple_operand(e.right, env)
        if operand is not None:
            self.emit(opcode, *operand)
            return
        self.push()
        self.gen_expr(e.right, env)
        self.push()
        self.emit(Opcode.LD, OperandType.STACK_OFFSET, 1)
        self.emit(opcode, OperandType.STACK_OFFSET, 0)
        self.pop()
        self.pop()

    def gen_binop(self, e: BinOp, env: Env) -> None:
        self.gen_flags(e, env)
        if e.op in ARITH_OPS:
            return
        true_label, end_label = self.new_label("true"), self.new_label("end")
        self.emit(COMPARE_TRUE[e.op], OperandType.ADDRESS, true_label)
        self.emit(Opcode.LD, OperandType.IMMEDIATE, 0)
        self.emit(Opcode.JMP, OperandType.ADDRESS, end_label)
        self.label(true_label)
        self.emit(Opcode.LD, OperandType.IMMEDIATE, 1)
        self.label(end_label)

    def gen_not(self, e: Not, env: Env) -> None:
        self.gen_expr(e.arg, env)
        self.emit(Opcode.SUB, OperandType.IMMEDIATE, 0)
        true_label, end_label = self.new_label("true"), self.new_label("end")
        self.emit(Opcode.JE, OperandType.ADDRESS, true_label)
        self.emit(Opcode.LD, OperandType.IMMEDIATE, 0)
        self.emit(Opcode.JMP, OperandType.ADDRESS, end_label)
        self.label(true_label)
        self.emit(Opcode.LD, OperandType.IMMEDIATE, 1)
        self.label(end_label)

    def gen_jump_if_false(self, cond: Expr, target: str, env: Env) -> None:
        if isinstance(cond, BinOp) and cond.op in COMPARE_FALSE:
            self.line = cond.line
            self.gen_flags(cond, env)
            self.emit(COMPARE_FALSE[cond.op], OperandType.ADDRESS, target)
        elif isinstance(cond, Not):
            self.gen_expr(cond.arg, env)
            self.emit(Opcode.SUB, OperandType.IMMEDIATE, 0)
            self.emit(Opcode.JNE, OperandType.ADDRESS, target)
        else:
            self.gen_expr(cond, env)
            self.emit(Opcode.SUB, OperandType.IMMEDIATE, 0)
            self.emit(Opcode.JE, OperandType.ADDRESS, target)

    def gen_if(self, e: If, env: Env, *, tail: bool) -> None:
        else_label, end_label = self.new_label("else"), self.new_label("endif")
        self.gen_jump_if_false(e.cond, else_label, env)
        self.gen_expr(e.then, env, tail=tail)
        self.emit(Opcode.JMP, OperandType.ADDRESS, end_label)
        self.label(else_label)
        self.gen_expr(e.otherwise, env, tail=tail)
        self.label(end_label)

    def gen_let(self, e: Let, env: Env, *, tail: bool) -> None:
        for name, init in e.bindings:
            self.gen_expr(init, env)
            self.push()
            env = (*env, (name, -self.ctx.depth, False))
        self.gen_body(e.body, env, tail=tail)
        for _ in e.bindings:
            self.pop()

    def gen_setq(self, e: Setq, env: Env) -> None:
        self.gen_expr(e.value, env)
        self.line = e.line
        operand_type, operand, readonly = self.lookup(e.name, env)
        if readonly:
            raise self.error(f"счётчик цикла {e.name!r} нельзя изменять")
        self.emit(Opcode.ST, operand_type, operand)

    def gen_loop(self, e: Loop, env: Env) -> None:
        self.gen_expr(e.start, env)
        self.push()
        slot_i = -self.ctx.depth
        self.gen_expr(e.end, env)
        self.push()
        slot_limit = -self.ctx.depth
        self.emit(Opcode.LD, OperandType.IMMEDIATE, 0)
        self.push()
        slot_result = -self.ctx.depth
        body_env = (*env, (e.var, slot_i, True))
        cond_label, end_label = self.new_label("loop"), self.new_label("loopend")

        def at(slot: int) -> int:
            return self.ctx.depth + slot

        self.label(cond_label)
        self.emit(Opcode.LD, OperandType.STACK_OFFSET, at(slot_i))
        self.emit(Opcode.SUB, OperandType.STACK_OFFSET, at(slot_limit))
        self.emit(Opcode.JG, OperandType.ADDRESS, end_label)
        self.gen_body(e.body, body_env, tail=False)
        self.emit(Opcode.ST, OperandType.STACK_OFFSET, at(slot_result))
        self.emit(Opcode.LD, OperandType.STACK_OFFSET, at(slot_i))
        self.emit(Opcode.ADD, OperandType.IMMEDIATE, 1)
        self.emit(Opcode.ST, OperandType.STACK_OFFSET, at(slot_i))
        self.emit(Opcode.JMP, OperandType.ADDRESS, cond_label)
        self.label(end_label)
        self.emit(Opcode.LD, OperandType.STACK_OFFSET, at(slot_result))
        self.pop()
        self.pop()
        self.pop()

    # вызовы

    def gen_call(self, e: Call, env: Env, *, tail: bool) -> None:
        if e.name in BUILTIN_ARITY:
            self.gen_builtin(e, env)
            return
        function = self.functions.get(e.name)
        if function is None:
            raise self.error(f"вызов неопределённой функции {e.name!r}")
        if len(e.args) != len(function.params):
            raise self.error(f"{e.name}: ожидалось аргументов {len(function.params)}, получено {len(e.args)}")
        if tail and self.ctx.name == e.name:
            self.gen_tail_call(e, env)
            return
        for arg in e.args:
            self.gen_expr(arg, env)
            self.push()
        self.emit(Opcode.CALL, OperandType.ADDRESS, f"fn_{e.name}")
        for _ in e.args:
            self.pop()

    def gen_tail_call(self, e: Call, env: Env) -> None:
        """Рекурсия: новые аргументы записываются на место параметров"""
        base_depth = self.ctx.depth
        count = len(e.args)
        for arg in e.args:
            self.gen_expr(arg, env)
            self.push()
        for index in range(count):
            self.emit(Opcode.LD, OperandType.STACK_OFFSET, count - 1 - index)
            self.emit(Opcode.ST, OperandType.STACK_OFFSET, self.ctx.depth + count - index)
        for _ in range(self.ctx.depth):
            self.emit(Opcode.POP)
        self.emit(Opcode.JMP, OperandType.ADDRESS, f"fn_{e.name}")
        self.ctx.depth = base_depth

    def gen_builtin(self, e: Call, env: Env) -> None:
        if len(e.args) != BUILTIN_ARITY[e.name]:
            raise self.error(f"{e.name}: ожидалось аргументов {BUILTIN_ARITY[e.name]}, получено {len(e.args)}")
        name = e.name
        if name == "port-in":
            self.emit(Opcode.IN, OperandType.PORT, self.port(e.args[0]))
        elif name == "port-out":
            port = self.port(e.args[0])
            self.gen_expr(e.args[1], env)
            self.emit(Opcode.OUT, OperandType.PORT, port)
        elif name == "getchar":
            self.gen_address(e.args[0], e.args[1], env)
            self.push()
            self.emit(Opcode.LD, OperandType.STACK_POINTER, 0)
            self.pop()
        elif name == "setchar":
            self.gen_address(e.args[0], e.args[1], env)
            self.push()
            self.gen_expr(e.args[2], env)
            self.emit(Opcode.ST, OperandType.STACK_POINTER, 0)
            self.pop()
        elif name == "makestring":
            self.emit(Opcode.LD, OperandType.IMMEDIATE, self.layout.buffers[id(e)])
        elif name in ("ei", "di"):
            self.emit(Opcode.EI if name == "ei" else Opcode.DI)
            self.emit(Opcode.LD, OperandType.IMMEDIATE, 0)
        else:
            self.emit(Opcode.HLT)

    def gen_address(self, base: Expr, index: Expr, env: Env) -> None:
        """AC = base + index, при index = 0 сложение не нужно"""
        if isinstance(index, Num) and index.value == 0:
            self.gen_expr(base, env)
        else:
            self.gen_expr(BinOp("+", base, index, base_line(base)), env)

    def port(self, e: Expr) -> int:
        if not isinstance(e, Num) or not 0 <= e.value <= PORT_LIMIT:
            raise self.error("номер порта должен быть числовым литералом")
        return e.value

# Программа

@dataclass
class Program:
    code: list[Instruction]
    data: list[int]
    labels: dict[str, int]
    origins: list[str]
    ast: str
    sources: dict[str, list[str]]

    def code_bytes(self) -> bytes:
        return b"".join(encode_instruction(i) for i in self.code)

    def data_bytes(self) -> bytes:
        return b"".join(encode_word(w) for w in self.data)

    def listing(self) -> str:
        """Отладка: адрес -- HEX -- мнемоника; источник"""
        names: dict[int, list[str]] = {}
        for name, index in self.labels.items():
            names.setdefault(index, []).append(name)
        lines = [f"; code: {len(self.code)} instructions"]
        for address, instr in enumerate(self.code):
            lines.extend(f"; {name}:" for name in names.get(address, []) if not name.startswith("."))
            text = instruction_to_string(instr)
            if instr.operand_type == OperandType.ADDRESS and instr.opcode.name.startswith(("J", "CALL")):
                target = [n for n in names.get(instr.operand, []) if not n.startswith(".")]
                if target:
                    text += f" <{target[0]}>"
            origin = self.origins[address]
            kind, _, line = origin.partition(":")
            if kind == "vector":
                comment = f"; interrupt vector {line}"
            else:
                src_lines = self.sources.get(kind, [])
                number = int(line)
                source = src_lines[number - 1].strip() if 1 <= number <= len(src_lines) else ""
                comment = f"; {origin} {source}"
            lines.append(f"{address:5d} - {encode_instruction(instr).hex()} - {text:<24} {comment}")
        lines.append(f"; data: {len(self.data)} words")
        for address, word in enumerate(self.data):
            shown = f"{word}" + (f" {chr(word)!r}" if 32 <= word < 127 else "")
            lines.append(f"{address:5d} - {word & 0xFFFFFFFF:08x} - {shown}")
        return "\n".join(lines) + "\n"


def _called_functions(item: Toplevel) -> set[str]:
    return {n.name for n in iter_nodes(item) if isinstance(n, Call)}


def _select_items(
    user: list[Toplevel], stdlib: list[Toplevel]
) -> tuple[list[tuple[Toplevel, str]], dict[str, tuple[Defun, str]]]:
    functions: dict[str, tuple[Defun, str]] = {}
    for items, origin in ((stdlib, "stdlib"), (user, "user")):
        for item in items:
            if isinstance(item, Defun):
                if item.name in functions:
                    raise TranslatorError(f"{origin}:{item.line}: повторное определение функции {item.name!r}")
                functions[item.name] = (item, origin)
    stdlib_module = [i for i in stdlib if not isinstance(i, Defun)]

    needed: set[str] = set()
    pending = list(user)
    module_included = False
    while pending:
        item = pending.pop()
        for name in _called_functions(item):
            if name in functions and name not in needed:
                needed.add(name)
                pending.append(functions[name][0])
        if not module_included and STDLIB_INPUT_TRIGGER in needed:
            module_included = True
            pending.extend(stdlib_module)

    selected: list[tuple[Toplevel, str]] = []
    for item in stdlib:
        if isinstance(item, Defun):
            if item.name in needed:
                selected.append((item, "stdlib"))
        elif module_included:
            selected.append((item, "stdlib"))
    selected += [(i, "user") for i in user]
    return selected, functions


def translate(source: str, stdlib: str | None = None) -> Program:
    user_items = Parser("user").parse_program(source)
    stdlib_items = Parser("stdlib").parse_program(stdlib) if stdlib else []
    selected, _ = _select_items(user_items, stdlib_items)

    layout = build_layout(selected)
    functions = {i.name: i for i, _ in selected if isinstance(i, Defun)}
    for item, origin in selected:
        if isinstance(item, Define) and item.name in functions:
            raise TranslatorError(f"{origin}:{item.line}: имя {item.name!r} уже занято функцией")
    handlers = [(i, o) for i, o in selected if isinstance(i, Defint)]
    if len(handlers) > 1:
        raise TranslatorError(f"{handlers[1][1]}:{handlers[1][0].line}: обработчик прерывания уже определён")

    gen = CodeGen(functions, layout)
    for item, origin in selected:
        if isinstance(item, Defun):
            gen.gen_function(item, origin)
    for item, origin in handlers:
        gen.gen_interrupt(item, origin)
    gen.gen_toplevel(selected)

    vector = [
        Asm(Opcode.JMP, OperandType.ADDRESS, "start", "vector", 0),
        Asm(Opcode.JMP, OperandType.ADDRESS, "int_handler", "vector", 1)
        if handlers
        else Asm(Opcode.HLT, OperandType.NONE, 0, "vector", 1),
    ]
    sequence: list[Asm | str] = [*vector, *gen.items]

    labels: dict[str, int] = {}
    count = 0
    for entry in sequence:
        if isinstance(entry, str):
            labels[entry] = count
        else:
            count += 1
    code: list[Instruction] = []
    origins: list[str] = []
    for entry in sequence:
        if isinstance(entry, str):
            continue
        operand = labels[entry.operand] if isinstance(entry.operand, str) else entry.operand
        code.append(Instruction(entry.opcode, entry.operand_type, operand, entry.line))
        origins.append(f"{entry.origin}:{entry.line}")
    return Program(
        code=code,
        data=layout.words,
        labels=labels,
        origins=origins,
        ast=ast_to_string(user_items),
        sources={"user": source.splitlines(), "stdlib": (stdlib or "").splitlines()},
    )

# Командная строка

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Транслятор Lisp -> машинный код (acc, harv)")
    parser.add_argument("source", help="исходный текст программы")
    parser.add_argument("code", help="выходной бинарный файл команд")
    parser.add_argument("data", help="выходной бинарный файл данных")
    parser.add_argument("--listing", help="текстовый листинг (по умолчанию <code>.lst)")
    parser.add_argument("--ast", help="записать AST программы в файл")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stdlib", default=str(DEFAULT_STDLIB), help="файл стандартной библиотеки")
    group.add_argument("--no-stdlib", action="store_true", help="не подключать стандартную библиотеку")
    args = parser.parse_args(argv)

    try:
        source = Path(args.source).read_text(encoding="utf-8")
        stdlib = None if args.no_stdlib else Path(args.stdlib).read_text(encoding="utf-8")
        program = translate(source, stdlib)
    except (TranslatorError, OSError) as exc:
        sys.stderr.write(f"ошибка: {exc}\n")
        return 1

    Path(args.code).write_bytes(program.code_bytes())
    Path(args.data).write_bytes(program.data_bytes())
    Path(args.listing or f"{args.code}.lst").write_text(program.listing(), encoding="utf-8")
    if args.ast:
        Path(args.ast).write_text(program.ast, encoding="utf-8")
    sys.stderr.write(f"команд: {len(program.code)}, слов данных: {len(program.data)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
