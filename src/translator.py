from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import re


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
    "port-in": 1,       # (port-in 0)
    "port-out": 2,      # (port-out 1 65)
    "getchar": 2,       # (getchar buffer index)
    "setchar": 3,       # (setchar buffer index c)
    "makestring": 1,    # (makestring length)
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
    type: str  # "(", ")", "num", "char", "str", "sym"
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
    i, chars = start + 1, []
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


SExpr = Union[SAtom, SList]


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


Expr = Union[Num, Str, Var, BinOp, Not, If, Progn, Let, Setq, Loop, Call]


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


Toplevel = Union[Define, Defun, Defint, Expr]


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
        builder = {
            "not": self.parse_not,
            "if": self.parse_if,
            "progn": self.parse_progn,
            "let": self.parse_let,
            "setq": self.parse_setq,
            "loop": self.parse_loop,
        }.get(word)
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
def _describe(node: object) -> tuple[str, list[tuple[str, object]]]:
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


def iter_nodes(node: object):  # noqa: ANN201
    """Все узлы поддерева в прямом порядке."""
    yield node
    for _, child in _describe(node)[1]:
        yield from iter_nodes(child)
