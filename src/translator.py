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
