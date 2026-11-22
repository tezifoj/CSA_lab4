"""
Golden-тесты
Один тест -- это файл `tests/<имя>.yml` со входом (`in_*`, `config`) и результатами (`out_*`)
"""

import io
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from machine import Machine, parse_schedule
from translator import DEFAULT_STDLIB, translate

GOLDEN_DIR = Path(__file__).parent
DEFAULT_CONFIG = {"limit": 1_000_000, "cache_lines": 4, "log_head": 40, "log_tail": 10}


class _Dumper(yaml.SafeDumper):
    """Многострочные строки пишутся блоком `|`, порядок ключей сохраняется"""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _represent_str)


def _clean(text: str) -> str:
    """Убрать пробелы в конце строк"""
    return "".join(line.rstrip() + "\n" for line in text.splitlines())


def _interrupt_window(lines: list[str], start: int) -> list[str]:
    """Вход в первое прерывание после строки start и возврат из него в основную программу"""
    entry = next((i for i in range(start, len(lines)) if "INTERRUPT:" in lines[i]), None)
    if entry is None:
        return []
    window = lines[entry - 2 : entry + 10]
    back = next((i for i in range(entry, len(lines)) if "| MAIN | IRET" in lines[i]), None)
    if back is not None:
        window += ["...", *lines[back - 2 : back + 2]]
    return window


def _trim_log(log: str, head: int, tail: int) -> str:
    """Начало и конец журнала; если были прерывания - ещё вход в первое из них и возврат"""
    lines = log.splitlines()
    if len(lines) <= head + tail:
        return _clean(log)
    middle = _interrupt_window(lines, head)
    parts = [*lines[:head], "...", *middle, "..." if middle else "", *lines[-tail:]]
    return _clean("\n".join(part for part in parts if part))


def run_case(case: dict[str, Any]) -> dict[str, str]:
    config = {**DEFAULT_CONFIG, **case.get("config", {})}
    stdlib = DEFAULT_STDLIB.read_text(encoding="utf-8")
    program = translate(case["in_source"], stdlib)

    log = io.StringIO()
    machine = Machine(
        program.code_bytes(),
        program.data_bytes(),
        parse_schedule(case.get("in_input", "[]")),
        log=log,
        cache_lines=config["cache_lines"],
    )
    result = machine.run(config["limit"])

    stats = (
        f"ticks: {result.ticks}\n"
        f"instructions: {result.instructions}\n"
        f"interrupts: {result.interrupts}\n"
        f"cache read hits: {result.read_hits}\n"
        f"cache read misses: {result.read_misses}\n"
        f"stop reason: {result.reason}\n"
    )
    return {
        "out_ast": program.ast,
        "out_listing": _clean(program.listing()),
        "out_stdout": result.output,
        "out_stats": stats,
        "out_log": _trim_log(log.getvalue(), config["log_head"], config["log_tail"]),
    }


def _dump(path: Path, case: dict[str, Any]) -> None:
    text = yaml.dump(case, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=1000)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("path", sorted(GOLDEN_DIR.glob("*.yml")), ids=lambda p: p.stem)
def test_golden(path: Path) -> None:
    case = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = run_case(case)

    if os.environ.get("UPDATE_GOLDEN"):
        _dump(path, {**{k: v for k, v in case.items() if not k.startswith("out_")}, **actual})
        return

    for key, value in actual.items():
        assert value == case[key], f"{path.stem}: расхождение в {key}"
