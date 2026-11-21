"""
Golden-тесты
Один тест -- это файл `tests/golden/<имя>.yml` со входом (`in_*`, `config`) и результатами (`out_*`)
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


def _trim_log(log: str, head: int, tail: int) -> str:
    lines = log.splitlines()
    if len(lines) <= head + tail:
        return _clean(log)
    omitted = len(lines) - head - tail
    return _clean("\n".join([*lines[:head], f"... {omitted} lines omitted ...", *lines[-tail:]]))


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
