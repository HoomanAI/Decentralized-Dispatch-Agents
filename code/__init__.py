"""Reproducible simulation study for the AEFDP paper.

The project layout intentionally follows the paper brief and uses a top-level package
named ``code``. We re-export the public interactive-console API from Python's standard
library module of the same name so tools such as pdb and pytest continue to work.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

_stdlib_path = Path(sysconfig.get_path("stdlib")) / "code.py"
_stdlib_spec = importlib.util.spec_from_file_location("_stdlib_code", _stdlib_path)
if _stdlib_spec is None or _stdlib_spec.loader is None:
    raise ImportError(f"Cannot load the standard library code module from {_stdlib_path}")
_stdlib_code = importlib.util.module_from_spec(_stdlib_spec)
_stdlib_spec.loader.exec_module(_stdlib_code)

InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
InteractiveConsole = _stdlib_code.InteractiveConsole
interact = _stdlib_code.interact
compile_command = _stdlib_code.compile_command

__all__ = [
    "InteractiveInterpreter",
    "InteractiveConsole",
    "interact",
    "compile_command",
]
