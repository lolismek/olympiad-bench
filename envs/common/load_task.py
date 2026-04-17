"""Import a task module by directory path (handles paths starting with a digit)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(path: Path, qualified_name: str) -> ModuleType:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_task_env(task_dir: Path, qualified_name: str = "_task_env"):
    return load_module(Path(task_dir) / "env.py", qualified_name)
