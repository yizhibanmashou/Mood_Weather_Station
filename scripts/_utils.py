"""
Shared utility functions for Mood Weather Station pipeline scripts.
"""
import importlib.util
import sys
from pathlib import Path


def load_local_module(name, scripts_dir=None):
    """Load a Python file as a module by its filename (without .py extension).

    Unified import mechanism for scripts/ modules — replaces three different
    dynamic import patterns that existed across the codebase.
    """
    if scripts_dir is None:
        scripts_dir = Path(__file__).resolve().parent
    path = scripts_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
