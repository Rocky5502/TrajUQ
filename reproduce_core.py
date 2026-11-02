#!/usr/bin/env python
from runpy import run_path
from pathlib import Path

run_path(str(Path(__file__).parent / "reproduction" / "reproduce_core.py"), run_name="__main__")
