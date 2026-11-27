"""Tests for the run CLI argument handling (Requirement 21.4). No dataset needed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py")


@pytest.mark.parametrize("args,flag", [
    (["--synthetic-samples", "1"], "synthetic-samples"),
    (["--sigma", "0"], "sigma"),
    (["--method", "foo"], "method"),
])
def test_invalid_flag_exits_nonzero_naming_flag(args, flag):
    """Invalid flag values exit non-zero with the flag named in stderr (Requirement 21.4)."""
    completed = subprocess.run([sys.executable, _SCRIPT, *args], capture_output=True, text=True)
    assert completed.returncode != 0
    assert flag in completed.stderr
