"""
Pytest configuration and environment fixtures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src directory is on sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Prevent OpenMP runtime library clash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
