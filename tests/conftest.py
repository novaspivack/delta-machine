"""Pytest configuration for Delta_machine tests."""

import sys
from pathlib import Path

# Add Delta_machine/src to PYTHONPATH
root = Path(__file__).resolve().parents[1]
src_dir = root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add parent directory to PYTHONPATH so pr0_system can be imported
parent_dir = root.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

