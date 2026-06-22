"""Import local worksheet helpers during math toolkit activation."""

from __future__ import annotations

import sys
from pathlib import Path

_support_dir = Path(__file__).resolve().parent / ".support"
_helper_path = _support_dir / "pure_trig_fourier_matching_game.py"

if _helper_path.is_file():
    _support_path = str(_support_dir)
    if _support_path not in sys.path:
        sys.path.insert(0, _support_path)

    from pure_trig_fourier_matching_game import create_mystery_sine_function
