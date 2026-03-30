#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ogbench_utils.vr_mapping_web import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
