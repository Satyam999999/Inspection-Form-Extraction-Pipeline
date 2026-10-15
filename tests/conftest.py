"""
pytest configuration: set PYTHONPATH so src/ is importable.
"""
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
