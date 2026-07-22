import sys
from pathlib import Path

# Add repo root to path so tests can import src.* modules
root_path = Path(__file__).parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
