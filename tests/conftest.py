"""Ensure the project root is on sys.path so `from pci import ...` works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
