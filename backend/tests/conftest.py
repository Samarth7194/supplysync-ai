"""Test configuration - ensure backend/src is on the import path."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
