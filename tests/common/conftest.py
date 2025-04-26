"""
Configuration file for pytest.
This file is automatically loaded by pytest and can be used to define fixtures
and other test setup.
"""

import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from scripts
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
