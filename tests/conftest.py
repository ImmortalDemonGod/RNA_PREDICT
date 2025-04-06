"""
Pytest configuration and plugins.
"""

import pytest
import faulthandler

# Enable faulthandler with a timeout
faulthandler.enable()
# Set timeout to 60 seconds
faulthandler.dump_traceback_later(60, repeat=False)

def pytest_configure(config):
    """Configure pytest."""
    pass  # No plugins to register 