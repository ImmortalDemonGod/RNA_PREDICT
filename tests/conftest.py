"""
Pytest configuration and plugins.
"""

import faulthandler
import io

# Enable faulthandler only if sys.stderr supports fileno (not always true under pytest-xdist or some CI)
def _safe_enable_faulthandler():
    try:
        faulthandler.enable()
        # Set timeout to 60 seconds
        faulthandler.dump_traceback_later(60, repeat=False)
    except (io.UnsupportedOperation, AttributeError):
        pass  # Running in an environment where fileno is not supported

_safe_enable_faulthandler()


def pytest_configure(config):
    """Configure pytest."""
    pass  # No plugins to register
