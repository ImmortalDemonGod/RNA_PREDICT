# Configuration settings for the screen finder application
import os

# Base directory for the script - adjust if necessary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to store template images
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Configuration file for templates
TEMPLATES_CONFIG = os.path.join(TEMPLATES_DIR, "templates_config.json")

# Default matching threshold (0.0 to 1.0) - higher means stricter match
DEFAULT_THRESHOLD = 0.8

# Log file path
LOG_FILE = os.path.join(BASE_DIR, "screen_finder.log")

# Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL = "INFO"

# Ensure templates directory exists
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)
    print(f"Created templates directory at: {TEMPLATES_DIR}")

# Example templates_config.json structure:
# {
#   "templates": [
#     {
#       "name": "template1",
#       "file": "template1.png",
#       "threshold": 0.85,
#       "action": {
#         "type": "click",
#         "button": "left",
#         "clicks": 1
#       }
#     },
#     {
#       "name": "template2",
#       "file": "template2.png",
#       "action": {
#         "type": "text",
#         "text": "Hello, World!"
#       }
#     },
#     {
#       "name": "template3",
#       "file": "template3.png",
#       "action": {
#         "type": "clipboard",
#         "text": "Copied text"
#       }
#     }
#   ]
# }
# Note: If threshold is not specified for a template, DEFAULT_THRESHOLD is used.
