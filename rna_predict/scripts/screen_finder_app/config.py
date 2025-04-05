# rna_predict/scripts/config.py

# Path to the template image file used for matching
TEMPLATE_PATH = 'template.png' # Default path, relative to the scripts directory
# Consider making this configurable via arguments or env variables

# Confidence threshold for template matching (0.0 to 1.0)
# Lower values are less strict, higher values are more strict.
THRESHOLD = 0.8

# Logging level for the application
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOGGING_LEVEL = 'INFO'

# Add any other configuration parameters needed for the application below
# Example: Default region if none selected, output directories, etc.