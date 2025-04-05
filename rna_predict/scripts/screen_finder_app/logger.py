# rna_predict/scripts/logger.py
import logging

def setup_logger(level=logging.INFO):
    """
    Sets up and returns a configured logger instance.

    :param level: The logging level (e.g., logging.INFO, logging.DEBUG).
    :return: Configured logger instance.
    """
    # Configure root logger if not already configured
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Get a logger specific to this module/application part
    logger = logging.getLogger(__name__)
    logger.setLevel(level) # Ensure the specific logger respects the level

    # Prevent adding multiple handlers if called multiple times
    if not logger.handlers:
        # Create a handler (e.g., StreamHandler to output to console)
        # BasicConfig already adds a handler to the root logger,
        # so child loggers will propagate messages to it by default.
        # If specific handling per logger (e.g., file output) is needed, add handlers here.
        pass # Using root logger's handler configured by basicConfig

    return logger

# Example of how to get a logger elsewhere:
# from logger import setup_logger
# logger = setup_logger()
# logger.info("This is an info message.")