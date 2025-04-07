# rna_predict/scripts/gui_launcher.py
import logging
import os
from typing import Any, Dict, Optional, Tuple

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image  # For loading image data

# Local application imports (assuming execution as a module)
from .config import LOGGING_LEVEL, TEMPLATE_PATH
from .config import THRESHOLD as SEARCH_THRESHOLD
from .logger import setup_logger
from .main import find_template_on_screen

# Global logger instance
# Note: This will be assigned in _setup_global_logger()
# We assert it's not None before use in other functions or module level.
logger: logging.Logger


def _setup_global_logger() -> logging.Logger:
    """Sets up and returns a logger instance based on config."""
    # Removed 'global logger' as we return the instance to be assigned globally
    log_level_str: str = "INFO"
    try:
        # Ensure LOGGING_LEVEL is treated as str
        log_level_str = str(LOGGING_LEVEL)
    except NameError:
        print("Warning: LOGGING_LEVEL not found in config, using INFO level.")

    log_level_map: Dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level_int = log_level_map.get(log_level_str.upper(), logging.INFO)

    try:
        # Assign to a local variable first
        _logger = setup_logger(level=log_level_int)
    except NameError:
        # Fallback if setup_logger is somehow not imported
        logging.basicConfig(level=log_level_int)
        _logger = logging.getLogger(__name__)
        _logger.warning(
            "setup_logger function not found, using basic logging configuration."
        )
    # Ensure the function returns the created logger instance
    return _logger
    return _logger


# Initialize the global logger immediately after imports and function definition
logger = _setup_global_logger()


# --- Resolve Template Path ---
full_template_path: Optional[str] = None
template_load_error: Optional[str] = None
try:
    script_dir = os.path.dirname(__file__)
    # TEMPLATE_PATH is now just the filename, relative to script_dir
    path1 = os.path.abspath(os.path.join(script_dir, TEMPLATE_PATH))

    logger.info(f"Checking for template at: {path1}")
    if os.path.exists(path1):
        full_template_path = path1
        logger.info(f"Found template at path: {full_template_path}")
    else:
        raise FileNotFoundError(f"Template image not found at {path1}")

except FileNotFoundError as e:
    error_msg = f"Template image not found: {e}"
    logger.error(error_msg)
    template_load_error = error_msg
except NameError:
    error_msg = "TEMPLATE_PATH configuration variable not found."
    logger.error(error_msg)
    template_load_error = error_msg
except Exception as e:
    error_msg = f"Error resolving template path: {e}"
    logger.error(f"Unexpected error resolving template path: {e}", exc_info=True)
    template_load_error = error_msg

# --- Dear PyGui Setup ---
dpg.create_context()

# --- Texture Loading ---
texture_id = None
texture_width = 0
texture_height = 0

if full_template_path:
    try:
        img = Image.open(full_template_path).convert("RGBA")
        img_data = np.array(img, dtype=np.float32) / 255.0
        texture_width = img.width
        texture_height = img.height

        with dpg.texture_registry(show=False):
            texture_id = dpg.add_static_texture(
                width=texture_width,
                height=texture_height,
                default_value=img_data.ravel(),  # Flatten the array
                parent=dpg.last_container(),
            )
        logger.info(
            f"Template image loaded into texture registry (ID: {texture_id}, Size: {texture_width}x{texture_height})."
        )
    except Exception as e:
        error_msg = f"Failed to load template image into texture: {e}"
        logger.error(error_msg, exc_info=True)
        template_load_error = error_msg  # Update error message
        texture_id = None  # Ensure texture_id is None if loading failed
else:
    # If template path wasn't resolved initially
    logger.warning("Skipping texture loading as template path was not resolved.")


# --- Callback Function ---
def search_callback() -> None:
    logger.info("Search button clicked.")
    dpg.set_value("-STATUS-", "Searching...")

    if not full_template_path or texture_id is None:
        status_msg = "Cannot search: Template image not loaded."
        logger.error(status_msg)
        dpg.set_value("-STATUS-", status_msg)
        # Consider adding a Dear PyGui modal/popup for errors later
        return

    try:
        if (
            "find_template_on_screen" not in globals()
            and "find_template_on_screen" not in locals()
        ):
            raise NameError("find_template_on_screen function not loaded.")

        threshold = (
            SEARCH_THRESHOLD if "SEARCH_THRESHOLD" in globals() else 0.8
        )  # Default if not found
        logger.debug(
            f"Calling find_template_on_screen with template='{full_template_path}', threshold={threshold}"
        )

        result: Dict[str, Any] = find_template_on_screen(
            logger=logger, template_path=full_template_path, threshold=threshold
        )

        if result.get("found", False):
            coords: Tuple[int, int] = result.get("coordinates", (-1, -1))
            correlation: float = result.get("correlation", 0.0)  # Get correlation score
            # Get monitor details dictionary and format description string
            monitor_info = result.get("monitor_info", {})
            monitor_desc = f"Monitor (L:{monitor_info.get('left', 'N/A')}, T:{monitor_info.get('top', 'N/A')}, W:{monitor_info.get('width', 'N/A')}, H:{monitor_info.get('height', 'N/A')})"
            # Format status text including correlation and detailed monitor info
            status_text = f"Found at: ({coords[0]}, {coords[1]}) on {monitor_desc} (Corr: {correlation:.4f})"
            dpg.set_value("-STATUS-", status_text)
            logger.info(
                f"Template found via GUI at {coords} on {monitor_desc} with correlation {correlation:.4f}"
            )  # Log correlation
        else:
            status_text = "Template not found."
            dpg.set_value("-STATUS-", status_text)
            logger.info("Template not found via GUI.")

    except NameError as e:
        error_msg = f"Search function unavailable: {e}"
        dpg.set_value("-STATUS-", error_msg)
        logger.error(error_msg)
    except Exception as e:
        error_msg = f"Search Error: {type(e).__name__} - {e}"
        dpg.set_value("-STATUS-", error_msg)
        logger.error(f"Error during template search via GUI: {e}", exc_info=True)


# --- GUI Layout ---
with dpg.window(label="Template Search", tag="Primary Window"):
    dpg.add_text("Template Search GUI")

    if texture_id:
        # Set a fixed width for the image display initially for testing
        display_width = 300  # Fixed width
        # Calculate height proportionally, prevent division by zero
        display_height = (
            int(texture_height * (display_width / texture_width))
            if texture_width > 0
            else 0
        )

        if display_width > 0 and display_height > 0:
            dpg.add_image(
                texture_id,
                width=display_width,
                height=display_height,
                tag="-TEMPLATE_IMG-",
            )
            logger.info(
                f"Template image widget added to layout (Display Size: {display_width}x{display_height})."
            )
        else:
            logger.warning(
                "Template image widget not added due to zero dimensions after calculation."
            )
            dpg.add_text(
                "Error: Could not display template image (invalid dimensions).",
                color=(255, 0, 0, 255),
            )
    elif template_load_error:
        dpg.add_text(
            f"Error: {template_load_error}", color=(255, 0, 0, 255)
        )  # Red color
    else:
        dpg.add_text(
            "Template image path could not be resolved.", color=(255, 0, 0, 255)
        )

    dpg.add_spacer(height=10)
    dpg.add_text("Status: Ready", tag="-STATUS-")
    dpg.add_spacer(height=10)
    # Removed horizontal group for simplicity
    dpg.add_button(label="Start Search", callback=search_callback, tag="-SEARCH-")
    # Exit is handled by closing the window in Dear PyGui

# --- Viewport and Start ---
dpg.create_viewport(
    title="Template Search GUI", width=600, height=500
)  # Increased height
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
logger.info("Starting Dear PyGui application.")
dpg.start_dearpygui()

# --- Cleanup ---
logger.info("Dear PyGui application stopped.")
dpg.destroy_context()
logger.info("GUI Launcher script finished.")
