import logging
import os
import cv2
import numpy as np
import pyautogui
import pyperclip
from mss import mss
from .config import LOG_FILE, LOG_LEVEL
from .template_loader import load_and_preload_templates
import time
import subprocess

# Setup logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console handler (optional, for seeing logs in console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

def execute_action(logger, action_config: dict, coordinates: tuple) -> None:
    """
    Performs the user-defined action (click, text input, clipboard, etc.) at the given coordinates.
    """
    import pyautogui  # Lazy import to avoid overhead if actions are never used
    action_type = action_config.get("action", "")
    logger.info(f"Executing action '{action_type}' at {coordinates} with config: {action_config}")

    if action_type == "click":
        pyautogui.click(x=coordinates[0], y=coordinates[1])
        time.sleep(0.5)  # Wait for click to register

    elif action_type == "text_input":
        text = action_config.get("text", "")
        pyautogui.click(x=coordinates[0], y=coordinates[1])
        time.sleep(0.5)  # Wait for click to register
        pyautogui.typewrite(text, interval=0.05)

    elif action_type == "clipboard":
        operation = action_config.get("operation", "copy")
        pyautogui.click(x=coordinates[0], y=coordinates[1])
        time.sleep(0.5)  # Wait for click to register
        if operation == "copy":
            pyautogui.hotkey('command', 'c')
            time.sleep(0.5)  # Wait for copy to complete
        elif operation == "paste":
            pyautogui.hotkey('command', 'v')
            time.sleep(0.5)  # Wait for paste to complete

    elif action_type == "double_click":
        pyautogui.doubleClick(x=coordinates[0], y=coordinates[1])
        time.sleep(0.5)  # Wait for double click to register

    else:
        logger.info(f"No recognized action for '{action_type}'. Doing nothing.")

def main():
    """Main function to load templates, search screens, and execute actions."""
    logger.info("Starting screen finder process.")

    # Load templates and their configurations
    loaded_templates = load_and_preload_templates()
    if not loaded_templates:
        logger.error("No templates loaded. Exiting.")
        return

    logger.info(f"Loaded {len(loaded_templates)} templates.")

    try:
        with mss() as sct:
            # Get information about all monitors
            monitors = sct.monitors[1:] # Index 0 is the combined virtual screen
            logger.info(f"Detected {len(monitors)} monitors.")

            found_match_in_cycle = False # Flag to track if any match was found

            # Capture screenshot for each monitor
            for i, monitor in enumerate(monitors):
                logger.debug(f"Processing Monitor {i+1}: {monitor}")
                # Capture the screen of the current monitor
                sct_img = sct.grab(monitor)
                # Convert to an OpenCV image (BGRA to BGR, then to Grayscale)
                screen_img_bgr = np.array(sct_img)
                screen_img_gray = cv2.cvtColor(screen_img_bgr, cv2.COLOR_BGRA2GRAY)

                # Iterate through each loaded template
                for template_data in loaded_templates:
                    template_name = template_data["name"]
                    template_img = template_data["image"]
                    threshold = template_data["threshold"]
                    action_config = template_data["action"]
                    tH, tW = template_img.shape[:2]

                    logger.debug(f"Searching for template '{template_name}' on Monitor {i+1}...")

                    # Perform template matching
                    # TM_CCOEFF_NORMED is generally robust
                    result = cv2.matchTemplate(screen_img_gray, template_img, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    logger.debug(f"Template '{template_name}' - Max match value: {max_val:.4f} (Threshold: {threshold})")

                    # Check if the match exceeds the threshold
                    if max_val >= threshold:
                        found_match_in_cycle = True
                        # Get the top-left corner of the matched area
                        match_x, match_y = max_loc
                        # Calculate the center coordinates relative to the monitor
                        center_x_rel = match_x + tW // 2
                        center_y_rel = match_y + tH // 2
                        # Convert to absolute screen coordinates
                        center_x_abs = monitor["left"] + center_x_rel
                        center_y_abs = monitor["top"] + center_y_rel

                        logger.info(f"Match found for '{template_name}' on Monitor {i+1} at relative ({match_x}, {match_y}), "
                                    f"absolute center ({center_x_abs}, {center_y_abs}) with confidence {max_val:.4f}")

                        # Execute the configured action
                        execute_action(logger, action_config, (center_x_abs, center_y_abs))

                        # Optional: break after first match per template or continue searching?
                        # For now, we execute action for the best match found on this monitor.
                        # If multiple instances are needed, the logic needs adjustment (e.g., find all matches above threshold).

            if not found_match_in_cycle:
                logger.info("No matches found in this cycle across all monitors.")

    except Exception as e:
        logger.error(f"An error occurred during the screen capture or matching process: {e}", exc_info=True)

    logger.info("Screen finder process finished cycle.")


if __name__ == "__main__":
    # This allows running the script directly for a single search cycle.
    # For timed loops, the GUI launcher or another script should call main() repeatedly.
    main()
