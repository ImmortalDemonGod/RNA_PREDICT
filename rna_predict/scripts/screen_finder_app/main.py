# rna_predict/scripts/main.py
import logging

from .config import LOGGING_LEVEL, TEMPLATE_PATH, THRESHOLD

# Import local modules using relative paths
from .logger import setup_logger

# from .region_selector import select_region # Keep commented out
from .screenshot import capture_all_monitors  # Updated import
from .template_matching import validate_and_match_template


def find_template_on_screen(logger, template_path, threshold):
    """
    Captures screenshots of all monitors and searches for a template image.

    Args:
        logger: The configured logger instance.
        template_path (str): Path to the template image file.
        threshold (float): The matching threshold for template detection.

    Returns:
        dict: A dictionary containing the result.
              If found: {'found': True, 'coordinates': (abs_x, abs_y), 'monitor_info': monitor_info, 'correlation': max_val}
              If not found: {'found': False}
    """
    logger.info("Attempting to find template on screen...")
    logger.info("Capturing screenshots for all monitors.")
    monitor_screenshots = capture_all_monitors()

    if not monitor_screenshots:
        logger.error("Failed to capture screenshots from any monitor.")
        return {"found": False}

    logger.info(f"Captured screenshots from {len(monitor_screenshots)} monitor(s).")

    for i, monitor_data in enumerate(monitor_screenshots):
        monitor_info = monitor_data["monitor_info"]
        screenshot_img = monitor_data["image"]
        monitor_dims_str = f"Monitor {i + 1} (Top: {monitor_info['top']}, Left: {monitor_info['left']}, W: {monitor_info['width']}, H: {monitor_info['height']})"

        logger.info(f"Processing {monitor_dims_str}")

        # Optionally save each monitor's screenshot for debugging
        # save_path = f"captured_monitor_{i+1}.png"
        # cv2.imwrite(save_path, screenshot_img)
        # logger.debug(f"Saved screenshot for {monitor_dims_str} to {save_path}")

        logger.info(
            f"Performing template matching on {monitor_dims_str} using template: '{template_path}' with threshold: {threshold}"
        )
        # validate_and_match_template now returns a tuple: (location, correlation_score) or None
        match_result = validate_and_match_template(
            screenshot=screenshot_img, template_path=template_path, threshold=threshold
        )

        if match_result:
            match_location_relative, correlation_score = (
                match_result  # Unpack the result
            )
            # Calculate absolute coordinates by adding monitor offset
            abs_x = monitor_info["left"] + match_location_relative[0]
            abs_y = monitor_info["top"] + match_location_relative[1]
            absolute_match_location = (abs_x, abs_y)
            logger.info(
                f"Template found on {monitor_dims_str} at relative coordinates {match_location_relative}, absolute coordinates {absolute_match_location} with correlation {correlation_score:.4f}"
            )
            # Return the full monitor_info dictionary along with coordinates and correlation
            return {
                "found": True,
                "coordinates": absolute_match_location,
                "monitor_info": monitor_info,
                "correlation": correlation_score,
            }

    logger.info("Template not found on any monitor within the specified threshold.")
    return {"found": False}


def main():
    """
    Main function to set up logging and initiate the template finding process.
    """
    # Convert LOGGING_LEVEL string from config to logging level constant
    log_level_str = LOGGING_LEVEL.upper()
    log_level = getattr(
        logging, log_level_str, logging.INFO
    )  # Default to INFO if invalid
    logger = setup_logger(level=log_level)

    logger.info("Application started.")
    logger.debug(
        f"Configuration: TEMPLATE_PATH='{TEMPLATE_PATH}', THRESHOLD={THRESHOLD}, LOGGING_LEVEL='{log_level_str}'"
    )

    # Call the refactored function
    result = find_template_on_screen(logger, TEMPLATE_PATH, THRESHOLD)

    # Process the result from the function
    if result["found"]:
        coords = result["coordinates"]
        monitor_info = result["monitor_info"]
        monitor_dims_str = f"Monitor (Top: {monitor_info['top']}, Left: {monitor_info['left']}, W: {monitor_info['width']}, H: {monitor_info['height']})"  # Reconstruct description if needed
        logger.info(
            f"Template successfully found on {monitor_dims_str} at absolute screen coordinates: {coords}"
        )
        print(
            f"Success: Template found on {monitor_dims_str} at absolute screen coordinates: {coords}"
        )
    else:
        logger.info("Template not found on any monitor.")
        print("Result: Template not found on any screen.")

    logger.info("Application finished.")


if __name__ == "__main__":
    # This block allows the script to be run directly for testing.
    main()
