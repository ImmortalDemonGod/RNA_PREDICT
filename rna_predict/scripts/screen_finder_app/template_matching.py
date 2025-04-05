# rna_predict/scripts/template_matching.py
import cv2 # type: ignore
import numpy as np
from typing import Optional, Tuple, Dict # Added Dict for potential future use, Tuple is primary
from .logger import setup_logger # Relative import
import logging

# Setup logger for this module
logger = setup_logger(logging.INFO)

def validate_and_match_template(
    screenshot: np.ndarray,
    template_path: str,
    threshold: float
) -> Optional[Tuple[Tuple[int, int], float]]: # Updated return type hint
    """
    Validates the template image, performs template matching on the screenshot,
    and returns the location and correlation score if the match exceeds the threshold.

    :param screenshot: The screenshot image (NumPy array in BGR format) to search within.
    :param template_path: Path to the template image file (BGR format).
    :param threshold: The correlation threshold (0.0 to 1.0) for a successful match.
    :return: A tuple containing ((x, y), correlation_score) representing the top-left
             coordinates and the match correlation, or None if no match is found
             above the threshold or if the template cannot be loaded.
    """
    logger.debug(f"Attempting to load template from: {template_path}")
    # Load the template image
    template = cv2.imread(template_path) # Reads in BGR format by default

    # Validate if the template image was loaded successfully
    if template is None:
        logger.error(f"Template image '{template_path}' could not be loaded. Check the path and file integrity.")
        return None
    logger.info(f"Template image '{template_path}' loaded successfully.")

    # Validate screenshot dimensions against template dimensions
    if screenshot.shape[0] < template.shape[0] or screenshot.shape[1] < template.shape[1]:
        logger.error(f"Screenshot dimensions ({screenshot.shape[:2]}) are smaller than template dimensions ({template.shape[:2]}). Matching is not possible.")
        return None

    logger.debug(f"Performing template matching with threshold: {threshold}")
    # Perform template matching using Normalized Cross-Correlation Coefficient
    # TM_CCOEFF_NORMED is generally robust to lighting changes
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

    # Find the maximum correlation value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    logger.debug(f"Template matching result: Max correlation = {max_val:.4f} at location {max_loc}")

    # Check if the maximum correlation value meets the threshold
    if max_val >= threshold:
        # max_loc gives the top-left corner of the matched area
        match_location = max_loc
        # Explicitly cast to Tuple[int, int] and float for type checker clarity
        match_location_tuple: Tuple[int, int] = (int(match_location[0]), int(match_location[1]))
        correlation_score: float = float(max_val)
        logger.info(f"Template match found with correlation {correlation_score:.4f} >= threshold {threshold} at {match_location_tuple}")
        # Return the explicitly typed tuple
        return match_location_tuple, correlation_score
    else:
        logger.info(f"No template match found above threshold {threshold}. Max correlation was {max_val:.4f}")
        return None

# Example usage (optional)
if __name__ == '__main__':
    # This requires a sample screenshot and template image to run
    # Create dummy images for basic testing if needed
    logger.info("Running template_matching.py example.")

    # Create a dummy screenshot (e.g., 500x500 gray image)
    dummy_screenshot = np.zeros((500, 500, 3), dtype=np.uint8) + 128 # Gray background
    # Create a dummy template (e.g., 50x50 white square)
    dummy_template_img = np.zeros((50, 50, 3), dtype=np.uint8) + 255 # White square
    # Place the template somewhere in the screenshot
    dummy_screenshot[100:150, 200:250] = dummy_template_img
    cv2.imwrite("dummy_screenshot.png", dummy_screenshot)
    cv2.imwrite("dummy_template.png", dummy_template_img)

    logger.info("Dummy screenshot and template created for testing.")

    # Define parameters for the test
    test_template_path = 'dummy_template.png'
    test_threshold = 0.9 # High threshold for exact match

    # Perform matching
    location = validate_and_match_template(dummy_screenshot, test_template_path, test_threshold)

    if location:
        logger.info(f"Test successful: Template found at {location}")
        # Expected location: (200, 100)
        assert location == (200, 100), f"Expected (200, 100), got {location}"
    else:
        logger.error("Test failed: Template not found.")

    # Clean up dummy files (optional)
    # import os
    # os.remove("dummy_screenshot.png")
    # os.remove("dummy_template.png")
    # logger.info("Cleaned up dummy files.")