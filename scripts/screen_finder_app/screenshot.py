# rna_predict/scripts/screenshot.py
import logging  # Import logging to access levels like INFO, DEBUG
from typing import Any, Dict, List

import cv2  # OpenCV is used for converting the image format
import mss
import numpy as np

from .logger import (
    setup_logger,
)  # Use relative import within the same package/directory structure

# Setup logger for this module
logger = setup_logger(logging.INFO)


def capture_all_monitors() -> List[Dict[str, Any]]:
    """
    Captures screenshots of all individual monitors using the mss library.

    :return: A list of dictionaries. Each dictionary contains:
             - 'monitor_info': The dictionary provided by mss for the monitor.
             - 'image': The screenshot as a NumPy array in BGR format (compatible with OpenCV).
             Returns an empty list if the capture fails or no monitors are found.
    """
    screenshots = []
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            logger.debug(
                f"Found {len(monitors) - 1} individual monitors (excluding the combined virtual screen)."
            )

            # The first monitor in sct.monitors is the combined virtual screen, skip it.
            for i, monitor in enumerate(monitors[1:], start=1):
                logger.debug(
                    f"Attempting to capture screenshot for monitor {i}: {monitor}"
                )
                # Capture the screen for the current monitor
                img = sct.grab(monitor)

                # Convert the captured BGRA image to a BGR NumPy array
                image_np_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

                screenshots.append({"monitor_info": monitor, "image": image_np_bgr})
                logger.info(f"Screenshot captured successfully for monitor {i}.")

    except Exception as e:
        logger.error(f"Failed to capture screenshots: {e}", exc_info=True)
        return []  # Return empty list on failure

    return screenshots


# Example usage (optional, can be uncommented for testing)
# if __name__ == '__main__':
#     captured_images = capture_all_monitors()
#     if captured_images:
#         logger.info(f"Successfully captured {len(captured_images)} monitor(s).")
#         # Example: Display the first captured screenshot
#         # cv2.imshow(f"Monitor 1 Screenshot", captured_images[0]['image'])
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#     else:
#         logger.error("Failed to capture any screenshots for testing.")
