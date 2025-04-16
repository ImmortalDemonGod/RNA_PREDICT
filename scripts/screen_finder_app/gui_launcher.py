import dearpygui.dearpygui as dpg
import logging
import threading
import time
import cv2
import numpy as np
from mss import mss

# Import necessary components from other modules
from .config import LOG_FILE, LOG_LEVEL # Corrected imports
from .template_loader import load_and_preload_templates
from .main import execute_action # Import the action execution logic

# --- Global Variables ---
search_running: bool = False
search_thread: threading.Thread | None = None
status_message: str = "Status: Idle"
loaded_template_names: list[str] = [] # To display in the GUI

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL) # Use LOG_LEVEL from config

# File handler (use the same log file as main.py)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# --- Core Search Logic (Threaded) ---
def periodic_search_thread():
    global search_running, status_message, loaded_template_names
    logger.info("Periodic search thread started.")

    # Load templates once when the thread starts
    templates = load_and_preload_templates()
    if not templates:
        status_message = "Status: Error - No templates loaded. Stopping."
        logger.error("No templates loaded in search thread.")
        search_running = False # Stop if templates fail to load
        # Update GUI status from the main thread later if needed, direct DPG calls from threads are risky
        return

    # Update template names for GUI display (schedule in main thread)
    loaded_template_names = [t['name'] for t in templates]
    # Consider using dpg.set_value for thread-safe GUI updates if directly updating list items

    logger.info(f"Periodic search active with {len(templates)} templates.")

    with mss() as sct:
        while search_running:
            start_time = time.time()
            interval = dpg.get_value("-INTERVAL-") # Get interval from slider
            status_message = f"Status: Running - Interval: {interval:.1f}s. Searching..."
            logger.debug(f"Starting search cycle. Interval: {interval}s")
            dpg.set_value("-STATUS-", status_message) # Update status at start of cycle

            found_match_in_cycle = False
            matches_found_this_cycle = []

            try:
                monitors = sct.monitors[1:] # Exclude the combined virtual screen

                for i, monitor in enumerate(monitors):
                    sct_img = sct.grab(monitor)
                    screen_img_bgr = np.array(sct_img)
                    screen_img_gray = cv2.cvtColor(screen_img_bgr, cv2.COLOR_BGRA2GRAY)

                    for template_data in templates:
                        if not search_running:
                            break

                        template_name = template_data["name"]
                        template_img = template_data["image"]
                        threshold = template_data["threshold"]
                        action_config = template_data["action"]
                        tH, tW = template_img.shape[:2]

                        result = cv2.matchTemplate(screen_img_gray, template_img, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                        if max_val >= threshold:
                            found_match_in_cycle = True
                            match_x, match_y = max_loc
                            center_x_rel = match_x + tW // 2
                            center_y_rel = match_y + tH // 2
                            center_x_abs = monitor["left"] + center_x_rel
                            center_y_abs = monitor["top"] + center_y_rel

                            match_info = f"Found '{template_name}' (Conf: {max_val:.2f}) at ({center_x_abs}, {center_y_abs})"
                            logger.info(match_info)
                            matches_found_this_cycle.append(match_info)

                            # Execute action
                            execute_action(logger, action_config, (center_x_abs, center_y_abs))

                            # Optional: break here if only one action per cycle is desired
                            # break # Uncomment to stop searching after the first match in the cycle

                    if not search_running:
                        break  # Exit monitor loop if stopped
                if not search_running:
                    break  # Exit main loop if stopped

                # Update status after cycle completes
                cycle_duration = time.time() - start_time
                if found_match_in_cycle:
                    status_message = f"Status: Found {len(matches_found_this_cycle)} match(es). Cycle took {cycle_duration:.2f}s. Waiting..."
                    # Optionally display which templates were found in status
                    # status_message += " Matches: " + ", ".join([m.split(' ')[1] for m in matches_found_this_cycle])
                else:
                    status_message = f"Status: No matches found. Cycle took {cycle_duration:.2f}s. Waiting..."

                logger.debug(f"Search cycle finished in {cycle_duration:.2f} seconds.")
                dpg.set_value("-STATUS-", status_message)

                # Wait for the next interval
                sleep_time = max(0, interval - cycle_duration)
                logger.debug(f"Sleeping for {sleep_time:.2f} seconds.")
                # Use a loop for sleeping to check the flag more often
                for _ in range(int(sleep_time * 10)): # Check every 100ms
                    if not search_running:
                        break
                    time.sleep(0.1)
                if not search_running:
                    break  # Exit loop if stopped during sleep


            except Exception as e:
                status_message = f"Status: Error occurred: {e}"
                logger.error(f"Error during periodic search: {e}", exc_info=True)
                dpg.set_value("-STATUS-", status_message)
                search_running = False # Stop on error

    logger.info("Periodic search thread finished.")
    status_message = "Status: Stopped."
    dpg.set_value("-STATUS-", status_message) # Final status update


# --- GUI Callbacks ---
def start_periodic_search():
    global search_running, search_thread, status_message
    if search_running:
        logger.warning("Search is already running.")
        return

    search_running = True
    status_message = "Status: Starting..."
    dpg.set_value("-STATUS-", status_message)
    logger.info("Start button clicked.")
    # Disable start button, enable stop button
    dpg.configure_item("-START_BUTTON-", enabled=False)
    dpg.configure_item("-STOP_BUTTON-", enabled=True)

    # Start the search logic in a separate thread
    search_thread = threading.Thread(target=periodic_search_thread, daemon=True)
    search_thread.start()

def stop_periodic_search():
    global search_running, search_thread, status_message
    if not search_running:
        logger.warning("Search is not running.")
        return

    logger.info("Stop button clicked.")
    status_message = "Status: Stopping..."
    dpg.set_value("-STATUS-", status_message)
    search_running = False

    # Wait briefly for the thread to notice the flag (optional)
    if search_thread and search_thread.is_alive():
         search_thread.join(timeout=1.0) # Wait max 1 sec

    # Enable start button, disable stop button
    dpg.configure_item("-START_BUTTON-", enabled=True)
    dpg.configure_item("-STOP_BUTTON-", enabled=False)
    status_message = "Status: Stopped." # Ensure final status is set
    dpg.set_value("-STATUS-", status_message)
    logger.info("Search stopped.")

def update_template_list():
    """Updates the listbox with currently loaded template names."""
    # Clear existing items
    if dpg.does_item_exist("-TEMPLATE_LIST-"):
         # Need a way to clear listbox items if DPG doesn't have a direct clear function
         # For now, we just set the items, replacing the old ones.
         dpg.configure_item("-TEMPLATE_LIST-", items=loaded_template_names)
    else:
         logger.warning("Template listbox item not found.")


# --- GUI Setup ---
def setup_gui():
    dpg.create_context()

    # Load initial templates to populate list (optional, could also load on start)
    global loaded_template_names
    try:
        initial_templates = load_and_preload_templates()
        loaded_template_names = [t['name'] for t in initial_templates]
    except Exception as e:
        logger.error(f"Failed to preload templates for GUI list: {e}")
        loaded_template_names = ["Error loading templates"]


    with dpg.window(label="Screen Finder Control", width=500, height=400, tag="-PRIMARY_WINDOW-"):
        dpg.add_text("Configure and run timed template searches.")
        dpg.add_separator()

        # Interval Slider
        dpg.add_text("Search Interval (seconds):")
        dpg.add_slider_float(label="Interval", tag="-INTERVAL-", default_value=5.0, min_value=0.5, max_value=60.0, format="%.1f s")
        dpg.add_separator()

        # Control Buttons
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Timed Search", tag="-START_BUTTON-", callback=start_periodic_search)
            dpg.add_button(label="Stop Timed Search", tag="-STOP_BUTTON-", callback=stop_periodic_search, enabled=False) # Initially disabled
        dpg.add_separator()

        # Status Display
        dpg.add_text(status_message, tag="-STATUS-")
        dpg.add_separator()

        # Loaded Templates List (Read-only display)
        dpg.add_text("Loaded Templates:")
        dpg.add_listbox(items=loaded_template_names, tag="-TEMPLATE_LIST-", num_items=8)
        # Add a button to refresh this list if needed, or update it via callbacks/threads carefully
        # dpg.add_button(label="Refresh List", callback=update_template_list)


    dpg.create_viewport(title='Screen Finder GUI', width=520, height=440)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Set primary window after setup
    dpg.set_primary_window("-PRIMARY_WINDOW-", True)

    # Main GUI loop (check search status for updates)
    # While loop is better handled by dpg.start_dearpygui()
    # Need a way to update status from thread: use dpg.set_value called from thread or use a queue

    # dpg.start_dearpygui() # This blocks, run rendering loop manually if needed or use callbacks

    while dpg.is_dearpygui_running():
        # Manual update of status text (simpler than complex thread-safe calls for this case)
        # This might miss rapid updates from the thread but avoids direct DPG calls from bg thread
        # A better approach involves dpg.mvCaptureContext() or thread-safe queues if updates are critical
        current_status = dpg.get_value("-STATUS-")
        if current_status != status_message:
             dpg.set_value("-STATUS-", status_message)

        # Update template list if names change (e.g., after loading)
        # This check is basic; might need refinement
        current_list_items = dpg.get_item_configuration("-TEMPLATE_LIST-")["items"]
        if current_list_items != loaded_template_names:
             dpg.configure_item("-TEMPLATE_LIST-", items=loaded_template_names)


        dpg.render_dearpygui_frame()

    # Cleanup
    global search_running
    if search_running:
        logger.info("GUI closing, stopping search thread...")
        search_running = False
        if search_thread and search_thread.is_alive():
            search_thread.join(timeout=1.0)
    dpg.destroy_context()
    logger.info("GUI closed and context destroyed.")

if __name__ == "__main__":
    setup_gui()
