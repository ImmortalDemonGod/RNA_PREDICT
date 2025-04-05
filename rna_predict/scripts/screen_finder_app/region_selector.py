# rna_predict/scripts/region_selector.py
import tkinter as tk
from PIL import ImageGrab # Used for getting screen dimensions reliably
from .logger import setup_logger # Relative import
import logging
from typing import Optional # Added for type hinting

# Setup logger for this module
logger = setup_logger(logging.INFO)

# Global variables to store coordinates and GUI elements
start_x, start_y = None, None
end_x, end_y = None, None
rect = None
region_coords = None # Stores the final [x, y, width, height]
root = None
canvas = None

def on_button_press(event):
    """Callback for mouse button press event."""
    global start_x, start_y, rect, canvas
    start_x, start_y = event.x, event.y
    # Create rectangle if it doesn't exist, else update coords
    if rect:
        canvas.coords(rect, start_x, start_y, start_x, start_y)
    else:
        # Dashed red rectangle for selection visualization
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y,
                                       outline='red', width=2, dash=(4, 2))
    logger.info(f"Selection started at screen coordinates ({event.x_root}, {event.y_root}), canvas coordinates ({start_x}, {start_y})")

def on_move(event):
    """Callback for mouse move event (while button is pressed)."""
    global rect, canvas, start_x, start_y
    current_x, current_y = event.x, event.y
    # Update the rectangle size as the mouse moves
    if rect:
        canvas.coords(rect, start_x, start_y, current_x, current_y)
    # logger.debug(f"Selection moving to ({event.x_root}, {event.y_root})") # Optional: Log move events

def on_button_release(event):
    """Callback for mouse button release event."""
    global end_x, end_y, region_coords, root, start_x, start_y
    end_x, end_y = event.x, event.y

    # Ensure width and height are positive
    width = abs(end_x - start_x)
    height = abs(end_y - start_y)

    # Calculate top-left corner coordinates (absolute screen coordinates)
    # event.x_root/y_root are screen coords, event.x/y are canvas coords
    # We need the top-left corner relative to the screen.
    # The canvas covers the whole screen, so canvas coords map directly if window is fullscreen borderless.
    # Let's use the canvas coordinates and assume they map 1:1 to screen pixels for simplicity here.
    # A more robust solution might involve mapping canvas coords to screen coords carefully.
    final_x = min(start_x, end_x)
    final_y = min(start_y, end_y)

    region_coords = [final_x, final_y, width, height]

    logger.info(f"Selection completed. Canvas coords: TopLeft=({final_x}, {final_y}), W={width}, H={height}")
    logger.info(f"Approx Screen coords (assuming 1:1 mapping): {region_coords}")

    # Close the Tkinter window
    if root:
        root.quit() # Use quit to break the mainloop
        root.destroy() # Then destroy the window

def select_region() -> Optional[list]:
    """
    Displays a fullscreen transparent window to select a screen region.

    :return: A list [x, y, width, height] of the selected region in screen coordinates,
             or None if selection is cancelled or invalid.
    """
    global root, canvas, region_coords, start_x, start_y, end_x, end_y, rect
    # Reset state variables
    start_x, start_y = None, None
    end_x, end_y = None, None
    rect = None
    region_coords = None

    logger.info("Initializing region selection GUI.")
    root = tk.Tk()
    # Make window borderless and cover the entire screen
    root.attributes('-fullscreen', True)
    # Make window transparent (may depend on OS and window manager)
    # Alpha transparency: 0.0 (fully transparent) to 1.0 (fully opaque)
    root.attributes('-alpha', 0.3) # Slightly visible for context
    root.wait_visibility(root) # Ensure window is visible before making it topmost
    root.attributes('-topmost', True) # Keep window on top

    # Use PIL ImageGrab to get screen dimensions for canvas size
    try:
        screen_width, screen_height = ImageGrab.grab().size
        logger.debug(f"Detected screen dimensions: {screen_width}x{screen_height}")
    except Exception as e:
        logger.error(f"Could not get screen dimensions using PIL: {e}. Falling back to Tkinter info.")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        logger.warning(f"Using Tkinter screen dimensions: {screen_width}x{screen_height}")


    canvas = tk.Canvas(root, width=screen_width, height=screen_height,
                       cursor="cross", bg='white', highlightthickness=0)
    # Set canvas background to be transparent requires specific handling per OS
    # For simplicity, using a semi-transparent window alpha instead.
    canvas.pack()

    # Bind mouse events
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    # Add instructions label
    label = tk.Label(root, text="Click and drag to select a region. Release to confirm.",
                     bg="yellow", fg="black", font=("Arial", 12))
    # Position label at the top center
    label.place(relx=0.5, rely=0.02, anchor=tk.CENTER)


    logger.info("Region selection window displayed. Waiting for user input.")
    root.mainloop() # Start the Tkinter event loop

    # After mainloop finishes (window closed), return the coordinates
    logger.info(f"Region selection finished. Returning coordinates: {region_coords}")
    # Validate coordinates (e.g., width and height > 0)
    if region_coords and region_coords[2] > 0 and region_coords[3] > 0:
        return region_coords
    else:
        logger.warning("Invalid or cancelled region selection.")
        return None

# Example usage (optional)
if __name__ == '__main__':
    logger.info("Running region_selector.py example.")
    selected_area = select_region()
    if selected_area:
        print(f"Selected Region Coordinates: x={selected_area[0]}, y={selected_area[1]}, width={selected_area[2]}, height={selected_area[3]}")
        logger.info(f"Example usage successful: Region selected {selected_area}")
    else:
        print("No region was selected or selection was invalid.")
        logger.info("Example usage: No valid region selected.")