import json
import os
import cv2
import logging
from .config import TEMPLATES_DIR, TEMPLATES_CONFIG, DEFAULT_THRESHOLD

# Setup logger
logger = logging.getLogger(__name__)
# Basic logging configuration if not already configured by the main app
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_templates_config():
    """Loads the template configurations from the JSON file."""
    if not os.path.exists(TEMPLATES_CONFIG):
        logger.error(f"Templates configuration file not found: {TEMPLATES_CONFIG}")
        # Create a default empty config file if it doesn't exist
        default_config = {"templates": []}
        try:
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(TEMPLATES_CONFIG), exist_ok=True)
            with open(TEMPLATES_CONFIG, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default empty templates config file: {TEMPLATES_CONFIG}")
            return default_config
        except IOError as e:
            logger.error(f"Failed to create default config file: {e}")
            return None
    try:
        with open(TEMPLATES_CONFIG, 'r') as f:
            config = json.load(f)
            logger.info(f"Successfully loaded templates configuration from {TEMPLATES_CONFIG}")
            # Validate basic structure
            if "templates" not in config or not isinstance(config["templates"], list):
                logger.error(f"Invalid format in {TEMPLATES_CONFIG}. 'templates' key missing or not a list.")
                return None
            return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {TEMPLATES_CONFIG}: {e}")
        return None
    except IOError as e:
        logger.error(f"Error reading file {TEMPLATES_CONFIG}: {e}")
        return None

def preload_images(templates_config):
    """Preloads template images specified in the configuration."""
    loaded_templates = []
    if not templates_config or "templates" not in templates_config:
        logger.warning("No templates found in configuration or configuration is invalid.")
        return loaded_templates

    for template_info in templates_config["templates"]:
        name = template_info.get("name", "Unnamed Template")
        file_name = template_info.get("file")
        if not file_name:
            logger.warning(f"Template '{name}' is missing 'file' attribute. Skipping.")
            continue

        file_path = os.path.join(TEMPLATES_DIR, file_name)
        if not os.path.exists(file_path):
            logger.warning(f"Template image file not found for '{name}': {file_path}. Skipping.")
            continue

        try:
            # Load image in grayscale for template matching
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Failed to load template image '{file_name}' for '{name}'. Skipping.")
                continue

            # Get threshold, default if not specified
            threshold = template_info.get("threshold", DEFAULT_THRESHOLD)
            # Get action, default to None if not specified
            action = template_info.get("action")

            loaded_templates.append({
                "name": name,
                "image": image,
                "threshold": threshold,
                "action": action,
                "file_path": file_path # Store for reference if needed
            })
            logger.info(f"Successfully preloaded template '{name}' from {file_path}")

        except Exception as e:
            logger.error(f"Error processing template '{name}' from {file_path}: {e}")

    return loaded_templates

def load_and_preload_templates():
    """Loads configuration and preloads images."""
    config = load_templates_config()
    if config:
        return preload_images(config)
    else:
        return []

if __name__ == '__main__':
    # Example usage: Load templates when script is run directly
    print("Loading templates...")
    templates = load_and_preload_templates()
    if templates:
        print(f"Successfully loaded {len(templates)} templates:")
        for t in templates:
            print(f"  - Name: {t['name']}, Threshold: {t['threshold']}, Action: {t['action']}, Shape: {t['image'].shape}")
    else:
        print("No templates loaded or an error occurred.")