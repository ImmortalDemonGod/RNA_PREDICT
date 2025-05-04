# rna_predict/pipeline/stageA/run_stageA.py
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor

# Initialize logger for Stage A
logger = logging.getLogger("rna_predict.pipeline.stageA.run_stageA")


def download_file(url: str, dest_path: str, debug_logging: bool = False,
                max_retries: int = 3, initial_backoff: float = 1.0, backoff_factor: float = 2.0):
    """
    Download file from URL to a local destination path with exponential backoff retry.

    Args:
        url: URL to download from
        dest_path: Local path to save the file
        debug_logging: Whether to log debug information
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff time by after each attempt

    Raises:
        RuntimeError: If download fails after all retries
    """
    if os.path.isfile(dest_path):
        # If it's a .zip file, let's verify it's valid
        if dest_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(dest_path, "r") as zip_ref:
                    bad_file_test = zip_ref.testzip()
                    if bad_file_test is not None:
                        raise zipfile.BadZipFile(f"Corrupted member: {bad_file_test}")
            except zipfile.BadZipFile:
                if debug_logging:
                    logger.warning(f"[Warning] Existing .zip is invalid or corrupted. Re-downloading: {dest_path}")
                os.remove(dest_path)
                # Continue to the download section after removing the corrupt file
            else:
                # It's a valid zip
                if debug_logging:
                    logger.info(f"[Info] File already exists and is valid zip, skipping download: {dest_path}")
                return
        else:
            # For non-zip files, just skip if it exists
            if debug_logging:
                logger.info(f"[Info] File already exists, skipping download: {dest_path}")
            return

    # If we get here, we need to download the file (either it doesn't exist or was corrupt)
    if debug_logging:
        logger.info(f"[Download] Fetching {url}")

    # Implement exponential backoff retry
    backoff_time = initial_backoff
    last_exception = None

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(dest_path, "wb") as f:
                shutil.copyfileobj(r, f)

            if debug_logging:
                logger.info(f"[Download] Saved to {dest_path}")
            return  # Success, exit the function

        except Exception as exc:
            last_exception = exc
            if debug_logging:
                logger.warning(f"[DL] Download attempt {attempt+1}/{max_retries} failed: {exc}")

            # Don't sleep after the last attempt
            if attempt < max_retries - 1:
                if debug_logging:
                    logger.info(f"[DL] Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
                backoff_time *= backoff_factor  # Increase backoff time for next attempt

    # If we get here, all retries failed
    if debug_logging:
        logger.error(f"[DL] All {max_retries} download attempts failed for {url}")

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts") from last_exception


def unzip_file(zip_path: str, extract_dir: str, debug_logging: bool = False):
    """
    Unzip the zip_path into extract_dir, overwriting existing files,
    using Python's built-in zipfile module so that 'unzip' command
    is not required.
    """
    if not os.path.isfile(zip_path):
        if debug_logging:
            logger.warning(f"[Warning] Zip file not found: {zip_path}")
        return
    if debug_logging:
        logger.info(f"[Unzip] Extracting {zip_path} into {extract_dir}")

    # ensure the directory exists
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def visualize_with_varna(ct_file: str, jar_path: str, output_png: str, resolution: float = 8.0, debug_logging: bool = False):
    """
    Small helper function to call the VARNA .jar to generate RNA secondary structure images.
    Requires Java on the system path and the jar at jar_path.
    """
    if not os.path.isfile(ct_file):
        if debug_logging:
            logger.warning(f"[Warning] CT file not found: {ct_file}")
        return
    if not os.path.isfile(jar_path):
        if debug_logging:
            logger.warning(f"[Warning] VARNA JAR not found at: {jar_path} -> skipping visualization.")
        return

    cmd = [
        "java",
        "-cp",
        jar_path,
        "fr.orsay.lri.varna.applications.VARNAcmd",
        "-i",
        ct_file,
        "-o",
        output_png,
        "-resolution",
        str(resolution), # Use resolution parameter
    ]
    if debug_logging:
        logger.info(f"[VARNA] Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if debug_logging:
        logger.info(f"[VARNA] Visualization saved to {output_png}")
    return stdout, stderr


# Updated config_path to use the proper module path for the conf directory
# When running as a module, we need to use a relative path from the module location
@hydra.main(version_base=None, config_path="../../../rna_predict/conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # With our fixed configuration structure, we can directly access the stageA config
    stage_cfg = cfg.model.stageA

    debug_logging = False
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageA') and hasattr(cfg.model.stageA, 'debug_logging'):
        debug_logging = cfg.model.stageA.debug_logging

    if debug_logging:
        logger.info("Starting Stage A pipeline with Hydra config")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        logger.debug("[UNIQUE-DEBUG-STAGEA-TEST] This should always appear if logger is working.")

    # Print the full config for debugging
    if debug_logging:
        logger.info(f"[Hydra Config] Full config:\n{OmegaConf.to_yaml(cfg)}")

    if debug_logging:
        logger.info(f"[Hydra Config] Loaded Stage A config:\n{stage_cfg}")

    # 1) Prepare environment (Checkpoint download/unzip with configurable paths)
    # Extract paths from config
    checkpoint_dir = os.path.dirname(stage_cfg.checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Extract checkpoint URL and zip path from config (using direct attribute access)
    checkpoint_url = stage_cfg.checkpoint_url
    checkpoint_zip = stage_cfg.checkpoint_zip_path
    download_file(checkpoint_url, checkpoint_zip, debug_logging)
    unzip_file(checkpoint_zip, os.path.dirname(checkpoint_dir), debug_logging)

    # 2) Build the predictor using Hydra config
    # Get device from config
    device_str = stage_cfg.device.lower()

    # Print the resolved device for StageA for systematic Hydra compliance debugging
    logger.info(f"[HYDRA-DEBUG][StageA] Resolved stage_cfg.device: {stage_cfg.device}")
    logger.info(f"[HYDRA-DEBUG][StageA] Global cfg.device: {cfg.device}")

    # Create the device object (device validation will be handled in the predictor)
    device = torch.device(device_str)
    if debug_logging:
        logger.info(f"[Device] Using device: {device}")

    # Instantiate predictor directly using config values
    # Assumes StageARFoldPredictor's __init__ signature matches the config keys + device
    # (Will be updated in the next subtask)
    # Pass the whole stage config object and the determined device
    predictor = StageARFoldPredictor(stage_cfg=stage_cfg, device=device)

    # --- ENFORCE: Assert that dummy_mode is not active for training ---
    if hasattr(cfg, 'train') and cfg.train.get('enabled', False):
        assert not getattr(predictor, 'dummy_mode', False), (
            "[STAGEA-TRAIN-ERROR] Dummy model is active! Training cannot proceed. "
            "Check your Hydra config for missing or incomplete fields."
        )

    # Predictor is now built

    # 4) Example inference (conditional based on config)
    if stage_cfg.run_example:
        # Use standardized test sequence from test_data config if available
        if hasattr(cfg, 'test_data') and hasattr(cfg.test_data, 'sequence'):
            sequence = cfg.test_data.sequence
            if debug_logging:
                logger.info(f"[Example] Running inference on standardized test sequence: {sequence} (length: {len(sequence)})")
        else:
            # Fall back to example sequence from stageA config
            sequence = stage_cfg.example_sequence
            if debug_logging:
                logger.info(f"[Example] Running inference on example sequence from config (length: {len(sequence)})")

        adjacency = predictor.predict_adjacency(sequence)
        if debug_logging:
            logger.info(f"[Example] Adjacency shape: {adjacency.shape}")
    else:
        if debug_logging:
            logger.info("[Example] Skipping example inference (disabled in config)")
        adjacency = None

    # 5) (Optional) Visualization - only if example was run and visualization is enabled
    if stage_cfg.run_example and adjacency is not None:
        # For demo, let's mock writing a .ct file:
        mock_ct_file = "test_seq.ct"
        with open(mock_ct_file, "w") as f:
            f.write(">TestSeq\n")
            f.write("1  A  0  2  0  1\n")
            f.write("2  A  1  3  0  2\n")

        # Check if visualization is enabled in the config
        if hasattr(stage_cfg, 'visualization') and stage_cfg.visualization.enabled:
            # Access visualization parameters directly from the structured config
            varna_jar_path = stage_cfg.visualization.varna_jar_path
            output_image_path = stage_cfg.visualization.output_path
            resolution = stage_cfg.visualization.resolution

            if debug_logging:
                logger.info(f"[Visualization] Attempting with JAR: {varna_jar_path}")
            visualize_with_varna(
                ct_file=mock_ct_file,
                jar_path=varna_jar_path,
                output_png=output_image_path,
                resolution=resolution,
                debug_logging=debug_logging
            )
        else:
            if debug_logging:
                logger.info("[Info] VARNA visualization disabled via config.")
    elif not stage_cfg.run_example:
        if debug_logging:
            logger.info("[Info] Skipping visualization since example inference was not run.")
    else:
        if debug_logging:
            logger.info("[Warning] Cannot visualize: example inference failed to produce adjacency.")



def run_stageA(seq, predictor):
    """
    Simple helper function to integrate with tests.
    Returns adjacency matrix from the given predictor.
    """
    return predictor.predict_adjacency(seq)


if __name__ == "__main__":
    main()
