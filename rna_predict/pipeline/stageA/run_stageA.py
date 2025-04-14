# rna_predict/pipeline/run_stageA.py
import os
import shutil
import subprocess
import urllib.request
import zipfile # Moved import here for clarity

import torch
import hydra
from omegaconf import DictConfig

# Assuming the predictor path is correct relative to project root
from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor


def download_file(url: str, dest_path: str):
    """
    Download file from URL to a local destination path.
    If the file already exists, check if it's a valid zip (when extension is .zip).
    If invalid, remove and re-download; otherwise skip download.
    """
    # import zipfile # Moved to top

    if os.path.isfile(dest_path):
        # If it's a .zip file, let's verify it's valid
        if dest_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(dest_path, "r") as zip_ref:
                    bad_file_test = zip_ref.testzip()
                    if bad_file_test is not None:
                        raise zipfile.BadZipFile(f"Corrupted member: {bad_file_test}")
            except zipfile.BadZipFile:
                print(
                    f"[Warning] Existing .zip is invalid or corrupted. Re-downloading: {dest_path}"
                )
                os.remove(dest_path)
                # Continue to the download section after removing the corrupt file
            else:
                # It's a valid zip
                print(
                    f"[Info] File already exists and is valid zip, skipping download: {dest_path}"
                )
                return
        else:
            # For non-zip files, just skip if it exists
            print(f"[Info] File already exists, skipping download: {dest_path}")
            return

    # If we get here, we need to download the file (either it doesn't exist or was corrupt)
    print(f"[Download] Fetching {url}")
    with urllib.request.urlopen(url) as r, open(dest_path, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[Download] Saved to {dest_path}")


def unzip_file(zip_path: str, extract_dir: str):
    """
    Unzip the zip_path into extract_dir, overwriting existing files,
    using Python's built-in zipfile module so that 'unzip' command
    is not required.
    """
    # import zipfile # Already imported at top

    if not os.path.isfile(zip_path):
        print(f"[Warning] Zip file not found: {zip_path}")
        return
    print(f"[Unzip] Extracting {zip_path} into {extract_dir}")

    # ensure the directory exists
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def visualize_with_varna(ct_file: str, jar_path: str, output_png: str, resolution: float = 8.0):
    """
    Small helper function to call the VARNA .jar to generate RNA secondary structure images.
    Requires Java on the system path and the jar at jar_path.
    """
    if not os.path.isfile(ct_file):
        print(f"[Warning] CT file not found: {ct_file}")
        return
    if not os.path.isfile(jar_path):
        print(
            f"[Warning] VARNA JAR not found at: {jar_path} -> skipping visualization."
        )
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
    print(f"[VARNA] Running: {' '.join(cmd)}")
    subprocess.Popen(
        cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    ).communicate()[0]
    print(f"[VARNA] Visualization saved to {output_png}")


# Updated config_path to use the proper module path for the conf directory
# When running as a module, we need to use a relative path from the module location
@hydra.main(version_base=None, config_path="../../../rna_predict/conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Extract the stageA config from the nested structure
    # The config could be nested under 'model' or 'stageA' depending on how it's loaded
    if 'model' in cfg and isinstance(cfg.model, DictConfig):
        stage_cfg = cfg.model
    elif 'stageA' in cfg and isinstance(cfg.stageA, DictConfig):
        stage_cfg = cfg.stageA
    else:
        # Fallback to using the config directly if the structure is different
        stage_cfg = cfg

    print(f"[Hydra Config] Loaded Stage A config:\n{stage_cfg}")

    # 1) Prepare environment (Checkpoint download/unzip remains for now)
    # Consider moving URL/paths fully into config later
    os.makedirs("RFold", exist_ok=True)
    os.makedirs("RFold/checkpoints", exist_ok=True)

    # Extract checkpoint URL from config, with a fallback
    checkpoint_url = stage_cfg.get('checkpoint_url', "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1")
    # Clean up the URL if it contains markdown formatting
    if '[' in checkpoint_url and '](' in checkpoint_url:
        # Extract the URL from markdown format [text](url)
        checkpoint_url = checkpoint_url.split('](')[1].rstrip(')')

    checkpoint_zip = "RFold/checkpoints.zip" # Keep local zip path for now
    # TODO: Consider making checkpoint_zip path also configurable if needed
    download_file(checkpoint_url, checkpoint_zip)
    unzip_file(checkpoint_zip, "RFold")

    # 2) Build the predictor using Hydra config
    # Determine device based on config string, with fallback for CUDA availability
    device_str = stage_cfg.get('device', 'cpu').lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA specified but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[Device] Using device: {device}")

    # Instantiate predictor directly using config values
    # Assumes StageARFoldPredictor's __init__ signature matches the config keys + device
    # (Will be updated in the next subtask)
    # Pass the whole stage config object and the determined device
    predictor = StageARFoldPredictor(stage_cfg=stage_cfg, device=device)

    # Predictor is now built

    # 4) Example inference
    sequence = "AAGUCUGGUGGACAUUGGCGUCCUGAGGUGUUAAAACCUCUUAUUGCUGACGCCAGAAAGAGAAGAACUUCGGUUCUACUAGUCGACUAUACUACAAGCUUUGGGUGUAUAGCGGCAAGACAACCUGGAUCGGGGGAGGCUAAGGGCGCAAGCCUAUGCUAACCCCGAGCCGAGCUACUGGAGGGCAACCCCCAGAUAGCCGGUGUAGAGCGCGGAAAGGUGUCGGUCAUCCUAUCUGAUAGGUGGCUUGAGGGACGUGCCGUCUCACCCGAAAGGGUGUUUCUAAGGAGGAGCUCCCAAAGGGCAAAUCUUAGAAAAGGGUGUAUACCCUAUAAUUUAACGGCCAGCAGCC"  # a short test
    adjacency = predictor.predict_adjacency(sequence)
    print("[INFO] Adjacency shape:", adjacency.shape)

    # 5) (Optional) If we want to convert adjacency to CT and visualize:
    # For demo, let's mock writing a .ct file:
    mock_ct_file = "test_seq.ct"
    with open(mock_ct_file, "w") as f:
        f.write(">TestSeq\n")
        f.write("1  A  0  2  0  1\n")
        f.write("2  A  1  3  0  2\n")

    # Visualization (conditional based on config)
    # Check if visualization section exists in the config
    if hasattr(stage_cfg, 'visualization') and stage_cfg.get('visualization', {}).get('enabled', False):
        varna_jar_path = stage_cfg.visualization.varna_jar_path # Use path from config
        output_image_path = "test_seq.png" # Keep example output name for now
        # TODO: Consider making output_image_path configurable
        print(f"Attempting visualization with JAR: {varna_jar_path}") # Added print for debugging path
        visualize_with_varna(
            ct_file=mock_ct_file,
            jar_path=varna_jar_path,
            output_png=output_image_path,
            resolution=stage_cfg.visualization.resolution # Pass resolution from config
        )
    else:
        print("[Info] VARNA visualization disabled via config.")


def run_stageA(seq, predictor):
    """
    Simple helper function to integrate with tests.
    Returns adjacency matrix from the given predictor.
    """
    return predictor.predict_adjacency(seq)


if __name__ == "__main__":
    main()
