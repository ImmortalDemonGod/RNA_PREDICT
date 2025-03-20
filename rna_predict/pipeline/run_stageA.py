# rna_predict/pipeline/run_stageA.py
import os
import shutil
import subprocess
import urllib.request

import torch

from rna_predict.pipeline.stageA.stage_a_predictor import StageAPredictor


def download_file(url: str, dest_path: str):
    """
    Download file from URL to a local destination path.
    If the file already exists, check if it's a valid zip (when extension is .zip).
    If invalid, remove and re-download; otherwise skip download.
    """
    import zipfile

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
    import zipfile

    if not os.path.isfile(zip_path):
        print(f"[Warning] Zip file not found: {zip_path}")
        return
    print(f"[Unzip] Extracting {zip_path} into {extract_dir}")

    # ensure the directory exists
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def visualize_with_varna(ct_file: str, jar_path: str, output_png: str):
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
        "8.0",
    ]
    print(f"[VARNA] Running: {' '.join(cmd)}")
    subprocess.Popen(
        cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    ).communicate()[0]
    print(f"[VARNA] Visualization saved to {output_png}")


def build_predictor(
    checkpoint_folder: str, config: dict, device: torch.device
) -> StageARFoldPredictor:
    """
    Create and return the StageARFoldPredictor from a checkpoint folder path.
    """
    checkpoint_path = os.path.join(
        checkpoint_folder, "RNAStralign_trainset_pretrained.pth"
    )
    predictor = StageARFoldPredictor(
        config, checkpoint_path=checkpoint_path, device=device
    )
    return predictor


def main() -> None:
    # 1) Prepare environment
    os.makedirs("RFold", exist_ok=True)

    ckp_folder = "RFold/checkpoints"

    # 2) Build the predictor
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_predictor(ckp_folder, config, device)

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

    varna_jar_path = "RFold/VARNAv3-93.jar"
    output_image_path = "test_seq.png"
    visualize_with_varna(mock_ct_file, varna_jar_path, output_image_path)


def run_stageA(seq, predictor):
    """
    Simple helper function to integrate with tests.
    Returns adjacency matrix from the given predictor.
    """
    return predictor.predict_adjacency(seq)


if __name__ == "__main__":
    main()
