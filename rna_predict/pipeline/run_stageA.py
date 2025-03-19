# rna_predict/pipeline/run_stageA.py
import os
import subprocess
import shutil
import urllib.request
import torch

from rna_predict.pipeline.stageA.rfold import StageARFoldPredictor

def download_file(url: str, dest_path: str):
    """
    Download file from URL to a local destination path.
    Skips download if the file already exists.
    """
    if os.path.isfile(dest_path):
        print(f"[Info] File already exists, skipping download: {dest_path}")
        return
    print(f"[Download] Fetching {url}")
    with urllib.request.urlopen(url) as r, open(dest_path, 'wb') as f:
        shutil.copyfileobj(r, f)
    print(f"[Download] Saved to {dest_path}")

def unzip_file(zip_path: str, extract_dir: str):
    """
    Unzip the zip_path into extract_dir, overwriting existing files.
    Requires 'unzip' command to be available (on Linux/macOS).
    """
    if not os.path.isfile(zip_path):
        print(f"[Warning] Zip file not found: {zip_path}")
        return
    print(f"[Unzip] Extracting {zip_path} into {extract_dir}")
    subprocess.run(["unzip", "-o", zip_path, "-d", extract_dir], check=False)

def visualize_with_varna(ct_file: str, jar_path: str, output_png: str):
    """
    Small helper function to call the VARNA .jar to generate RNA secondary structure images.
    Requires Java on the system path and the jar at jar_path.
    """
    if not os.path.isfile(ct_file):
        print(f"[Warning] CT file not found: {ct_file}")
        return
    if not os.path.isfile(jar_path):
        print(f"[Warning] VARNA JAR not found at: {jar_path} -> skipping visualization.")
        return

    cmd = [
        "java", "-cp", jar_path,
        "fr.orsay.lri.varna.applications.VARNAcmd",
        "-i", ct_file,
        "-o", output_png,
        "-resolution", "8.0"
    ]
    print(f"[VARNA] Running: {' '.join(cmd)}")
    subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    print(f"[VARNA] Visualization saved to {output_png}")

def main():
    # 1) Prepare to download/unzip checkpoints (mimicking the Colab notebook)
    os.makedirs("RFold", exist_ok=True)
    checkpoint_zip_url = "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=0"
    data_zip_url = "https://www.dropbox.com/s/wzbkd3q43haax0r/data.zip?dl=0"

    ckp_zip_path = "RFold/checkpoints.zip"
    data_zip_path = "RFold/data.zip"
    ckp_folder = "RFold/checkpoints"

    # Download zip files
    download_file(checkpoint_zip_url, ckp_zip_path)
    download_file(data_zip_url, data_zip_path)

    # Unzip them
    unzip_file(ckp_zip_path, "RFold/")
    unzip_file(data_zip_path, "RFold/")

    # 2) The specific checkpoint we want to load (as in the Colab demo)
    # Make sure the checkpoint file name matches what's inside the zip
    checkpoint = os.path.join(ckp_folder, "RNAStralign_trainset_pretrained.pth")

    # 3) Build the predictor
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = StageARFoldPredictor(config, checkpoint_path=checkpoint, device=device)

    # 4) Example inference
    sequence = "AAGUCUGGUGGACAUUGGCU"  # a short test
    adjacency = predictor.predict_adjacency(sequence)
    print("[INFO] Adjacency shape:", adjacency.shape)

    # 5) (Optional) If we want to convert adjacency to CT and visualize:
    # For demo, let's mock writing a .ct file:
    ct_file = "test_seq.ct"
    with open(ct_file, "w") as f:
        f.write(">TestSeq\n")
        f.write("1  A  0  2  0  1\n")  # minimal lines
        f.write("2  A  1  3  0  2\n")  # etc. only placeholder data

    # Then call VARNA if desired:
    varna_jar = "VARNAv3-93.jar"  # update if you have a different path
    output_png = "test_seq.png"
    visualize_with_varna(ct_file, varna_jar, output_png)

if __name__ == "__main__":
    main()