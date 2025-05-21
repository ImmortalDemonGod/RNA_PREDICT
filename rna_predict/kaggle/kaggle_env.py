"""
kaggle_env.py: Utilities for setting up the Kaggle environment for offline inference.
Encapsulates all Kaggle-specific setup logic (requirements cleaning, wheel install, symlinks, env vars).

Usage:
    from rna_predict.utils.kaggle_env import setup_kaggle_environment
    setup_kaggle_environment()

This function should be called at the start of any CLI/script that needs to run on Kaggle.
"""
import os
import pathlib
import shutil
import subprocess
import sys

def is_kaggle():
    # Kaggle sets this env var in kernels; also check for /kaggle/input dir
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or pathlib.Path("/kaggle/input").is_dir()
    )

def clean_requirements():
    req_path = pathlib.Path("/kaggle/requirements/input_requirements.txt")
    if req_path.is_file():
        cleaned_lines = []
        for line in req_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("pip install"):
                line = f"# {line}"
            cleaned_lines.append(line)
        req_path.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""))
        print(f"[INFO] requirements cleaned – {len(cleaned_lines)} valid pip-install line(s) kept.")
    else:
        print(f"[INFO] {req_path} not found – nothing to clean.")

def install_wheels():
    # Install wheels from every subdir in /kaggle/input
    wheel_root = "/kaggle/input"
    if not pathlib.Path(wheel_root).is_dir():
        print(f"[WARN] {wheel_root} not found; skipping wheel install.")
        return
    find_links_args = []
    for d in [wheel_root] + [str(p) for p in pathlib.Path(wheel_root).iterdir() if p.is_dir()]:
        find_links_args.extend(["--find-links", d])
    # List of packages to install (minimal set; can be expanded as needed)
    pkgs = [
        "numpy==1.24.3", "pandas==2.2.3", "scipy==1.10.1", "tqdm==4.67.1", "seaborn==0.12.2", "biopython==1.85",
        "torch", "huggingface_hub==0.31.1", "transformers==4.51.3", "pytorch_lightning==2.5.0.post0",
        "hydra-core==1.3.2", "omegaconf==2.3.0", "ml_collections==1.1.0", "datasets==3.6.0", "einops==0.8.1",
        "hypothesis==6.131.15", "black==25.1.0", "pathspec==0.12.1", "isort==6.0.1", "ruff==0.11.9", "mss==10.0.0",
        "mdanalysis==2.9.0", "mmtf-python==1.1.3", "GridDataFormats==1.0.2", "mrcfile==1.5.4", "lxml==5.4.0",
        "dearpygui==2.0.0", "py-cpuinfo==9.0.0", "Pillow", "exit-codes==1.3.0"
    ]
    for pkg in pkgs:
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--quiet"] + find_links_args + [pkg]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[WARN] install failed → skipped: {pkg}")

def symlink_models():
    # Example: symlink TorsionBERT weights to expected location
    # Extend as needed for other models/checkpoints
    src = pathlib.Path("/kaggle/input/rna-torsionbert/rna_torsionBERT")
    dst = pathlib.Path("/kaggle/working/rna_torsionBERT")
    if src.is_dir() and not dst.exists():
        try:
            os.symlink(src, dst, target_is_directory=True)
            print(f"[INFO] Symlinked {src} → {dst}")
        except Exception as e:
            print(f"[WARN] Failed to symlink {src} → {dst}: {e}")

def set_offline_env_vars():
    os.environ.update({
        "HF_HUB_OFFLINE":      "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE":"1",
        "HF_HOME":             "/kaggle/working",
    })
    print("[INFO] Set HuggingFace offline environment variables.")

def setup_kaggle_environment():
    if not is_kaggle():
        return
    print("[INFO] Detected Kaggle environment. Running Kaggle-specific setup...")
    clean_requirements()
    install_wheels()
    symlink_models()
    set_offline_env_vars()
    print("[INFO] Kaggle environment setup complete.")
