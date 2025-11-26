import torch
import sys
from pathlib import Path

def main(ckpt_path):
    print(f"Inspecting checkpoint: {ckpt_path}")
    if not Path(ckpt_path).exists():
        print("[ERROR] Checkpoint file does not exist.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Top-level type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"state_dict keys (first 10): {list(state_dict.keys())[:10]}")
        else:
            print(f"Checkpoint keys (first 10): {list(ckpt.keys())[:10]}")
    else:
        print("[WARN] Checkpoint is not a dict. Printing repr:")
        print(repr(ckpt))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <path_to_ckpt>")
        sys.exit(1)
    main(sys.argv[1])
