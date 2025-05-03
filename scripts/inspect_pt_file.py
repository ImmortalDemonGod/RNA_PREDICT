import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python inspect_pt_file.py <path_to_pt_file>")
    sys.exit(1)

pt_path = sys.argv[1]
try:
    d = torch.load(pt_path, map_location='cpu')
    print(f"Type: {type(d)}")
    if hasattr(d, 'keys'):
        print(f"Keys: {list(d.keys())}")
    else:
        print(f"Dir: {dir(d)}")
    print("---\nPreview of content:\n")
    preview = str(d)
    if len(preview) > 2000:
        print(preview[:2000] + '\n... [truncated] ...')
    else:
        print(preview)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(2)
