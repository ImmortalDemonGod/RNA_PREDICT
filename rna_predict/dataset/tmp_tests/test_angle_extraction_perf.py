import os
import time
import torch
import numpy as np
from rna_predict.dataset.preprocessing.angles import extract_rna_torsions
from MDAnalysis import Universe
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
import tempfile

# Use a few representative files (adjust path as needed)
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
example_files = [
    os.path.join(EXAMPLES_DIR, f)
    for f in os.listdir(EXAMPLES_DIR)
    if f.endswith('.pdb') or f.endswith('.cif')
]

def convert_cif_to_pdb(cif_file):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("mmcif_structure", cif_file)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_handle:
        pdb_path = tmp_handle.name
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    return pdb_path


def get_first_chain_id(structure_file):
    import os
    _, ext = os.path.splitext(structure_file)
    ext = ext.lower()
    if ext == ".cif":
        try:
            pdb_path = convert_cif_to_pdb(structure_file)
            u = Universe(pdb_path)
            chainids = list(set(u.atoms.chainIDs) | set(u.atoms.segids))
            chainids = [c for c in chainids if c and c != '']
            os.remove(pdb_path)
            if chainids:
                return chainids[0]
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Could not convert or load {structure_file} to determine chain id: {e}")
            return None
    else:
        try:
            u = Universe(structure_file)
            chainids = list(set(u.atoms.chainIDs) | set(u.atoms.segids))
            chainids = [c for c in chainids if c and c != '']
            if chainids:
                return chainids[0]
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Could not load {structure_file} to determine chain id: {e}")
            return None


def test_torsion_angle_extraction_speed_and_memory():
    times = []
    peak_mems = []
    successes = 0
    failures = 0
    try:
        from memory_profiler import memory_usage
    except ImportError:
        memory_usage = None

    for fname in example_files:
        chain_id = get_first_chain_id(fname)
        if not chain_id:
            print(f"[SKIP] {os.path.basename(fname)}: No chain id found")
            continue
        start = time.perf_counter()
        if memory_usage:
            mem, result = memory_usage(
                (extract_rna_torsions, (fname,), {'chain_id': chain_id, 'backend': 'mdanalysis'}),
                retval=True, interval=0.01, timeout=10, max_usage=True
            )
            peak_mem = max(mem)
        else:
            result = extract_rna_torsions(fname, chain_id=chain_id, backend='mdanalysis')
            peak_mem = None
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        peak_mems.append(peak_mem)
        if isinstance(result, (np.ndarray, torch.Tensor)):
            successes += 1
        else:
            failures += 1
            print(f"[FAIL] {os.path.basename(fname)}: result type={type(result)}, value={result}")
        print(f"{os.path.basename(fname)}: chain_id={chain_id}, {elapsed:.3f}s, peak_mem={peak_mem} MB")
    print(f"Mean extraction time: {np.mean(times):.3f}s, Median: {np.median(times):.3f}s, Max: {np.max(times):.3f}s")
    if any(peak_mems):
        print(f"Peak memory usage (MB): {peak_mems}")
    print(f"Successes: {successes}, Failures: {failures}")

if __name__ == '__main__':
    test_torsion_angle_extraction_speed_and_memory()
