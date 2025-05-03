# rna_predict/dataset/loader.py
import torch
import pandas as pd
from torch.utils.data import Dataset
from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBParser
from .atom_lists import STANDARD_ATOMS

# Utility stubs (implement or replace with actual logic)
def parse_pdb_atoms(pdb_file):
    """Parse atom coordinates from .cif or .pdb file using Biopython."""
    import os
    ext = os.path.splitext(pdb_file)[1].lower()
    coords = {}
    if ext == ".cif":
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("rna", pdb_file)
    elif ext == ".pdb":
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("rna", pdb_file)
    else:
        raise ValueError(f"Unsupported structure file extension: {ext}")
    for model in structure:
        for chain in model:
            for residue in chain:
                res_idx = residue.id[1] - 1  # 0-based
                for atom in residue:
                    coords[(res_idx, atom.get_name())] = atom.get_coord()
    return coords

def element_one_hot(atom_name):
    # Minimal one-hot encoding for element type (P, C, N, O, S)
    import torch
    elements = ["P", "C", "N", "O", "S"]
    one_hot = torch.zeros(len(elements), dtype=torch.float32)
    if atom_name and atom_name[0] in elements:
        one_hot[elements.index(atom_name[0])] = 1.0
    return one_hot

def atom_name_embedding(atom_name):
    # Minimal one-hot encoding for atom name (first 10 standard names)
    import torch
    standard_names = [
        "P", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8"
    ]
    one_hot = torch.zeros(10, dtype=torch.float32)
    if atom_name in standard_names:
        one_hot[standard_names.index(atom_name)] = 1.0
    return one_hot

class RNADataset(Dataset):
    """PyTorch Dataset for RNA structure prediction.
    Loads sequence, coordinates, atom features, and optional adjacency/angle data for each sample.
    Args:
        index_csv (str): Path to index CSV file.
        cfg (omegaconf.DictConfig): Hydra config object.
        load_adj (bool): Whether to load adjacency data.
        load_ang (bool): Whether to load angle data.
    """
    def __init__(self, index_csv, cfg, *, load_adj=False, load_ang=False, verbose=False):
        self.cfg = cfg
        # Use pandas for robust CSV loading, preserving empty strings and forcing all columns to str
        df = pd.read_csv(index_csv, dtype=str, keep_default_na=False, na_values=[''])
        df = df.fillna('')  # Ensure all NaN are replaced with empty strings
        if df.shape[0] == 0:
            print("[RNADataset] WARNING: index_csv loaded but contains no rows!")
        self.meta = df.to_records(index=False)
        print(f"[RNADataset] Loaded index_csv from: {index_csv}")
        print(f"[RNADataset] type(self.meta): {type(self.meta)}")
        print(f"[RNADataset] repr(self.meta): {repr(self.meta)}")
        if hasattr(self.meta, '__len__') and len(self.meta) > 0:
            print(f"[RNADataset] First row fields: {self.meta[0].dtype.names}")
            print(f"[RNADataset] First row values: {self.meta[0]}")
        elif hasattr(self.meta, 'dtype'):
            print(f"[RNADataset] Single row fields: {self.meta.dtype.names}")
            print(f"[RNADataset] Single row values: {self.meta}")
        else:
            print("[RNADataset] WARNING: index_csv loaded but contains no rows or is malformed!")
        self.load_adj = load_adj
        self.load_ang = load_ang
        self.max_res = cfg.data.max_residues
        self.max_atoms = cfg.data.max_atoms
        self.verbose = verbose
        # Config validation
        required_keys = ["max_residues", "max_atoms", "batch_size", "index_csv"]
        for k in required_keys:
            if not hasattr(cfg.data, k):
                raise ValueError(f"Missing required config key: cfg.data.{k}")

    def __len__(self):
        return len(self.meta)


    def __getitem__(self, i):
        row = self.meta[i]
        seq_id = row["id"]
        seq = self._load_sequence(row["sequence_path"], row["target_id"])
        L = len(seq)

        coords, atom_mask, atom_to_tok, elem_emb, name_emb, tgt_names, tgt_indices = self._load_atom_features(row["pdb_path"], L)

        residue_mask = torch.zeros(self.max_res, dtype=torch.bool)
        residue_mask[:L] = True

        sample = dict(
            sequence_id=seq_id,
            sequence=seq,
            coords_true=coords,
            attention_mask=residue_mask,
            atom_mask=atom_mask,
            atom_to_token_idx=atom_to_tok,
            ref_element=elem_emb,
            ref_atom_name_chars=name_emb,
            ref_charge=torch.zeros_like(atom_mask, dtype=torch.float32)[..., None],
            atom_names=tgt_names,  # NEW: target atom names
            residue_indices=tgt_indices,  # NEW: target residue indices
        )
        # Add all CSV metadata fields to the sample dict
        for field in row.dtype.names:
            sample[field] = row[field]
        if self.load_adj:
            sample["adjacency_true"] = self._load_adj(row, L)
        if self.load_ang:
            sample["angles_true"] = self._load_angles(row, L)
        if self.verbose:
            print(f"[DEBUG][RNADataset.__getitem__] index={i}, sample keys={list(sample.keys())}")
            for k, v in sample.items():
                print(f"  - {k}: {type(v)} shape={getattr(v, 'shape', None)}")
        return sample

    def _load_sequence(self, sequence_path, target_id=None):
        """
        Load sequence from a CSV (Kaggle format), FASTA, or fallback to parsing .cif if needed.
        Args:
            sequence_path: Path to the sequence file (CSV or FASTA)
            target_id: ID to look up in CSV (if needed)
        """
        import os
        if sequence_path.endswith('.csv'):
            import csv
            if not target_id:
                raise ValueError("target_id must be provided for CSV sequence lookup.")
            with open(sequence_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['target_id'] == target_id:
                        return row['sequence'][: self.max_res]
            print(f"[WARNING] target_id {target_id} not found in {sequence_path}, returning dummy sequence")
            return "A" * 10
        elif sequence_path and os.path.exists(sequence_path):
            try:
                rec = next(SeqIO.parse(sequence_path, "fasta"))
                return str(rec.seq)[: self.max_res]
            except Exception as e:
                print(f"[WARNING] Failed to parse FASTA: {sequence_path}, error: {e}")
        # Fallback: try to parse sequence from .cif
        cif_file = sequence_path.replace('.fasta', '.cif')
        if os.path.exists(cif_file):
            try:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("rna", cif_file)
                seq = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if hasattr(residue, 'resname'):
                                seq.append(residue.resname.strip())
                three_to_one = {"A": "A", "C": "C", "G": "G", "U": "U"}
                seq1 = ''.join([three_to_one.get(r[0], "N") for r in seq])
                return seq1[: self.max_res]
            except Exception as e:
                print(f"[WARNING] Failed to parse sequence from CIF: {cif_file}, error: {e}")
        print(f"[WARNING] Sequence file not found: {sequence_path}, returning dummy sequence")
        return "A" * 10

    def _load_atom_features(self, pdb_file, L):
        # Get dtype and fill value from config
        coord_dtype = getattr(torch, self.cfg.data.coord_dtype)
        fill_value = float('nan') if str(self.cfg.data.coord_fill_value).lower() == 'nan' else float(self.cfg.data.coord_fill_value)
        # If pdb_file is missing or empty, return dummy tensors
        if not pdb_file:
            print("[RNADataset] Empty pdb_file path, returning dummy atom features.")
            coords = torch.zeros((L, self.max_atoms, 3), dtype=coord_dtype)
            atom_mask = torch.zeros((L, self.max_atoms), dtype=torch.float32)
            atom_to_tok = torch.zeros((L, self.max_atoms), dtype=torch.int32)
            elem_emb = torch.zeros((L, self.max_atoms, self.cfg.data.ref_element_size), dtype=torch.float32)
            name_emb = torch.zeros((L, self.max_atoms, self.cfg.data.ref_atom_name_chars_size), dtype=torch.float32)
            tgt_names = []
            tgt_indices = []
            return coords, atom_mask, atom_to_tok, elem_emb, name_emb, tgt_names, tgt_indices
        coords_dict = parse_pdb_atoms(pdb_file)
        coords = torch.full((self.max_atoms, 3), fill_value, dtype=coord_dtype)
        atom_mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        atom_to_tok = torch.zeros(self.max_atoms, dtype=torch.long)
        elem_emb = torch.zeros(self.max_atoms, self.cfg.data.ref_element_size)  # Configurable shape
        name_emb = torch.zeros(self.max_atoms, self.cfg.data.ref_atom_name_chars_size)  # Configurable shape
        tgt_names = []  # NEW: List of atom names for present atoms
        tgt_indices = []  # NEW: List of residue indices for present atoms
        a_idx = 0
        for r in range(L):
            for atom_name in STANDARD_ATOMS:
                if a_idx >= self.max_atoms:
                    break
                key = (r, atom_name)
                if key in coords_dict:
                    coords[a_idx] = torch.from_numpy(coords_dict[key])
                    atom_mask[a_idx] = True
                    tgt_names.append(atom_name)  # NEW: Add atom name
                    tgt_indices.append(r)  # NEW: Add residue index
                atom_to_tok[a_idx] = r
                elem_emb[a_idx] = element_one_hot(atom_name)
                name_emb[a_idx] = atom_name_embedding(atom_name)
                a_idx += 1
        return coords, atom_mask, atom_to_tok, elem_emb, name_emb, tgt_names, tgt_indices

    def _load_adj(self, row, L):
        # TODO: Implement adjacency loading
        return torch.zeros((self.max_res, self.max_res))

    def _load_angles(self, row, L):
        # TODO: Implement angles loading
        return torch.zeros((self.max_res, 14))
