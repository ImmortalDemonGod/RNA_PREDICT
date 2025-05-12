from typing import Optional, Literal
import os
import tempfile
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
import MDAnalysis as mda  # type: ignore
import snoop

@snoop
def extract_rna_torsions(
    structure_file: str,
    chain_id: str = "A",
    backend: Literal["mdanalysis", "dssr"] = "mdanalysis",
    angle_set: Literal["canonical", "full"] = "canonical",
    **kwargs
) -> Optional[np.ndarray]:
    """
    Extracts RNA torsion angles from a structure file for a specified chain.
    
    Given a PDB or mmCIF file, computes either the canonical set of 7 RNA torsion
    angles (alpha, beta, gamma, delta, epsilon, zeta, chi) or an extended set of
    14 angles (including ribose and pseudo-torsions) for each residue in the
    selected chain, using the specified backend.
    
    Args:
        structure_file: Path to the RNA structure file (.pdb or .cif).
        chain_id: Chain identifier to extract torsions from.
        backend: Extraction backend to use ("mdanalysis" or "dssr").
        angle_set: "canonical" for 7 angles, or "full" for 14 angles.
    
    Returns:
        A NumPy array of shape [L, 7] or [L, 14] with torsion angles in radians
        for each residue, or None if extraction fails.
    """
    print(f"[DEBUG] extract_rna_torsions called with: structure_file={structure_file}, chain_id={chain_id}, backend={backend}, angle_set={angle_set}")
    if backend == "mdanalysis":
        try:
            result = _extract_rna_torsions_mdanalysis(structure_file, chain_id, angle_set=angle_set)
            print(f"[DEBUG] _extract_rna_torsions_mdanalysis returned type: {type(result)}, value: {result if result is None else 'array shape ' + str(result.shape)}")
            return result
        except Exception as e:
            print(f"[ERROR] extract_rna_torsions failed for {structure_file}: {e}")
            return None
    elif backend == "dssr":
        # Placeholder for future DSSR backend
        raise NotImplementedError("DSSR backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _convert_cif_to_pdb(cif_file: str) -> str:
    """
    Converts an mmCIF file to a temporary PDB file.
    
    Parses the input mmCIF structure and writes it to a temporary PDB file. The path to the temporary file is returned; the caller is responsible for deleting the file after use.
    
    Args:
        cif_file: Path to the input mmCIF file.
    
    Returns:
        Path to the generated temporary PDB file.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("mmcif_structure", cif_file)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_handle:
        pdb_path = tmp_handle.name
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    return pdb_path


def _load_universe(structure_file: str) -> Optional['mda.Universe']:
    """Load structure file with MDAnalysis, return Universe or None on failure."""
    try:
        return mda.Universe(structure_file)
    except Exception as e:
        print(f"MDAnalysis failed to load {structure_file}: {e}")
        return None


def _select_chain(u: 'mda.Universe', chain_id: str) -> Optional['mda.core.groups.AtomGroup']:
    """Select chain by segid or chainID. Return AtomGroup or None if not found."""
    all_chainids = set(u.atoms.chainIDs)
    all_segids = set(u.atoms.segids)
    print(f"[DEBUG] _select_chain: available chainIDs={all_chainids}, segids={all_segids}, requested={chain_id}")
    chain = u.select_atoms(f"(segid {chain_id}) or (chainID {chain_id})")
    if len(chain) == 0:
        print(f"[DEBUG] _select_chain: No atoms found for chain_id={chain_id}")
        return None
    print(f"[DEBUG] _select_chain: Found {len(chain)} atoms for chain_id={chain_id}")
    return chain


def _select_chain_with_fallback(universe: 'mda.Universe', chain_id: str) -> Optional['mda.core.groups.AtomGroup']:
    """
    Selects a chain from the universe, with fallback to the only available chain or all nucleic atoms.
    
    If the requested chain is not found but exactly one unique chain or segment exists, selects that chain instead. If multiple chains exist and the requested chain is missing, returns None. If no chains are present, or if no atoms are found for the requested or fallback chain, attempts to select all nucleic atoms. Returns None if no suitable atoms are found.
    """
    chain = _select_chain(universe, chain_id)
    if (chain is None or len(chain) == 0) and chain_id is not None:
        # List all unique chain IDs and segids
        all_chainids = set(universe.atoms.chainIDs)
        all_segids = set(universe.atoms.segids)
        # Remove blanks
        all_chainids = {c for c in all_chainids if c and c.strip()}
        all_segids = {s for s in all_segids if s and s.strip()}
        all_ids = all_chainids | all_segids
        if len(all_ids) == 1:
            only_chain = list(all_ids)[0]
            print(f"[INFO] Requested chain_id={chain_id} not found, but only one chain ({only_chain}) present. Using it.")
            chain = _select_chain(universe, only_chain)
            if chain is not None and len(chain) > 0:
                return chain
            else:
                print(f"[WARN] Fallback to only chain {only_chain} failed to select atoms.")
                return None
        elif len(all_ids) == 0:
            print("[DEBUG] No chains present in structure.")
            return None
        else:
            print(f"[DEBUG] _select_chain_with_fallback: No chain found for requested chain_id={chain_id}, multiple chains present ({all_ids}), not falling back.")
            return None
    # If no chain found or no chain specified, fallback to all nucleic atoms
    if chain is None or len(chain) == 0:
        chain = universe.select_atoms("nucleic")
        print(f"[DEBUG] _select_chain_with_fallback: Fallback to nucleic, found {len(chain)} atoms.")
        if len(chain) == 0:
            print("[DEBUG] _select_chain_with_fallback: No nucleic atoms found.")
            return None
    return chain


def _safe_select_atom(res, name: str):
    """Safely select atom by name from residue. Return position or None."""
    if res is None:
        return None
    sel = res.atoms.select_atoms(f"name {name}")
    if sel and len(sel.positions) > 0:
        return sel.positions[0]
    return None


def _calc_dihedral(p1, p2, p3, p4):
    """
    Calculates the dihedral angle in radians defined by four 3D points.
    
    Returns:
        The dihedral angle in radians, or np.nan if any input is missing or the calculation is invalid.
    """
    if p1 is None or p2 is None or p3 is None or p4 is None:
        return np.nan
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-12 or norm_n2 < 1e-12:
        return np.nan
    cos_angle = np.dot(n1, n2) / (norm_n1 * norm_n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    sign = np.dot(b2, np.cross(n1, n2))
    phi = np.arccos(cos_angle)
    if sign < 0.0:
        phi = -phi
    return np.deg2rad(np.degrees(phi))


def _alpha_torsion(prev_res, res):
    """Calculate alpha torsion for residue i (needs i-1 and i)."""
    if prev_res is None:
        return np.nan
    atoms = [_safe_select_atom(prev_res, "O3'"), _safe_select_atom(res, "P"), _safe_select_atom(res, "O5'"), _safe_select_atom(res, "C5'")]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _beta_torsion(res):
    atoms = [_safe_select_atom(res, "P"), _safe_select_atom(res, "O5'"), _safe_select_atom(res, "C5'"), _safe_select_atom(res, "C4'")]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _gamma_torsion(res):
    atoms = [_safe_select_atom(res, "O5'"), _safe_select_atom(res, "C5'"), _safe_select_atom(res, "C4'"), _safe_select_atom(res, "C3'")]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _delta_torsion(res):
    atoms = [_safe_select_atom(res, "C5'"), _safe_select_atom(res, "C4'"), _safe_select_atom(res, "C3'") , _safe_select_atom(res, "O3'")]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _epsilon_torsion(res, next_res):
    if next_res is None:
        return np.nan
    atoms = [_safe_select_atom(res, "C4'"), _safe_select_atom(res, "C3'"), _safe_select_atom(res, "O3'"), _safe_select_atom(next_res, "P")]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _zeta_torsion(res, next_res):
    if next_res is None:
        return np.nan
    atoms = [_safe_select_atom(res, "C3'"), _safe_select_atom(res, "O3'"), _safe_select_atom(next_res, "P"), _safe_select_atom(next_res, "O5'")]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _chi_torsion(res):
    """
    Calculates the chi torsion angle for a nucleic acid residue.
    
    The chi angle is defined by the atoms O4', C1', N1/N9, and C2/C4, with fallback to N9 and C4 if N1 or C2 are not present. Returns the angle in radians, or np.nan if required atoms are missing.
    """
    O4 = _safe_select_atom(res, "O4'")
    C1 = _safe_select_atom(res, "C1'")
    N_base = _safe_select_atom(res, "N1")
    if N_base is None:
        N_base = _safe_select_atom(res, "N9")
    C_base = _safe_select_atom(res, "C2")
    if C_base is None:
        C_base = _safe_select_atom(res, "C4")
    atoms = [O4, C1, N_base, C_base]
    return _calc_dihedral(*atoms) if all(a is not None for a in atoms) else np.nan

def _nu0_torsion(res):
    """
    Calculates the nu0 ribose ring torsion angle for a residue.
    
    The nu0 torsion is defined by the atoms C4', O4', C1', and C2' of the ribose ring.
    Returns the dihedral angle in radians, or np.nan if any atom is missing.
    """
    p1 = _safe_select_atom(res, "C4'")
    p2 = _safe_select_atom(res, "O4'")
    p3 = _safe_select_atom(res, "C1'")
    p4 = _safe_select_atom(res, "C2'")
    return _calc_dihedral(p1, p2, p3, p4)

def _nu1_torsion(res):
    """
    Calculates the nu1 ribose ring torsion angle for a residue.
    
    The nu1 torsion is defined by the atoms O4', C1', C2', and C3' of the given residue.
    Returns the dihedral angle in radians, or np.nan if any atom is missing.
    """
    p1 = _safe_select_atom(res, "O4'")
    p2 = _safe_select_atom(res, "C1'")
    p3 = _safe_select_atom(res, "C2'")
    p4 = _safe_select_atom(res, "C3'")
    return _calc_dihedral(p1, p2, p3, p4)

def _nu2_torsion(res):
    """
    Calculates the nu2 ribose ring torsion angle for a residue.
    
    The nu2 angle is defined by the dihedral formed by the atoms C1', C2', C3', and C4' within the same residue. Returns the angle in radians, or np.nan if any atom is missing.
    """
    p1 = _safe_select_atom(res, "C1'")
    p2 = _safe_select_atom(res, "C2'")
    p3 = _safe_select_atom(res, "C3'")
    p4 = _safe_select_atom(res, "C4'")
    return _calc_dihedral(p1, p2, p3, p4)

def _nu3_torsion(res):
    """
    Calculates the nu3 ribose ring torsion angle for a residue.
    
    The nu3 angle is defined by the dihedral formed by the atoms C2', C3', C4', and O4' within the same residue. Returns the angle in radians, or np.nan if any atom is missing.
    """
    p1 = _safe_select_atom(res, "C2'")
    p2 = _safe_select_atom(res, "C3'")
    p3 = _safe_select_atom(res, "C4'")
    p4 = _safe_select_atom(res, "O4'")
    return _calc_dihedral(p1, p2, p3, p4)

def _nu4_torsion(res):
    """
    Calculates the nu4 ribose ring torsion angle for a residue.
    
    The nu4 angle is defined by the dihedral formed by the atoms C3', C4', O4', and C1' within the same residue. Returns the angle in radians, or np.nan if any atom is missing.
    """
    p1 = _safe_select_atom(res, "C3'")
    p2 = _safe_select_atom(res, "C4'")
    p3 = _safe_select_atom(res, "O4'")
    p4 = _safe_select_atom(res, "C1'")
    return _calc_dihedral(p1, p2, p3, p4)

def _eta_torsion(prev_res, res, next_res):
    """
    Calculates the eta pseudo-torsion angle for an RNA residue.
    
    The eta angle is defined by the dihedral formed by the C4' atom of the previous residue, the P and C4' atoms of the current residue, and the P atom of the next residue. Returns the angle in radians, or np.nan if any required atom is missing.
    """
    if prev_res is None or next_res is None:
        return np.nan
    p1 = _safe_select_atom(prev_res, "C4'")
    p2 = _safe_select_atom(res, "P")
    p3 = _safe_select_atom(res, "C4'")
    p4 = _safe_select_atom(next_res, "P")
    return _calc_dihedral(p1, p2, p3, p4)

def _theta_torsion(res, next_res, next_next_res):
    """
    Calculates the theta pseudo-torsion angle for RNA using three consecutive residues.
    
    The theta angle is defined by the dihedral formed by the atoms: P and C4' of the current residue,
    P of the next residue, and C4' of the residue after that. Returns the angle in radians, or np.nan
    if any required atom is missing.
    """
    if next_res is None or next_next_res is None:
        return np.nan
    p1 = _safe_select_atom(res, "P")
    p2 = _safe_select_atom(res, "C4'")
    p3 = _safe_select_atom(next_res, "P")
    p4 = _safe_select_atom(next_next_res, "C4'")
    return _calc_dihedral(p1, p2, p3, p4)

def _extract_torsions_from_residues(residues, angle_set="canonical") -> np.ndarray:
    """
    Extracts torsion angles for each residue in a list, returning canonical or full sets.
    
    Args:
        residues: List of residue objects to process.
        angle_set: Either "canonical" for 7 standard RNA torsions or "full" for 14 angles
            including ribose and pseudo-torsions.
    
    Returns:
        A NumPy array of shape (N, 7) or (N, 14) with torsion angles in radians for each
        residue, using np.nan for missing or undefined values.
    """
    n_res = len(residues)
    if angle_set == "full":
        torsion_data = np.full((n_res, 14), np.nan, dtype=np.float32)
    else:
        torsion_data = np.full((n_res, 7), np.nan, dtype=np.float32)
    for i, res in enumerate(residues):
        prev_res = residues[i - 1] if i > 0 else None
        next_res = residues[i + 1] if i < n_res - 1 else None
        next_next_res = residues[i + 2] if i < n_res - 2 else None
        torsion_data[i, 0] = _alpha_torsion(prev_res, res)
        torsion_data[i, 1] = _beta_torsion(res)
        torsion_data[i, 2] = _gamma_torsion(res)
        torsion_data[i, 3] = _delta_torsion(res)
        torsion_data[i, 4] = _epsilon_torsion(res, next_res)
        torsion_data[i, 5] = _zeta_torsion(res, next_res)
        torsion_data[i, 6] = _chi_torsion(res)
        if angle_set == "full":
            torsion_data[i, 7] = _nu0_torsion(res)
            torsion_data[i, 8] = _nu1_torsion(res)
            torsion_data[i, 9] = _nu2_torsion(res)
            torsion_data[i, 10] = _nu3_torsion(res)
            torsion_data[i, 11] = _nu4_torsion(res)
            torsion_data[i, 12] = _eta_torsion(prev_res, res, next_res)
            torsion_data[i, 13] = _theta_torsion(res, next_res, next_next_res)
    return torsion_data


class TempFileManager:
    """Context manager for handling temporary files."""
    def __init__(self, original_file: str):
        self.original_file = original_file
        self.temp_file = None
        self.using_temp = False

    def __enter__(self):
        _, ext = os.path.splitext(self.original_file)
        ext = ext.lower()

        if ext == ".cif":
            self.using_temp = True
            self.temp_file = _convert_cif_to_pdb(self.original_file)
            return self.temp_file
        else:
            return self.original_file

    def __exit__(self, *_):
        """
        Cleans up the temporary file if one was created during context management.
        """
        if self.using_temp and self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)


def _extract_rna_torsions_mdanalysis(structure_file: str, chain_id: str, angle_set: str = "canonical") -> Optional[np.ndarray]:
    """
    Extracts RNA torsion angles from a structure file for a specified chain using MDAnalysis.
    
    Loads the structure, selects the specified chain (with fallback if necessary), and computes torsion angles for each residue. Returns a NumPy array of shape [L, 7] for canonical angles or [L, 14] for the full angle set, with angles in radians and np.nan for missing values. Returns None if loading or selection fails.
    
    Args:
        structure_file: Path to the PDB or mmCIF structure file.
        chain_id: Chain identifier to extract torsions from.
        angle_set: "canonical" for 7 standard angles, "full" for 14 angles including ribose and pseudo-torsions.
    
    Returns:
        NumPy array of torsion angles (shape [L, 7] or [L, 14]), or None on failure.
    """
    print(f"[DEBUG] _extract_rna_torsions_mdanalysis called with: {structure_file}, chain_id={chain_id}, angle_set={angle_set}")
    with TempFileManager(structure_file) as mda_file:
        u = _load_universe(mda_file)
        if u is None:
            print(f"[ERROR] Could not load universe for {structure_file}")
            return None
        chain = _select_chain_with_fallback(u, chain_id)
        if chain is None or len(chain.residues) == 0:
            print(f"[ERROR] No residues found in chain {chain_id} for {structure_file}")
            return None
        residues = chain.residues
        print(f"[DEBUG] Number of residues in chain: {len(residues)}")
        torsion_data = _extract_torsions_from_residues(residues, angle_set=angle_set)
        print(f"[DEBUG] torsion_data shape: {torsion_data.shape} (should be [L,7] or [L,14])")
        return torsion_data
