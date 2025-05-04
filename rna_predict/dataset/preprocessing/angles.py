from typing import Optional, Literal
import os
import tempfile
from typing import Optional, Literal
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
import MDAnalysis as mda  # type: ignore
import numpy as np


def extract_rna_torsions(
    structure_file: str,
    chain_id: str = "A",
    backend: Literal["mdanalysis", "dssr"] = "mdanalysis",
    **kwargs
) -> Optional[np.ndarray]:
    """
    Extract backbone and glycosidic torsion angles for an RNA structure.
    Args:
        structure_file: Path to .pdb or .cif file.
        chain_id: Which chain to extract (default: "A").
        backend: Which backend to use ("mdanalysis" [default], "dssr" [future]).
        kwargs: Backend-specific options.
    Returns:
        np.ndarray of shape [L, 7] (alpha, beta, gamma, delta, epsilon, zeta, chi), radians.
        Returns None if extraction fails.
    """
    print(f"[DEBUG] extract_rna_torsions called with: structure_file={structure_file}, chain_id={chain_id}, backend={backend}")
    if backend == "mdanalysis":
        try:
            result = _extract_rna_torsions_mdanalysis(structure_file, chain_id)
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
    """Convert mmCIF file to PDB using Biopython. Returns path to temp PDB."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("mmcif_structure", cif_file)
    tmp_handle = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp_handle.close()
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
    """Select chain with fallback to nucleic atoms if no chain is found."""
    chain = _select_chain(universe, chain_id)
    # If chain not found and chain_id is specified, return None (strict mode)
    if (chain is None or len(chain) == 0) and chain_id is not None:
        print(f"[DEBUG] _select_chain_with_fallback: No chain found for requested chain_id={chain_id}, not falling back.")
        return None
    # If no chain found or no chain specified, fallback to all nucleic atoms
    if chain is None or len(chain) == 0:
        chain = universe.select_atoms("nucleic")
        print(f"[DEBUG] _select_chain_with_fallback: Fallback to nucleic, found {len(chain)} atoms.")
        if len(chain) == 0:
            print(f"[DEBUG] _select_chain_with_fallback: No nucleic atoms found.")
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
    """Calculate dihedral angle in radians given 4 points. Return None if invalid."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-12 or norm_n2 < 1e-12:
        return None
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

def _extract_torsions_from_residues(residues) -> np.ndarray:
    """Extract torsion angles for each residue in a residue list."""
    n_res = len(residues)
    torsion_data = np.full((n_res, 7), np.nan, dtype=np.float32)
    for i, res in enumerate(residues):
        prev_res = residues[i - 1] if i > 0 else None
        next_res = residues[i + 1] if i < n_res - 1 else None
        torsion_data[i, 0] = _alpha_torsion(prev_res, res)
        torsion_data[i, 1] = _beta_torsion(res)
        torsion_data[i, 2] = _gamma_torsion(res)
        torsion_data[i, 3] = _delta_torsion(res)
        torsion_data[i, 4] = _epsilon_torsion(res, next_res)
        torsion_data[i, 5] = _zeta_torsion(res, next_res)
        torsion_data[i, 6] = _chi_torsion(res)
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
        if self.using_temp and self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)


def _extract_rna_torsions_mdanalysis(structure_file: str, chain_id: str) -> Optional[np.ndarray]:
    """
    Implementation of RNA torsion extraction using MDAnalysis.
    Returns [L, 7] array (radians), np.nan for missing.
    """
    print(f"[DEBUG] _extract_rna_torsions_mdanalysis called with: {structure_file}, chain_id={chain_id}")
    with TempFileManager(structure_file) as mda_file:
        print(f"[DEBUG] TempFileManager returned: {mda_file}")
        universe = _load_universe(mda_file)
        if universe is None:
            print(f"[WARN] _load_universe returned None for file: {mda_file}")
            return None
        chain = _select_chain_with_fallback(universe, chain_id)
        if chain is None:
            print(f"[WARN] _select_chain_with_fallback returned None for chain_id: {chain_id}")
            return None
        residues = chain.residues
        print(f"[DEBUG] Number of residues in chain: {len(residues)}")
        torsion_data = _extract_torsions_from_residues(residues)
        print(f"[DEBUG] torsion_data shape: {torsion_data.shape} (should be [L,7])")
        return torsion_data
