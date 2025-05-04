import os
import numpy as np
import pytest
from rna_predict.dataset.preprocessing.angles import extract_rna_torsions
from hypothesis import given, strategies as st

# Set EXAMPLES_DIR to the correct absolute path
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXAMPLES_DIR = os.path.join(dataset_dir, 'examples')

def test_extract_rna_torsions_wrong_backend():
    with pytest.raises(ValueError):
        extract_rna_torsions("not_a_file.pdb", backend="nonsense")

def test_extract_rna_torsions_dssr_not_implemented():
    with pytest.raises(NotImplementedError):
        extract_rna_torsions("not_a_file.pdb", backend="dssr")

def test_extract_rna_torsions_file_not_found():
    assert extract_rna_torsions("/tmp/this_file_does_not_exist.pdb") is None

def test_extract_rna_torsions_chain_not_found(tmp_path):
    # Use a real file but a bogus chain
    example = os.path.join(EXAMPLES_DIR, "1a34_1_B.cif")
    result = extract_rna_torsions(example, chain_id="Z")
    assert result is None or np.all(np.isnan(result)), "Should return None or all-nan array if chain not found"

def test_extract_rna_torsions_empty_file(tmp_path):
    empty = tmp_path / "empty.pdb"
    empty.write_text("")
    assert extract_rna_torsions(str(empty)) is None

@pytest.mark.parametrize("filename,chain_id,expected_shape", [
    ("1a34_1_B.cif", "B", (None, 7)),
    ("1a9n_1_R.cif", "R", (None, 7)),
    ("RNA_NET_1a9n_1_Q.cif", "Q", (None, 7)),
    ("synthetic_cppc_0000001.pdb", "B", (None, 7)),  # Use correct chain 'B' for this file
])
def test_extract_rna_torsions_smoke(filename, chain_id, expected_shape):
    path = os.path.join(EXAMPLES_DIR, filename)
    angles = extract_rna_torsions(path, chain_id=chain_id, backend="mdanalysis")
    print(f"\nDEBUG: {filename} shape={angles.shape}, dtype={angles.dtype}")
    print(f"DEBUG: angles=\n{angles}")
    assert angles is not None, f"Failed to extract angles for {filename}"
    assert angles.shape[-1] == 7
    assert angles.ndim in (1, 2)
    # Check that at least one angle per row is not nan (handle 1D and 2D, vectorized)
    if angles.ndim == 2:
        print(f"DEBUG: np.any(~np.isnan(angles), axis=1) = {np.any(~np.isnan(angles), axis=1)}")
        print(f"DEBUG: np.all(np.any(~np.isnan(angles), axis=1)) = {np.all(np.any(~np.isnan(angles), axis=1))}")
        assert np.all(np.any(~np.isnan(angles), axis=1)), f"All angles are nan in at least one row for {filename}"
    else:
        print(f"DEBUG: np.any(~np.isnan(angles)) = {np.any(~np.isnan(angles))}")
        assert np.any(~np.isnan(angles)), f"All angles are nan for {filename}"
    # Check angles are finite or nan
    print(f"DEBUG: np.isfinite(angles) =\n{np.isfinite(angles)}")
    print(f"DEBUG: np.isnan(angles) =\n{np.isnan(angles)}")
    print(f"DEBUG: np.isfinite(angles) | np.isnan(angles) =\n{np.isfinite(angles) | np.isnan(angles)}")
    assert np.all(np.isfinite(angles) | np.isnan(angles))
    # Check angles are in valid range
    valid = np.abs(angles[~np.isnan(angles)]) <= np.pi + 1e-3
    print(f"DEBUG: valid = {valid}")
    print(f"DEBUG: np.all(valid) = {np.all(valid)}")
    assert np.all(valid), f"Angles out of range for {filename}"

@given(
    chain_id=st.text(min_size=1, max_size=2),
    backend=st.sampled_from(["mdanalysis"]),
    bogus_path=st.text(min_size=1, max_size=30)
)
def test_extract_rna_torsions_property_invalid_inputs(chain_id, backend, bogus_path):
    # Should always return None or raise for invalid files
    result = extract_rna_torsions(bogus_path, chain_id=chain_id, backend=backend)
    assert result is None or (isinstance(result, np.ndarray) and result.size == 0)

# --- 100% coverage: error/fallback/edge-case tests ---
def test__load_universe_invalid(monkeypatch):
    import rna_predict.dataset.preprocessing.angles as angles
    # Simulate MDAnalysis failure
    class DummyUniverse:
        def __init__(self, *a, **kw):
            raise Exception("fail")
    monkeypatch.setattr(angles.mda, "Universe", DummyUniverse)
    assert angles._load_universe("badfile.pdb") is None

def test__select_chain_with_fallback_no_chain():
    from rna_predict.dataset.preprocessing.angles import _select_chain_with_fallback
    class DummyAtoms:
        chainIDs = []
        segids = []
    class DummyUniverse:
        atoms = DummyAtoms()
        def select_atoms(self, query):
            return DummyChain()
    class DummyChain:
        def __len__(self): return 0
    u = DummyUniverse()
    # Should return None when no chain and no nucleic atoms
    assert _select_chain_with_fallback(u, "Z") is None

def test__chi_torsion_missing_atoms():
    from rna_predict.dataset.preprocessing.angles import _chi_torsion
    class DummyRes:
        def __init__(self, atoms): self.atoms = atoms
    class DummyAtoms:
        def select_atoms(self, name): return DummySel()
    class DummySel:
        positions = []
        def __bool__(self): return False
        def __len__(self): return 0
    res = DummyRes(DummyAtoms())
    # All atoms missing, should return nan
    assert np.isnan(_chi_torsion(res))

def test_TempFileManager_context(monkeypatch, tmp_path):
    import rna_predict.dataset.preprocessing.angles as angles
    # Simulate .cif file conversion
    dummy_pdb = tmp_path / "dummy.pdb"
    dummy_pdb.write_text("ATOM\n")
    def fake_convert_cif_to_pdb(cif):
        return str(dummy_pdb)
    monkeypatch.setattr(angles, "_convert_cif_to_pdb", fake_convert_cif_to_pdb)
    mgr = angles.TempFileManager("foo.cif")
    with mgr as f:
        assert f == str(dummy_pdb)
    # After exit, file should be deleted
    assert not dummy_pdb.exists() or not mgr.using_temp

def test__convert_cif_to_pdb_invalid(monkeypatch, tmp_path):
    import rna_predict.dataset.preprocessing.angles as angles
    # Patch parser to throw
    monkeypatch.setattr(angles, "MMCIFParser", lambda *_, **__: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(Exception):
        angles._convert_cif_to_pdb("bad.cif")
