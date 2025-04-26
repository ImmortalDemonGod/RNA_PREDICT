import torch
import pytest
from rna_predict.pipeline.stageC.mp_nerf.rna import rna_folding

def test_rna_fold_patch_seeds_initial_atoms():
    """
    Systematic test for the RNA folding patch:
    - Confirms the first 9 backbone atoms of residue 0 are nonzero (seeded).
    - Confirms all atoms in residues 1, 2, and 3 are nonzero and distinct.
    - Confirms no zero-propagation occurs.
    """
    # Minimal plausible scaffolds for 4 residues, 10 backbone atoms each
    L, B = 4, 10
    device = "cpu"
    dtype = torch.float32
    scaffolds = {
        "bond_mask": torch.ones((L, B), dtype=torch.bool, device=device),
        "angle_mask": torch.ones((L, B), dtype=torch.bool, device=device),
        "torsions": torch.randn((L, 7), dtype=dtype, device=device),
    }
    coords = rna_folding.rna_fold(scaffolds, device=device, do_ring_closure=False)
    # coords: shape [L, B, 3]
    assert coords.shape == (L, B, 3)
    # Check residue 0: first 9 atoms are nonzero
    for j in range(9):
        assert not torch.allclose(coords[0, j], torch.zeros(3, dtype=dtype)), f"Residue 0 atom {j} is zero!"
    # Check residue 1-3: all atoms are nonzero and distinct
    for i in range(1, 4):
        for j in range(B):
            assert not torch.allclose(coords[i, j], torch.zeros(3, dtype=dtype)), f"Residue {i} atom {j} is zero!"
    # Check that atoms in residues 1-3 are distinct from each other
    for i in range(1, 4):
        for j in range(B-1):
            assert not torch.allclose(coords[i, j], coords[i, j+1]), f"Residue {i} atoms {j} and {j+1} are identical!"
    # Optionally: print summary table
    print("\nSummary Table (After Patch):")
    for i in range(4):
        print(f"Residue {i}: ", [coords[i, j].tolist() for j in range(B)])
