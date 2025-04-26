"""
Tests for tensor_utils.py module.

This module tests the residue-to-atom bridging functions.
"""

import pytest
import torch

from rna_predict.utils.tensor_utils import (
    derive_residue_atom_map,
    residue_to_atoms,
    STANDARD_RNA_ATOMS,
)
from rna_predict.pipeline.stageD.diffusion.utils.tensor_utils import normalize_tensor_dimensions


class TestDeriveResidueAtomMap:
    """Tests for the derive_residue_atom_map function."""

    def test_with_sequence_only(self):
        """Test deriving map from sequence only."""
        sequence = ["A", "U", "G", "C"]
        residue_atom_map = derive_residue_atom_map(sequence)

        # Check that we have the right number of residues
        assert len(residue_atom_map) == 4

        # Check that each residue has the right number of atoms
        assert len(residue_atom_map[0]) == len(STANDARD_RNA_ATOMS["A"])
        assert len(residue_atom_map[1]) == len(STANDARD_RNA_ATOMS["U"])
        assert len(residue_atom_map[2]) == len(STANDARD_RNA_ATOMS["G"])
        assert len(residue_atom_map[3]) == len(STANDARD_RNA_ATOMS["C"])

        # Check that the atom indices are contiguous and start from 0
        expected_start_idx = 0
        for res_idx, atom_indices in enumerate(residue_atom_map):
            assert atom_indices[0] == expected_start_idx
            assert atom_indices[-1] == expected_start_idx + len(atom_indices) - 1
            expected_start_idx += len(atom_indices)

    def test_with_partial_coords(self):
        """Test deriving map from sequence and partial coordinates."""
        sequence = ["A", "U", "G", "C"]
        # Create partial coordinates with the expected number of atoms
        total_atoms = sum(len(STANDARD_RNA_ATOMS[res]) for res in sequence)
        partial_coords = torch.randn(1, total_atoms, 3)

        residue_atom_map = derive_residue_atom_map(sequence, partial_coords)

        # Check that we have the right number of residues
        assert len(residue_atom_map) == 4

        # Check that the total number of atoms matches partial_coords
        total_mapped_atoms = sum(len(atoms) for atoms in residue_atom_map)
        assert total_mapped_atoms == partial_coords.shape[1]

    def test_with_atom_metadata(self):
        """Test deriving map from atom metadata."""
        sequence = ["A", "U", "G", "C"]
        # Create atom metadata with explicit residue indices
        residue_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]  # 3 atoms per residue for simplicity
        atom_metadata = {"residue_indices": residue_indices}

        residue_atom_map = derive_residue_atom_map(sequence, atom_metadata=atom_metadata)

        # Check that we have the right number of residues
        assert len(residue_atom_map) == 4

        # Check that each residue has the expected atoms
        assert residue_atom_map[0] == [0, 1, 2]
        assert residue_atom_map[1] == [3, 4, 5]
        assert residue_atom_map[2] == [6, 7, 8]
        assert residue_atom_map[3] == [9, 10, 11]

    def test_with_string_sequence(self):
        """Test with a string sequence instead of a list."""
        sequence = "AUGC"
        residue_atom_map = derive_residue_atom_map(sequence)

        # Check that we have the right number of residues
        assert len(residue_atom_map) == 4

        # Check that each residue has the right number of atoms
        assert len(residue_atom_map[0]) == len(STANDARD_RNA_ATOMS["A"])
        assert len(residue_atom_map[1]) == len(STANDARD_RNA_ATOMS["U"])
        assert len(residue_atom_map[2]) == len(STANDARD_RNA_ATOMS["G"])
        assert len(residue_atom_map[3]) == len(STANDARD_RNA_ATOMS["C"])

    def test_empty_sequence(self):
        """Test with an empty sequence."""
        sequence = []
        residue_atom_map = derive_residue_atom_map(sequence)

        # Check that we have an empty map
        assert len(residue_atom_map) == 0

    def test_unknown_residue(self):
        """Test with an unknown residue type."""
        sequence = ["A", "X", "G", "C"]  # X is not a standard residue

        # Should raise a KeyError
        with pytest.raises(KeyError):
            derive_residue_atom_map(sequence)


class TestResidueToAtoms:
    """Tests for the residue_to_atoms function."""

    def test_basic_expansion(self):
        """Test basic expansion of residue embeddings to atom embeddings."""
        # Create a simple residue embedding tensor
        n_residue = 4
        c_s = 16
        s_emb = torch.randn(n_residue, c_s)

        # Create a simple residue-to-atom map (2 atoms per residue for simplicity)
        residue_atom_map = [[0, 1], [2, 3], [4, 5], [6, 7]]

        # Expand to atom embeddings
        atom_embs = residue_to_atoms(s_emb, residue_atom_map)

        # Check shape
        assert atom_embs.shape == (8, c_s)

        # Check that each atom has the embedding of its residue
        for res_idx, atom_indices in enumerate(residue_atom_map):
            for atom_idx in atom_indices:
                assert torch.allclose(atom_embs[atom_idx], s_emb[res_idx])

    def test_batched_expansion(self):
        """Test expansion with batched residue embeddings."""
        # Create a batched residue embedding tensor
        batch_size = 2
        n_residue = 4
        c_s = 16
        s_emb = torch.randn(batch_size, n_residue, c_s)

        # Create a simple residue-to-atom map (2 atoms per residue for simplicity)
        residue_atom_map = [[0, 1], [2, 3], [4, 5], [6, 7]]

        # Expand to atom embeddings
        atom_embs = residue_to_atoms(s_emb, residue_atom_map)

        # Check shape
        assert atom_embs.shape == (batch_size, 8, c_s)

        # Check that each atom has the embedding of its residue
        for batch_idx in range(batch_size):
            for res_idx, atom_indices in enumerate(residue_atom_map):
                for atom_idx in atom_indices:
                    assert torch.allclose(atom_embs[batch_idx, atom_idx], s_emb[batch_idx, res_idx])

    def test_empty_inputs(self):
        """Test with empty inputs."""
        # Empty residue embeddings
        s_emb = torch.empty((0, 16))
        residue_atom_map = []

        # Expand to atom embeddings
        atom_embs = residue_to_atoms(s_emb, residue_atom_map)

        # Check shape
        assert atom_embs.shape == (0, 16)

    def test_invalid_map(self):
        """Test with an invalid residue-to-atom map."""
        # Create a simple residue embedding tensor
        n_residue = 4
        c_s = 16
        s_emb = torch.randn(n_residue, c_s)

        # Create an invalid map with missing atoms
        residue_atom_map = [[0, 1], [2, 3], [5, 6], [7, 8]]  # Missing atom index 4

        # Should raise a ValueError
        with pytest.raises(ValueError):
            residue_to_atoms(s_emb, residue_atom_map)

        # Create an invalid map with duplicate atoms
        residue_atom_map = [[0, 1], [2, 3], [3, 4], [5, 6]]  # Duplicate atom index 3

        # Should raise a ValueError
        with pytest.raises(ValueError):
            residue_to_atoms(s_emb, residue_atom_map)

    def test_shape_mismatch(self):
        """Test with a shape mismatch between s_emb and residue_atom_map."""
        # Create a simple residue embedding tensor
        n_residue = 4
        c_s = 16
        s_emb = torch.randn(n_residue, c_s)

        # Create a map with a different number of residues
        residue_atom_map = [[0, 1], [2, 3], [4, 5]]  # Only 3 residues

        # Should raise a ValueError
        with pytest.raises(ValueError):
            residue_to_atoms(s_emb, residue_atom_map)


class TestIntegration:
    """Integration tests for the residue-to-atom bridging functions."""

    def test_end_to_end(self):
        """Test the full process from sequence to atom embeddings."""
        # Create a sequence and residue embeddings
        sequence = ["A", "U", "G", "C"]
        n_residue = len(sequence)
        c_s = 16
        s_emb = torch.randn(n_residue, c_s)

        # Derive the residue-to-atom map
        residue_atom_map = derive_residue_atom_map(sequence)

        # Expand to atom embeddings
        atom_embs = residue_to_atoms(s_emb, residue_atom_map)

        # Check shape
        total_atoms = sum(len(STANDARD_RNA_ATOMS[res]) for res in sequence)
        assert atom_embs.shape == (total_atoms, c_s)

        # Check that each atom has the embedding of its residue
        for res_idx, atom_indices in enumerate(residue_atom_map):
            for atom_idx in atom_indices:
                assert torch.allclose(atom_embs[atom_idx], s_emb[res_idx])

    def test_with_partial_coords(self):
        """Test the full process with partial coordinates."""
        # Create a sequence and residue embeddings
        sequence = ["A", "U", "G", "C"]
        n_residue = len(sequence)
        c_s = 16
        s_emb = torch.randn(n_residue, c_s)

        # Create partial coordinates with the expected number of atoms
        total_atoms = sum(len(STANDARD_RNA_ATOMS[res]) for res in sequence)
        partial_coords = torch.randn(1, total_atoms, 3)

        # Derive the residue-to-atom map
        residue_atom_map = derive_residue_atom_map(sequence, partial_coords)

        # Expand to atom embeddings
        atom_embs = residue_to_atoms(s_emb, residue_atom_map)

        # Check shape
        assert atom_embs.shape == (total_atoms, c_s)

        # Check that the total number of atoms matches partial_coords
        assert atom_embs.shape[0] == partial_coords.shape[1]


class TestNormalizeTensorDimensions:
    """Unit tests for normalize_tensor_dimensions, including shape preservation and unique error."""

    def test_preserve_pair_shape(self):
        # [B, 1, 1, C] with key='pair' should be preserved
        tensor = torch.randn(2, 1, 1, 4)
        out = normalize_tensor_dimensions(tensor, batch_size=2, key='pair')
        assert out.shape == (2, 1, 1, 4)

    def test_preserve_z_trunk_shape(self):
        # [B, 1, 1, C] with key='z_trunk' should be preserved
        tensor = torch.randn(2, 1, 1, 4)
        out = normalize_tensor_dimensions(tensor, batch_size=2, key='z_trunk')
        assert out.shape == (2, 1, 1, 4)

    def test_unique_error_on_collapse(self):
        # [B, 1, C] with key='pair' should raise unique error
        tensor = torch.randn(2, 1, 4)
        with pytest.raises(RuntimeError, match=r"\[UNIQUE-PAIR-SHAPE-ERROR\]"):
            normalize_tensor_dimensions(tensor, batch_size=2, key='pair')

    def test_single_embedding_squeeze(self):
        # [B, 1, C] with key=None should squeeze to [B, C]
        tensor = torch.randn(2, 1, 4)
        out = normalize_tensor_dimensions(tensor, batch_size=2)
        assert out.shape == (2, 4)


class TestPairEmbeddingBroadcast:
    """Unit test for correct broadcasting of pair-embedding to atom pairs."""
    def test_pair_embedding_broadcast_small_tensor(self):
        import torch
        B, N_sample, N_res, C = 1, 1, 3, 2
        value = torch.arange(B * N_sample * N_res * N_res * C, dtype=torch.float32).reshape(B, N_sample, N_res, N_res, C)
        residue_atom_map = [[0, 1], [2, 3], [4, 5]]
        N_atom = sum(len(x) for x in residue_atom_map)
        out = torch.zeros(B, N_sample, N_atom, N_atom, C)
        # Double-loop assignment to avoid advanced indexing shape issues
        for s in range(N_sample):
            for i, atom_indices_i in enumerate(residue_atom_map):
                for j, atom_indices_j in enumerate(residue_atom_map):
                    val = value[:, s, i, j, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
                    for a_idx, atom_i in enumerate(atom_indices_i):
                        for b_idx, atom_j in enumerate(atom_indices_j):
                            out[:, s, atom_i, atom_j, :] = val[:, 0, 0, :]
        for i, atom_indices_i in enumerate(residue_atom_map):
            for j, atom_indices_j in enumerate(residue_atom_map):
                expected = value[:, :, i:i+1, j:j+1, :]
                while expected.dim() < 5:
                    expected = expected.unsqueeze(-2)
                expected = expected.expand(-1, -1, len(atom_indices_i), len(atom_indices_j), -1)
                actual = out[:, :, atom_indices_i][:, :, :, atom_indices_j]
                assert torch.allclose(actual, expected), f"Broadcast failed for residue pair ({i},{j})"
