"""
Test script for the fixed initialize_features_from_config function.
This script tests the fix for the tensor size mismatch issue in Stage D.
"""
import torch
import logging
import sys
from omegaconf import DictConfig, OmegaConf

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
log = logging.getLogger(__name__)

# Import the function we want to test
from rna_predict.pipeline.stageD.run_stageD import initialize_features_from_config

def create_test_config():
    """Create a test configuration with the necessary settings."""
    config_dict = {
        "model": {
            "stageD": {
                "input_features": {
                    "atom_to_token_idx": {
                        "repeats": 44  # Standard RNA residue has ~44 atoms
                    },
                    "profile": {
                        "size": [32]  # Example size for profile embeddings
                    }
                }
            }
        }
    }
    return OmegaConf.create(config_dict)

def test_with_small_atom_count():
    """Test the function with a small number of atoms (less than atoms_per_residue)."""
    # Create a small test tensor with only 21 atoms
    batch_size = 1
    num_atoms = 21  # Less than the standard 44 atoms per residue
    coords = torch.randn(batch_size, num_atoms, 3)
    
    # Create test config
    cfg = create_test_config()
    
    # Call the function
    log.info(f"Testing initialize_features_from_config with {num_atoms} atoms")
    features = initialize_features_from_config(cfg, coords)
    
    # Check the results
    log.info(f"atom_to_token_idx shape: {features['atom_to_token_idx'].shape}")
    log.info(f"atom_to_token_idx content: {features['atom_to_token_idx']}")
    log.info(f"restype shape: {features['restype'].shape}")
    
    # Verify that atom_to_token_idx has the correct shape
    assert features["atom_to_token_idx"].shape == (batch_size, num_atoms), \
        f"Expected shape {(batch_size, num_atoms)}, got {features['atom_to_token_idx'].shape}"
    
    # Verify that all atoms are mapped to residue 0
    assert (features["atom_to_token_idx"] == 0).all(), \
        "Expected all atoms to be mapped to residue 0"
    
    log.info("Test passed!")
    return features

def test_with_normal_atom_count():
    """Test the function with a normal number of atoms (more than atoms_per_residue)."""
    # Create a test tensor with 88 atoms (2 residues * 44 atoms)
    batch_size = 1
    num_atoms = 88  # 2 residues * 44 atoms
    coords = torch.randn(batch_size, num_atoms, 3)
    
    # Create test config
    cfg = create_test_config()
    
    # Call the function
    log.info(f"Testing initialize_features_from_config with {num_atoms} atoms")
    features = initialize_features_from_config(cfg, coords)
    
    # Check the results
    log.info(f"atom_to_token_idx shape: {features['atom_to_token_idx'].shape}")
    log.info(f"atom_to_token_idx unique values: {torch.unique(features['atom_to_token_idx'])}")
    log.info(f"restype shape: {features['restype'].shape}")
    
    # Verify that atom_to_token_idx has the correct shape
    assert features["atom_to_token_idx"].shape == (batch_size, num_atoms), \
        f"Expected shape {(batch_size, num_atoms)}, got {features['atom_to_token_idx'].shape}"
    
    # Verify that atoms are mapped to residues 0 and 1
    assert set(torch.unique(features["atom_to_token_idx"]).tolist()) == {0, 1}, \
        "Expected atoms to be mapped to residues 0 and 1"
    
    log.info("Test passed!")
    return features

def test_with_non_divisible_atom_count():
    """Test the function with a non-divisible number of atoms."""
    # Create a test tensor with 100 atoms (not divisible by 44)
    batch_size = 1
    num_atoms = 100  # Not divisible by 44
    coords = torch.randn(batch_size, num_atoms, 3)
    
    # Create test config
    cfg = create_test_config()
    
    # Call the function
    log.info(f"Testing initialize_features_from_config with {num_atoms} atoms")
    features = initialize_features_from_config(cfg, coords)
    
    # Check the results
    log.info(f"atom_to_token_idx shape: {features['atom_to_token_idx'].shape}")
    log.info(f"atom_to_token_idx unique values: {torch.unique(features['atom_to_token_idx'])}")
    log.info(f"restype shape: {features['restype'].shape}")
    
    # Verify that atom_to_token_idx has the correct shape
    assert features["atom_to_token_idx"].shape == (batch_size, num_atoms), \
        f"Expected shape {(batch_size, num_atoms)}, got {features['atom_to_token_idx'].shape}"
    
    # Verify that the number of unique residue indices is correct
    num_residues = max(1, num_atoms // 44)
    assert len(torch.unique(features["atom_to_token_idx"])) == num_residues, \
        f"Expected {num_residues} unique residue indices, got {len(torch.unique(features['atom_to_token_idx']))}"
    
    log.info("Test passed!")
    return features

if __name__ == "__main__":
    try:
        # Run the tests
        test_with_small_atom_count()
        test_with_normal_atom_count()
        test_with_non_divisible_atom_count()
        
        log.info("All tests passed!")
    except Exception as e:
        log.error(f"Test failed: {str(e)}")
        sys.exit(1)
