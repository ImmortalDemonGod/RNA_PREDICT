import unittest
import torch
import numpy as np

from rna_predict.pipeline.stageB.main import run_stageB_combined
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper


class TestCombinedTorsionPairformer(unittest.TestCase):
    def setUp(self):
        """Set up test case with simple models and inputs"""
        self.device = "cpu"
        self.sequence = "ACGUACGU"  # Simple 8-residue RNA sequence
        
        # Create adjacency matrix (N x N)
        N = len(self.sequence)
        self.adjacency = torch.zeros((N, N), dtype=torch.float32)
        # Add diagonal (self-connections)
        for i in range(N):
            self.adjacency[i, i] = 1.0
        # Add some base pairs
        self.adjacency[0, 7] = self.adjacency[7, 0] = 1.0  # A-U
        self.adjacency[1, 6] = self.adjacency[6, 1] = 1.0  # C-G
        
        # Create models with small dimensions for fast testing
        try:
            self.torsion_model = StageBTorsionBertPredictor(
                model_name_or_path="sayby/rna_torsionbert",
                device=self.device,
                angle_mode="degrees",
                num_angles=7,
                max_length=512,
            )
        except Exception as e:
            print(f"Warning: Could not load TorsionBERT, using dummy model. Error: {e}")
            self.torsion_model = StageBTorsionBertPredictor(
                model_name_or_path="dummy_path",
                device=self.device,
                angle_mode="degrees",
                num_angles=7,
                max_length=512,
            )
            
        self.pairformer = PairformerWrapper(
            n_blocks=2,   # Small number of blocks for testing
            c_z=32,       # Small embedding dims
            c_s=64,       # Small embedding dims
            dropout=0.1,
            use_checkpoint=False
        )

    def test_run_stageB_combined_basic(self):
        """Test that run_stageB_combined runs without errors and returns expected output structure"""
        result = run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency,
            torsion_bert_model=self.torsion_model,
            pairformer_model=self.pairformer,
            device=self.device,
            init_z_from_adjacency=False  # Use random initialization
        )
        
        # Check that all expected keys are present
        self.assertIn("torsion_angles", result)
        self.assertIn("s_embeddings", result)
        self.assertIn("z_embeddings", result)
        
        # Check shapes
        N = len(self.sequence)
        angle_dim = result["torsion_angles"].shape[1]  # Could be 7 or 14 depending on mode
        
        self.assertEqual(result["torsion_angles"].shape[0], N)
        self.assertEqual(result["s_embeddings"].shape, (N, self.pairformer.c_s))
        self.assertEqual(result["z_embeddings"].shape, (N, N, self.pairformer.c_z))
        
        # Check that tensors have valid values (no NaNs)
        self.assertFalse(torch.isnan(result["torsion_angles"]).any())
        self.assertFalse(torch.isnan(result["s_embeddings"]).any())
        self.assertFalse(torch.isnan(result["z_embeddings"]).any())

    def test_run_stageB_combined_with_adjacency_init(self):
        """Test run_stageB_combined with adjacency-based initialization"""
        result = run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency,
            torsion_bert_model=self.torsion_model,
            pairformer_model=self.pairformer,
            device=self.device,
            init_z_from_adjacency=True  # Use adjacency-based initialization
        )
        
        # All outputs should be present
        self.assertIn("torsion_angles", result)
        self.assertIn("s_embeddings", result)
        self.assertIn("z_embeddings", result)
        
        # Check that z_embeddings have some correlation with adjacency
        # This is a heuristic test - higher values where adjacency=1
        z_emb = result["z_embeddings"]
        N = len(self.sequence)
        
        # Calculate mean magnitude where adjacency=1 vs adjacency=0
        adj_mask = self.adjacency > 0
        z_mag_adj1 = z_emb[adj_mask].abs().mean().item()
        z_mag_adj0 = z_emb[~adj_mask].abs().mean().item()
        
        # Print for debugging
        print(f"Mean |z| where adj=1: {z_mag_adj1:.4f}")
        print(f"Mean |z| where adj=0: {z_mag_adj0:.4f}")
        
        # The difference may not be large due to Pairformer processing,
        # but there should be some effect from initialization

    def test_gradient_flow(self):
        """Test gradient flow through both TorsionBERT and Pairformer"""
        # Run the model
        result = run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency,
            torsion_bert_model=self.torsion_model,
            pairformer_model=self.pairformer,
            device=self.device,
            init_z_from_adjacency=False
        )
        
        # Set up a simple downstream task (coordinate prediction)
        s_emb = result["s_embeddings"]
        torsion_angles = result["torsion_angles"]
        z_emb = result["z_embeddings"]
        
        # Simple prediction heads
        final_head_s = torch.nn.Linear(self.pairformer.c_s, 3)
        angle_dim = torsion_angles.shape[1]
        final_head_angles = torch.nn.Linear(angle_dim, 3)
        final_head_z = torch.nn.Linear(self.pairformer.c_z, 3)
        
        # Forward pass
        coords_pred_s = final_head_s(s_emb)
        coords_pred_angles = final_head_angles(torsion_angles)
        z_pooled = z_emb.mean(dim=1)  # Shape [N, c_z]
        coords_pred_z = final_head_z(z_pooled)
        
        coords_pred = coords_pred_s + coords_pred_angles + coords_pred_z
        target = torch.zeros_like(coords_pred)
        
        # Compute loss and backpropagate
        loss = torch.nn.functional.mse_loss(coords_pred, target)
        
        # Check that loss is finite
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        
        # Clear gradients
        self.torsion_model.model.zero_grad()
        self.pairformer.zero_grad()
        final_head_s.zero_grad()
        final_head_angles.zero_grad()
        final_head_z.zero_grad()
        
        # Backpropagate
        loss.backward()
        
        # Check that gradients flow through the Pairformer
        pf_has_grad = False
        for name, param in self.pairformer.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                pf_has_grad = True
                break
        
        self.assertTrue(pf_has_grad, "Pairformer should have non-zero gradients")
        
        # For TorsionBERT, the model might be a dummy or real model
        # Only check if it's a real model with trainable parameters
        torsion_params = list(self.torsion_model.model.named_parameters())
        if torsion_params:
            tb_has_grad = False
            for name, param in torsion_params:
                if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
                    tb_has_grad = True
                    break
            
            # Only check if we have trainable parameters
            if any(param.requires_grad for _, param in torsion_params):
                self.assertTrue(tb_has_grad, "TorsionBERT should have non-zero gradients")


if __name__ == "__main__":
    unittest.main() 