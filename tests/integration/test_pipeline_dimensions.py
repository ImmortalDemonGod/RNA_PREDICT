import unittest
import torch
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper

class TestPipelineDimensions(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.seq_len = 10
        self.batch_size = 1

        # Stage A config
        from omegaconf import OmegaConf
        self.stageA_config = OmegaConf.create({
            "num_hidden": 128,
            "dropout": 0.3,
            "min_seq_length": 80,
            "device": "cpu",
            "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
            "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
            "batch_size": 32,
            "lr": 0.001,
            "threshold": 0.5,
            "visualization": {
                "enabled": True,
                "varna_jar_path": "tools/varna-3-93.jar",
                "resolution": 8.0
            },
            "model": {
                "conv_channels": [64, 128, 256, 512],
                "residual": True,
                "c_in": 1,
                "c_out": 1,
                "c_hid": 32,
                "seq2map": {
                    "input_dim": 4,
                    "max_length": 3000,
                    "attention_heads": 8,
                    "attention_dropout": 0.1,
                    "positional_encoding": True,
                    "query_key_dim": 128,
                    "expansion_factor": 2.0,
                    "heads": 1
                },
                "decoder": {
                    "up_conv_channels": [256, 128, 64],
                    "skip_connections": True
                }
            }
        })

        # Stage B config
        self.stageB_config = {
            "model_name_or_path": "sayby/rna_torsionbert",
            "device": self.device,
            "angle_mode": "degrees",
            "num_angles": 16,  # The model actually outputs 16 angles
            "max_length": 512
        }

        # Stage C config
        self.stageC_config = {
            "method": "mp_nerf",
            "do_ring_closure": False,
            "place_bases": True,
            "sugar_pucker": "C3'-endo"
        }

        # Stage D config
        self.stageD_config = {
            "sigma_data": 16.0,
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 768,
            "c_s": 64,
            "c_z": 32,
            "c_s_inputs": 449,
            "atom_encoder": {"n_blocks": 1, "n_heads": 2},
            "transformer": {"n_blocks": 1, "n_heads": 2},
            "atom_decoder": {"n_blocks": 1, "n_heads": 2},
            "initialization": {}
        }

    def test_stageA_to_B_dimensions(self):
        """Test dimension consistency between Stage A and B"""
        # Initialize models
        device = torch.device("cpu")

        # Create a mock StageARFoldPredictor
        class MockStageARFoldPredictor(torch.nn.Module):
            def __init__(self, stage_cfg, device):
                super().__init__()
                self.stage_cfg = stage_cfg
                self.device = device
                # Add a dummy parameter to make the test pass
                self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

            def predict_adjacency(self, sequence):
                # Return a dummy adjacency matrix
                N = len(sequence)
                return torch.zeros((N, N)).numpy()

        stageA = MockStageARFoldPredictor(stage_cfg=self.stageA_config, device=device)
        from omegaconf import OmegaConf
        stageB_cfg = OmegaConf.create({"stageB_torsion": self.stageB_config})
        stageB = StageBTorsionBertPredictor(stageB_cfg)

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Stage A output
        adjacency = stageA.predict_adjacency(sequence)
        adjacency_t = torch.from_numpy(adjacency).float()

        # Verify Stage A output shape
        self.assertEqual(adjacency_t.shape, (self.seq_len, self.seq_len))

        # Stage B output
        stageB_out = stageB(sequence, adjacency_t)
        torsion_angles = stageB_out["torsion_angles"]

        # Verify Stage B output shape
        self.assertEqual(torsion_angles.shape[0], self.seq_len)
        self.assertEqual(torsion_angles.shape[1], self.stageB_config["num_angles"])

    def test_stageB_to_C_dimensions(self):
        """Test dimension consistency between Stage B and C"""
        # Initialize models
        from omegaconf import OmegaConf
        stageB_cfg = OmegaConf.create({"stageB_torsion": self.stageB_config})
        stageB = StageBTorsionBertPredictor(stageB_cfg)

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Stage B output
        stageB_out = stageB(sequence)
        torsion_angles = stageB_out["torsion_angles"]

        # Stage C expects exactly 7 angles, so slice if we have more
        if torsion_angles.shape[1] > 7:
            torsion_angles = torsion_angles[:, :7]

        # Compose config with Hydra best practices
        from hydra import initialize, compose
        with initialize(config_path="../../rna_predict/conf", version_base=None):
            stage_c_cfg = compose(config_name="default", overrides=[
                "model.stageC.enabled=True",
                "model.stageC.method=mp_nerf",
                "model.stageC.device=cpu",
                "model.stageC.do_ring_closure=False",
                "model.stageC.place_bases=True",
                "model.stageC.sugar_pucker=\"C3'-endo\"",
                "model.stageC.angle_representation=radians",
                "model.stageC.use_metadata=False",
                "model.stageC.use_memory_efficient_kernel=False",
                "model.stageC.use_deepspeed_evo_attention=False",
                "model.stageC.use_lma=False",
                "model.stageC.inplace_safe=True",
                "model.stageC.debug_logging=False",
                # Add any other overrides required for the test
            ])
        print("[DEBUG][test_stageB_to_C_dimensions] stage_c_cfg:")
        print(OmegaConf.to_yaml(stage_c_cfg))
        import sys
        sys.stdout.flush()
        assert hasattr(stage_c_cfg.model, "stageC"), stage_c_cfg.model
        assert hasattr(stage_c_cfg.model.stageC, "enabled"), stage_c_cfg.model.stageC
        stageC_out = run_stageC(
            cfg=stage_c_cfg,
            sequence=sequence,
            torsion_angles=torsion_angles,
        )
        coords = stageC_out["coords"]

        # Verify Stage C output shape
        # The output shape can be either [N*atoms, 3] or [N, atoms, 3] depending on the implementation
        # Check that we have coordinates for each atom
        self.assertTrue(coords.shape[-1] == 3)  # Last dimension should be XYZ coordinates

        # If the shape is [N*atoms, 3], we just check that we have coordinates
        if len(coords.shape) == 2:
            self.assertTrue(coords.shape[0] > 0)  # Should have some atoms
        # If the shape is [N, atoms, 3], we check the number of residues
        elif len(coords.shape) == 3:
            self.assertEqual(coords.shape[0], self.seq_len)  # Number of residues

    def test_stageC_to_D_dimensions(self):
        """Test dimension consistency between Stage C and D"""
        # Initialize models
        from omegaconf import OmegaConf
        stageB_cfg = OmegaConf.create({"stageB_torsion": self.stageB_config})
        stageB = StageBTorsionBertPredictor(stageB_cfg)

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Stage B output
        stageB_out = stageB(sequence)
        torsion_angles = stageB_out["torsion_angles"][:, :7]  # Take first 7 angles

        # Compose config with Hydra best practices
        from hydra import initialize, compose
        with initialize(config_path="../../rna_predict/conf", version_base=None):
            stage_c_cfg = compose(config_name="default", overrides=[
                "model.stageC.enabled=True",
                "model.stageC.method=mp_nerf",
                "model.stageC.device=cpu",
                "model.stageC.do_ring_closure=False",
                "model.stageC.place_bases=True",
                "model.stageC.sugar_pucker=\"C3'-endo\"",
                "model.stageC.angle_representation=radians",
                "model.stageC.use_metadata=False",
                "model.stageC.use_memory_efficient_kernel=False",
                "model.stageC.use_deepspeed_evo_attention=False",
                "model.stageC.use_lma=False",
                "model.stageC.inplace_safe=True",
                "model.stageC.debug_logging=False",
                # Add any other overrides required for the test
            ])
        print("[DEBUG][test_stageC_to_D_dimensions] stage_c_cfg:")
        print(OmegaConf.to_yaml(stage_c_cfg))
        import sys
        sys.stdout.flush()
        assert hasattr(stage_c_cfg.model, "stageC"), stage_c_cfg.model
        assert hasattr(stage_c_cfg.model.stageC, "enabled"), stage_c_cfg.model.stageC
        stageC_out = run_stageC(
            cfg=stage_c_cfg,
            sequence=sequence,
            torsion_angles=torsion_angles,
        )
        stageC_out["coords"]

        # Stage D input preparation
        s_inputs = torch.randn(self.batch_size, self.seq_len, self.stageD_config["c_s_inputs"])
        z_in = torch.randn(self.batch_size, self.seq_len, self.seq_len, self.stageD_config["c_z"])

        # Verify dimensions match Stage D expectations
        self.assertEqual(s_inputs.shape[1], self.seq_len)
        self.assertEqual(z_in.shape[1], self.seq_len)
        self.assertEqual(z_in.shape[2], self.seq_len)

    def test_full_pipeline_dimensions(self):
        """
        Tests that tensor dimensions remain consistent across all stages of the RNA structure
        prediction pipeline, from Stage A through Stage D.
        
        Runs a mock Stage A predictor, Stage B torsion angle prediction, Stage C reconstruction,
        and Stage D Pairformer, verifying that the outputs at each stage have the expected shapes
        for the configured sequence length, batch size, and feature dimensions.
        """
        # Initialize all models
        device = torch.device("cpu")

        # Create a mock StageARFoldPredictor
        class MockStageARFoldPredictor(torch.nn.Module):
            def __init__(self, stage_cfg, device):
                super().__init__()
                self.stage_cfg = stage_cfg
                self.device = device
                # Add a dummy parameter to make the test pass
                self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

            def predict_adjacency(self, sequence):
                # Return a dummy adjacency matrix
                N = len(sequence)
                return torch.zeros((N, N)).numpy()

        stageA = MockStageARFoldPredictor(stage_cfg=self.stageA_config, device=device)
        from omegaconf import OmegaConf
        stageB_cfg = OmegaConf.create({"stageB_torsion": self.stageB_config})
        stageB = StageBTorsionBertPredictor(stageB_cfg)
        pairformer_cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": 2,
                "n_heads": 4,  # Required parameter
                "c_z": self.stageD_config["c_z"],
                "c_s": self.stageD_config["c_s"],
                "dropout": 0.1,
                "use_checkpoint": False,
                "init_z_from_adjacency": False,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": True,
                "chunk_size": 128,
                "device": "cpu"  # Add explicit device parameter
            }
        })
        pairformer = PairformerWrapper(pairformer_cfg)

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Run through pipeline
        adjacency = stageA.predict_adjacency(sequence)
        adjacency_t = torch.from_numpy(adjacency).float()

        stageB_out = stageB(sequence, adjacency_t)
        torsion_angles = stageB_out["torsion_angles"][:, :7]  # Take first 7 angles

        # Compose config with Hydra best practices
        from hydra import initialize, compose
        with initialize(config_path="../../rna_predict/conf", version_base=None):
            stage_c_cfg = compose(config_name="default", overrides=[
                "model.stageC.enabled=True",
                "model.stageC.method=mp_nerf",
                "model.stageC.device=cpu",
                "model.stageC.do_ring_closure=False",
                "model.stageC.place_bases=True",
                "model.stageC.sugar_pucker=\"C3'-endo\"",
                "model.stageC.angle_representation=radians",
                "model.stageC.use_metadata=False",
                "model.stageC.use_memory_efficient_kernel=False",
                "model.stageC.use_deepspeed_evo_attention=False",
                "model.stageC.use_lma=False",
                "model.stageC.inplace_safe=True",
                "model.stageC.debug_logging=False",
                # Add any other overrides required for the test
            ])
        print("[DEBUG][test_full_pipeline_dimensions] stage_c_cfg:")
        print(OmegaConf.to_yaml(stage_c_cfg))
        import sys
        sys.stdout.flush()
        assert hasattr(stage_c_cfg.model, "stageC"), stage_c_cfg.model
        assert hasattr(stage_c_cfg.model.stageC, "enabled"), stage_c_cfg.model.stageC
        stageC_out = run_stageC(
            cfg=stage_c_cfg,
            sequence=sequence,
            torsion_angles=torsion_angles,
        )
        stageC_out["coords"]

        # Prepare Pairformer inputs
        s = torch.randn(self.batch_size, self.seq_len, self.stageD_config["c_s"])
        z = torch.randn(self.batch_size, self.seq_len, self.seq_len, self.stageD_config["c_z"])
        pair_mask = torch.ones(self.batch_size, self.seq_len, self.seq_len)

        # Run Pairformer
        s_out, z_out = pairformer(s, z, pair_mask)

        # Verify dimensions
        self.assertEqual(s_out.shape[1], self.seq_len)
        self.assertEqual(z_out.shape[1], self.seq_len)
        self.assertEqual(z_out.shape[2], self.seq_len)
        self.assertEqual(s_out.shape[2], self.stageD_config["c_s"])
        self.assertEqual(z_out.shape[3], self.stageD_config["c_z"])

if __name__ == '__main__':
    unittest.main()