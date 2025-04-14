import unittest
import torch
from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
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
        stageA = StageARFoldPredictor(stage_cfg=self.stageA_config, device=device)
        stageB = StageBTorsionBertPredictor(**self.stageB_config)

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
        stageB = StageBTorsionBertPredictor(**self.stageB_config)

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Stage B output
        stageB_out = stageB(sequence)
        torsion_angles = stageB_out["torsion_angles"]

        # Stage C expects exactly 7 angles, so slice if we have more
        if torsion_angles.shape[1] > 7:
            torsion_angles = torsion_angles[:, :7]

        # Stage C output
        stageC_out = run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device=self.device,
            do_ring_closure=False,
            place_bases=True,
            sugar_pucker="C3'-endo"
        )
        coords = stageC_out["coords"]

        # Verify Stage C output shape
        self.assertTrue(len(coords.shape) == 3)  # Should be [N, atoms, 3]
        self.assertEqual(coords.shape[0], self.seq_len)  # Number of residues
        self.assertEqual(coords.shape[2], 3)  # XYZ coordinates

    def test_stageC_to_D_dimensions(self):
        """Test dimension consistency between Stage C and D"""
        # Initialize models
        stageB = StageBTorsionBertPredictor(**self.stageB_config)

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Stage B output
        stageB_out = stageB(sequence)
        torsion_angles = stageB_out["torsion_angles"][:, :7]  # Take first 7 angles

        # Stage C output
        stageC_out = run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device=self.device,
            do_ring_closure=False,
            place_bases=True,
            sugar_pucker="C3'-endo"
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
        """Test dimension consistency through the entire pipeline"""
        # Initialize all models
        device = torch.device("cpu")
        stageA = StageARFoldPredictor(stage_cfg=self.stageA_config, device=device)
        stageB = StageBTorsionBertPredictor(**self.stageB_config)
        pairformer = PairformerWrapper(
            n_blocks=2,
            c_z=self.stageD_config["c_z"],
            c_s=self.stageD_config["c_s"],
            use_checkpoint=False
        )

        # Test sequence - ensure exact length
        sequence = "A" * self.seq_len

        # Run through pipeline
        adjacency = stageA.predict_adjacency(sequence)
        adjacency_t = torch.from_numpy(adjacency).float()

        stageB_out = stageB(sequence, adjacency_t)
        torsion_angles = stageB_out["torsion_angles"][:, :7]  # Take first 7 angles

        # Stage C output
        stageC_out = run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device=self.device,
            do_ring_closure=False,
            place_bases=True,
            sugar_pucker="C3'-endo"
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