import os
import tempfile
import unittest

import numpy as np
import torch

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
from rna_predict.pipeline.stageA.run_stageA import run_stageA


class TestStageARFoldPredictor(unittest.TestCase):
    def setUp(self):
        # Create a config object compatible with the new StageARFoldPredictor
        from omegaconf import OmegaConf

        self.stage_cfg = OmegaConf.create({
            "num_hidden": 128,
            "dropout": 0.3,
            "min_seq_length": 80,
            "checkpoint_path": None,  # We'll create a dummy checkpoint
            "checkpoint_url": None,
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

        # Create a temporary checkpoint file with a dummy state dict
        self.temp_checkpoint = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        torch.save({}, self.temp_checkpoint.name)
        self.temp_checkpoint.close()

        # Update the config with the checkpoint path
        self.stage_cfg.checkpoint_path = self.temp_checkpoint.name

        # Create the predictor with the new parameters
        device = torch.device("cpu")
        self.predictor = StageARFoldPredictor(stage_cfg=self.stage_cfg, device=device)

    def tearDown(self):
        # Only remove the checkpoint file
        os.remove(self.temp_checkpoint.name)

    def test_short_sequence(self):
        # For sequences shorter than 4 nucleotides, the adjacency matrix should be all zeros
        seq = "ACG"
        adjacency = self.predictor.predict_adjacency(seq)
        self.assertTrue((adjacency == 0).all())

    def test_run_stageA_integration(self):
        seq = "ACGUACGU"
        adjacency = run_stageA(seq, self.predictor)
        self.assertEqual(adjacency.shape[0], len(seq))
        self.assertEqual(adjacency.shape[1], len(seq))
        self.assertTrue(np.isin(adjacency, [0, 1]).all())
        row_sums = adjacency.sum(axis=1)
        self.assertTrue((row_sums <= 1).all())


if __name__ == "__main__":
    unittest.main()
