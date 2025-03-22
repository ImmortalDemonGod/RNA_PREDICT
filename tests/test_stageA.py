import json
import os
import tempfile
import unittest

import numpy as np
import torch

from rna_predict.pipeline.stageA.run_stageA import run_stageA
from rna_predict.pipeline.stageA.rfold_predictor import StageARFoldPredictor


class TestStageARFoldPredictor(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file with dummy model parameters
        self.temp_config = tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        )
        config = {"model_params": {"input_dim": 4, "hidden_dim": 8, "output_dim": 4}}
        json.dump(config, self.temp_config)
        self.temp_config.close()

        # Create a temporary checkpoint file with a dummy state dict
        self.temp_checkpoint = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        torch.save({}, self.temp_checkpoint.name)
        self.temp_checkpoint.close()

        self.predictor = StageARFoldPredictor(
            self.temp_config.name, self.temp_checkpoint.name, device=torch.device("cpu")
        )

    def tearDown(self):
        os.remove(self.temp_config.name)
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
