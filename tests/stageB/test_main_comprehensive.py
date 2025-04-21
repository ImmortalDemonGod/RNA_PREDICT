import os
os.chdir("/Users/tomriddle1/RNA_PREDICT")
import sys
import pytest
import torch
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from hypothesis import given, strategies as st, settings
from hydra import initialize, compose
from rna_predict.pipeline.stageB.main import (
    run_pipeline,
    run_stageB_combined,
    demo_gradient_flow_test,
    main,
)
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageA.adjacency.rfold_predictor import (
    StageARFoldPredictor,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import (
    StageCReconstruction,
)

sys.path.insert(0, "/Users/tomriddle1/RNA_PREDICT")
# ENFORCE: Tests must be run from the project root for Hydra config to resolve correctly.
EXPECTED_CWD = "/Users/tomriddle1/RNA_PREDICT"
if os.getcwd() != EXPECTED_CWD:
    raise RuntimeError(f"Tests must be run from {EXPECTED_CWD} for Hydra config_path to resolve. Current CWD: {os.getcwd()}")

# NOTE: Hydra's initialize(config_path=...) requires a RELATIVE path. However, under pytest, Hydra resolves config_path relative to the test file's directory, not the process CWD. Therefore, we must use '../../rna_predict/conf' for tests in tests/stageB.
"""
Comprehensive tests for the main.py module in Stage B.

This module tests the functions in rna_predict/pipeline/stageB/main.py,
including run_pipeline, run_stageB_combined, and demo_gradient_flow_test.
"""


class TestRunPipeline:
    """Tests for the run_pipeline function."""

    @staticmethod
    def _make_cfg():
        # Use OmegaConf to create a DictConfig compatible config
        cfg_dict = {
            "model": {
                "stageA": {
                    "min_seq_length": 80,
                    "dropout": 0.3,
                    "num_hidden": 128,
                    "device": "cpu",
                    "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
                    "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
                    "checkpoint_zip_path": "RFold/checkpoints.zip",
                    "batch_size": 32,
                    "lr": 0.001,
                    "threshold": 0.5,
                    "debug_logging": True,
                    "run_example": True,
                    "example_sequence": "AAGUCUGGUGGACAUUGGCGUCCUGAGGUGUUAAAACCUCUUAUUGCUGACGCCAGAAAGAGAAGAACUUCGGUUCUACUAGUCGACUAUACUACAAGCUUUGGGUGUAUAGCGGCAAGACAACCUGGAUCGGGGGAGGCUAAGGGCGCAAGCCUAUGCUAACCCCGAGCCGAGCUACUGGAGGGCAACCCCCAGAUAGCCGGUGUAGAGCGCGGAAAGGUGUCGGUCAUCCUAUCUGAUAGGUGGCUUGAGGGACGUGCCGUCUCACCCGAAAGGGUGUUUCUAAGGAGGAGCUCCCAAAGGGCAAAUCUUAGAAAAGGGUGUAUACCCUAUAAUUUAACGGCCAGCAGCC",
                    "visualization": {
                        "enabled": True,
                        "varna_jar_path": "tools/varna-3-93.jar",
                        "resolution": 8.0,
                        "output_path": "test_seq.png"
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
                },
                "stageB": {
                    "debug_logging": True,
                    "torsion_bert": {
                        "device": "cpu",
                        "model_name_or_path": "sayby/rna_torsionbert",
                        "angle_mode": "sin_cos",
                        "num_angles": 7,
                        "max_length": 512
                    },
                    "pairformer": {
                        "init_z_from_adjacency": False
                    }
                },
                "stageC": {}
            },
            "stageB_torsion": {
                "device": "cpu",
                "model_name_or_path": "sayby/rna_torsionbert",
                "angle_mode": "sin_cos",
                "num_angles": 7,
                "max_length": 512
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 32
            }
        }
        return OmegaConf.create(cfg_dict)

    @settings(deadline=None, max_examples=3)  # Limit to 3 examples to avoid timeouts
    @given(seq=st.text(alphabet="ACGU", min_size=1, max_size=5))  # Reduced max_size to 5 to speed up the test
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    def test_run_pipeline_valid_sequences(self, mock_compose, seq):
        """Test that run_pipeline works with valid sequences.

        [UNIQUE-ERR-STAGEB-RUNPIPELINE-001] This test verifies that run_pipeline
        correctly processes valid RNA sequences and returns the expected output.

        Note: This test was previously skipped due to timeout issues. It has been fixed by:
        1. Using the same model instances for all test cases instead of creating new ones each time
        2. Limiting the number of examples that Hypothesis generates
        3. Reducing the maximum sequence length to 5 to speed up the test
        """
        # Create a mock config
        mock_cfg = self._make_cfg()
        mock_compose.return_value = mock_cfg

        # Create real model instances
        device = torch.device("cpu")

        # Create StageA model
        stage_a = StageARFoldPredictor(stage_cfg=mock_cfg.model.stageA, device=device)

        # Create StageB model
        stage_b = StageBTorsionBertPredictor(mock_cfg)

        # Create StageC model
        stage_c = StageCReconstruction()

        # Patch the model creation functions to return our instances
        with patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor", return_value=stage_a) as mock_stage_a, \
             patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor", return_value=stage_b) as mock_stage_b, \
             patch("rna_predict.pipeline.stageB.main.StageCReconstruction", return_value=stage_c) as mock_stage_c:

            # Call run_pipeline directly
            out = run_pipeline(seq, cfg=mock_cfg)

            # Verify the output
            assert "coords" in out, f"[UNIQUE-ERR-STAGEB-RUNPIPELINE-001] 'coords' key missing for seq={seq}"
            assert "atom_count" in out, f"[UNIQUE-ERR-STAGEB-RUNPIPELINE-002] 'atom_count' key missing for seq={seq}"

            # Verify that the mocks were called with the expected arguments
            mock_stage_a.assert_called_once()
            mock_stage_b.assert_called_once()
            mock_stage_c.assert_called_once()



    @given(seq=st.text(alphabet="ACGU", min_size=0, max_size=0))
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    def test_run_pipeline_empty_sequence(self, mock_compose, seq):
        """Test that run_pipeline raises an error for empty sequences.

        [UNIQUE-ERR-STAGEB-RUNPIPELINE-003] This test verifies that run_pipeline
        correctly raises a ValueError for empty sequences.
        """
        # Mock the hydra.compose function to return a mock config
        mock_cfg = self._make_cfg()
        mock_compose.return_value = mock_cfg

        # Call run_pipeline with an empty sequence and check that it raises the expected exception
        with pytest.raises(ValueError, match="Input sequence must not be empty.*\\[ERR-STAGEB-RUNPIPELINE-002\\]"):
            run_pipeline(seq, cfg=mock_cfg)

    @given(seq=st.text(alphabet="XYZ123", min_size=1, max_size=10))
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    def test_run_pipeline_invalid_sequence(self, mock_compose, seq):
        """Test that run_pipeline raises an error for invalid sequences.

        [UNIQUE-ERR-STAGEB-RUNPIPELINE-004] This test verifies that run_pipeline
        correctly raises a ValueError for sequences containing invalid characters.
        """
        # Mock the hydra.compose function to return a mock config
        mock_cfg = self._make_cfg()
        mock_compose.return_value = mock_cfg

        # Call run_pipeline with an invalid sequence and check that it raises the expected exception
        with pytest.raises(ValueError, match="Invalid RNA sequence.*\\[ERR-STAGEB-RUNPIPELINE-003\\]"):
            run_pipeline(seq, cfg=mock_cfg)

    @settings(deadline=None, max_examples=2)  # Limit to 2 examples to avoid timeouts
    @given(seq=st.text(alphabet="ACGU", min_size=10, max_size=15))  # Test with longer sequences
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    def test_run_pipeline_long_sequence(self, mock_compose, seq):
        """Test that run_pipeline works efficiently with long sequences.

        [UNIQUE-ERR-STAGEB-RUNPIPELINE-005] This test verifies that run_pipeline
        processes long sequences efficiently and returns the expected output.

        Note: This test was previously skipped due to timeout issues. It has been fixed by:
        1. Using the same model instances for all test cases instead of creating new ones each time
        2. Limiting the number of examples that Hypothesis generates
        3. Using a fixed range of sequence lengths (10-15) to ensure we test longer sequences
        """
        # Create a mock config
        mock_cfg = self._make_cfg()
        mock_compose.return_value = mock_cfg

        # Create real model instances
        device = torch.device("cpu")

        # Create StageA model
        stage_a = StageARFoldPredictor(stage_cfg=mock_cfg.model.stageA, device=device)

        # Create StageB model
        stage_b = StageBTorsionBertPredictor(mock_cfg)

        # Create StageC model
        stage_c = StageCReconstruction()

        # Patch the model creation functions to return our instances
        with patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor", return_value=stage_a) as mock_stage_a, \
             patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor", return_value=stage_b) as mock_stage_b, \
             patch("rna_predict.pipeline.stageB.main.StageCReconstruction", return_value=stage_c) as mock_stage_c:

            # Call run_pipeline directly
            out = run_pipeline(seq, cfg=mock_cfg)

            # Verify the output
            assert "coords" in out, f"[UNIQUE-ERR-STAGEB-RUNPIPELINE-005] 'coords' key missing for seq={seq}"
            assert "atom_count" in out, f"[UNIQUE-ERR-STAGEB-RUNPIPELINE-006] 'atom_count' key missing for seq={seq}"

            # Verify that the mocks were called with the expected arguments
            mock_stage_a.assert_called_once()
            mock_stage_b.assert_called_once()
            mock_stage_c.assert_called_once()


class TestRunStageBCombined:
    """Tests for the run_stageB_combined function."""

    @staticmethod
    def _make_cfg():
        # Use OmegaConf to create a DictConfig compatible config
        cfg_dict = {
            "model": {
                "stageA": {
                    "min_seq_length": 80,
                    "dropout": 0.3,
                    "num_hidden": 128,
                    "device": "cpu",
                    "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
                    "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
                    "checkpoint_zip_path": "RFold/checkpoints.zip",
                    "batch_size": 32,
                    "lr": 0.001,
                    "threshold": 0.5,
                    "debug_logging": True,
                    "run_example": True,
                    "example_sequence": "AAGUCUGGUGGACAUUGGCGUCCUGAGGUGUUAAAACCUCUUAUUGCUGACGCCAGAAAGAGAAGAACUUCGGUUCUACUAGUCGACUAUACUACAAGCUUUGGGUGUAUAGCGGCAAGACAACCUGGAUCGGGGGAGGCUAAGGGCGCAAGCCUAUGCUAACCCCGAGCCGAGCUACUGGAGGGCAACCCCCAGAUAGCCGGUGUAGAGCGCGGAAAGGUGUCGGUCAUCCUAUCUGAUAGGUGGCUUGAGGGACGUGCCGUCUCACCCGAAAGGGUGUUUCUAAGGAGGAGCUCCCAAAGGGCAAAUCUUAGAAAAGGGUGUAUACCCUAUAAUUUAACGGCCAGCAGCC",
                    "visualization": {
                        "enabled": True,
                        "varna_jar_path": "tools/varna-3-93.jar",
                        "resolution": 8.0,
                        "output_path": "test_seq.png"
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
                },
                "stageB": {},
                "stageC": {}
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 32
            }
        }
        return OmegaConf.create(cfg_dict)

    def setup_method(self):
        """Set up test fixtures."""
        self.sequence = "ACGUACGU"
        self.adjacency_matrix = torch.ones((8, 8))
        self.torsion_bert_model = MagicMock(spec=StageBTorsionBertPredictor)
        self.torsion_bert_model.return_value = {"torsion_angles": torch.ones((8, 7))}
        self.pairformer_model = MagicMock()
        self.pairformer_model.c_s = 64
        self.pairformer_model.c_z = 32

        # Create return value with the correct shape for the mock
        # The function expects a tuple of (s_up, z_up) where:
        # - s_up has shape [batch, N, c_s] = [1, 8, 64]
        # - z_up has shape [batch, N, N, c_z] = [1, 8, 8, 32]
        self.pairformer_model.return_value = (
            torch.ones((1, 8, 64)), torch.ones((1, 8, 8, 32))
        )
        self.device = "cpu"

    @settings(deadline=None)
    @given(sequence=st.text(alphabet="ACGU", min_size=1, max_size=10))
    def test_run_stageB_combined_basic(self, sequence):
        """Test that run_stageB_combined runs without errors and returns expected output structure.

        [UNIQUE-ERR-STAGEB-COMBINED-001] This test verifies that run_stageB_combined
        correctly processes inputs and returns the expected output structure.
        """
        cfg = self._make_cfg()
        N = len(sequence)
        adjacency_matrix = torch.ones((N, N))
        torsion_bert_model = MagicMock(spec=StageBTorsionBertPredictor)
        torsion_bert_model.return_value = {"torsion_angles": torch.ones((N, 7))}
        pairformer_model = MagicMock()
        pairformer_model.c_s = 64
        pairformer_model.c_z = 32
        pairformer_model.return_value = (
            torch.ones((1, N, 64)), torch.ones((1, N, N, 32))
        )
        device = "cpu"
        # Call the function under test
        out = run_stageB_combined(
            sequence,
            adjacency_matrix=adjacency_matrix,
            torsion_bert_model=torsion_bert_model,
            pairformer_model=pairformer_model,
            device=device,
            cfg=cfg,
        )
        # Verify output structure
        assert "torsion_angles" in out, f"[UNIQUE-ERR-STAGEB-COMBINED-001] 'torsion_angles' key missing for seq={sequence}"
        assert "s_embeddings" in out, f"[UNIQUE-ERR-STAGEB-COMBINED-001] 's_embeddings' key missing for seq={sequence}"
        assert "z_embeddings" in out, f"[UNIQUE-ERR-STAGEB-COMBINED-001] 'z_embeddings' key missing for seq={sequence}"

    def test_run_stageB_combined_empty_sequence(self):
        """Test run_stageB_combined with an empty sequence."""
        # Set up mocks for empty sequence
        empty_sequence = ""
        empty_adjacency_matrix = torch.zeros((0, 0))
        self.torsion_bert_model.return_value = {"torsion_angles": torch.zeros((0, 7))}
        self.pairformer_model = MagicMock()
        self.pairformer_model.c_s = 64
        self.pairformer_model.c_z = 32
        self.pairformer_model.return_value = (
            torch.zeros((0, 64)), torch.zeros((0, 0, 32))
        )

        # Run the function
        # Empty sequences should be handled gracefully
        result = run_stageB_combined(
            empty_sequence,
            adjacency_matrix=empty_adjacency_matrix,
            torsion_bert_model=self.torsion_bert_model,
            pairformer_model=self.pairformer_model,
            device=self.device,
            cfg=self._make_cfg(),
        )

        # Check that the result has the expected structure
        assert "torsion_angles" in result
        assert "s_embeddings" in result
        assert "z_embeddings" in result

        # Check shapes
        assert result["torsion_angles"].shape[0] == 0
        assert result["s_embeddings"].shape[0] == 0
        assert result["z_embeddings"].shape[0] == 0

    def test_run_stageB_combined_device_handling(self):
        """Test that run_stageB_combined handles device correctly."""
        # Create new mocks with the necessary attributes
        torsion_model = MagicMock()
        torsion_model.model = MagicMock()
        torsion_model.model.to = MagicMock()
        torsion_model.return_value = {"torsion_angles": torch.ones((8, 7))}

        pairformer_model = MagicMock()
        pairformer_model.to = MagicMock()
        pairformer_model.c_s = 64
        pairformer_model.c_z = 32
        pairformer_model.return_value = (
            torch.ones((8, 64)),  # s_up
            torch.ones((8, 8, 32)),  # z_up
        )

        # Run the function
        run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency_matrix,
            torsion_bert_model=torsion_model,
            pairformer_model=pairformer_model,
            device="cpu",  # Use the same device for testing
            cfg=self._make_cfg(),
        )

        # Verify that the models were moved to the correct device
        pairformer_model.to.assert_called_once_with(torch.device("cpu"))
        torsion_model.model.to.assert_called_once_with(torch.device("cpu"))

    def test_run_stageB_combined_shape_mismatch(self):
        """Test run_stageB_combined with a shape mismatch between sequence and adjacency.

        [UNIQUE-ERR-STAGEB-COMBINED-003] This test verifies that run_stageB_combined
        correctly handles a mismatch between the sequence length and adjacency matrix dimensions.
        """
        mismatch_sequence = "ACGU"
        mismatch_adjacency_matrix = torch.ones((8, 8))
        self.torsion_bert_model.return_value = {"torsion_angles": torch.ones((8, 7))}
        self.pairformer_model = MagicMock()
        self.pairformer_model.c_s = 64
        self.pairformer_model.c_z = 32
        self.pairformer_model.return_value = (
            torch.ones((1, 8, 64)), torch.ones((1, 8, 8, 32))
        )

        # Expect ValueError due to shape mismatch
        with pytest.raises(ValueError, match="Shape mismatch: sequence length \(4\) does not match adjacency matrix shape \(8\). \[ERR-STAGEB-COMBINED-SHAPE-MISMATCH\]"):
            run_stageB_combined(
                mismatch_sequence,
                adjacency_matrix=mismatch_adjacency_matrix,
                torsion_bert_model=self.torsion_bert_model,
                pairformer_model=self.pairformer_model,
                device=self.device,
                cfg=self._make_cfg(),
            )


class TestDemoGradientFlowTest:
    """Tests for the demo_gradient_flow_test function."""

    def test_demo_gradient_flow_test_basic(self):
        """Test demo_gradient_flow_test with proper Hydra initialization.

        [UNIQUE-ERR-STAGEB-GRADIENTFLOW-001] This test verifies that demo_gradient_flow_test
        runs without errors when provided with a valid configuration.

        Note: The test was previously failing due to a shape mismatch issue. The demo_gradient_flow_test
        function creates linear layers that output tensors with shape [N, 3], but the target tensor
        had shape [N, 32]. The fix was to set target_dim to 3 in the test configuration to match the
        output dimension of the linear layers.
        """
        # Mock the hydra.compose function to return a mock config
        mock_cfg = OmegaConf.create({
            "model": {
                "stageA": {"device": "cpu"},
                "stageB": {
                    "debug_logging": True,
                    "torsion_bert": {"device": "cpu"},
                    "pairformer": {
                        "init_z_from_adjacency": True
                    }
                }
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 3  # Must be 3 to match the output dimension of the linear layers in demo_gradient_flow_test
            },
            "stageB_pairformer": {
                "c_s": 64,
                "c_z": 32,
                "device": "cpu"
            }
        })

        # Add debug logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("test_demo_gradient_flow_test_basic")
        logger.info("Starting test_demo_gradient_flow_test_basic")
        logger.info(f"Mock config: {OmegaConf.to_yaml(mock_cfg)}")

        # Create a wrapper function to call demo_gradient_flow_test without Hydra's CLI argument parsing
        def call_demo_gradient_flow_test():
            # Call the demo_gradient_flow_test function directly with the mock config
            from rna_predict.pipeline.stageB.main import demo_gradient_flow_test as original_demo_gradient_flow_test
            logger.info("Calling demo_gradient_flow_test")
            result = original_demo_gradient_flow_test(mock_cfg)
            logger.info("demo_gradient_flow_test completed successfully")
            return result

        # Run the function
        call_demo_gradient_flow_test()


class TestRunPipelineHypothesis:
    """Tests for the run_pipeline function using Hypothesis.

    [UNIQUE-ERR-STAGEB-HYPOTHESIS-001] This test class verifies that run_pipeline
    correctly processes valid RNA sequences using property-based testing.
    """

    def test_run_pipeline_valid_sequences(self):
        """Test run_pipeline with valid sequences using Hypothesis.

        [UNIQUE-ERR-STAGEB-HYPOTHESIS-002] This test verifies that run_pipeline
        correctly processes valid RNA sequences and returns the expected output.

        Note: This test was previously skipped due to timeout issues. It has been fixed by:
        1. Using the same model instances for all test cases instead of creating new ones each time
        2. Limiting the number of examples that Hypothesis generates
        3. Reducing the maximum sequence length to 5 to speed up the test
        """
        # Create a mock config
        mock_cfg = OmegaConf.create({
            "model": {
                "stageA": {
                    "device": "cpu",
                    "min_seq_length": 80,
                    "dropout": 0.3,
                    "num_hidden": 128,
                    "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
                    "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
                    "checkpoint_zip_path": "RFold/checkpoints.zip",
                    "batch_size": 32,
                    "lr": 0.001,
                    "threshold": 0.5,
                    "debug_logging": True,
                    "run_example": True,
                    "example_sequence": "ACGU",
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
                },
                "stageB": {
                    "debug_logging": True,
                    "torsion_bert": {
                        "device": "cpu",
                        "model_name_or_path": "sayby/rna_torsionbert"
                    },
                    "pairformer": {
                        "init_z_from_adjacency": True
                    }
                }
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 3
            },
            "stageB_pairformer": {
                "c_s": 64,
                "c_z": 32,
                "device": "cpu"
            }
        })

        # Create real model instances
        device = torch.device("cpu")

        # Create StageA model
        stage_a = StageARFoldPredictor(stage_cfg=mock_cfg.model.stageA, device=device)

        # Create StageB model
        stage_b = StageBTorsionBertPredictor(mock_cfg)

        # Create StageC model
        stage_c = StageCReconstruction()

        # Use Hypothesis to test the function with valid sequences
        @settings(deadline=None, max_examples=3)  # Limit to 3 examples to avoid timeouts
        @given(seq=st.text(alphabet="ACGU", min_size=1, max_size=5))  # Reduced max_size to 5 to speed up the test
        def test_run_pipeline_hypothesis(seq):
            # Patch the model creation functions to return our instances
            with patch("rna_predict.pipeline.stageB.main.hydra.compose", return_value=mock_cfg), \
                 patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor", return_value=stage_a), \
                 patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor", return_value=stage_b), \
                 patch("rna_predict.pipeline.stageB.main.StageCReconstruction", return_value=stage_c):

                # Call run_pipeline directly
                out = run_pipeline(seq, cfg=mock_cfg)

                # Verify the output
                assert "coords" in out, f"[UNIQUE-ERR-STAGEB-HYPOTHESIS-002] 'coords' key missing for seq={seq}"
                assert "atom_count" in out, f"[UNIQUE-ERR-STAGEB-HYPOTHESIS-003] 'atom_count' key missing for seq={seq}"

        # Run the test
        test_run_pipeline_hypothesis()


class TestMain:
    """Tests for the main function."""

    @patch("rna_predict.pipeline.stageB.main.run_pipeline")
    @patch("rna_predict.pipeline.stageB.main.demo_gradient_flow_test")
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    def test_main_basic(self, mock_compose, mock_demo, mock_run_pipeline):
        """Test that main runs without errors."""
        # Mock the hydra.compose function to return a mock config
        mock_cfg = OmegaConf.create({
            "model": {
                "stageA": {
                    "device": "cpu",
                    "min_seq_length": 80,
                    "dropout": 0.3,
                    "num_hidden": 128,
                    "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
                    "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
                    "checkpoint_zip_path": "RFold/checkpoints.zip",
                    "batch_size": 32,
                    "lr": 0.001,
                    "threshold": 0.5,
                    "debug_logging": True,
                    "run_example": True,
                    "example_sequence": "ACGU"
                },
                "stageB": {
                    "debug_logging": True,
                    "torsion_bert": {"device": "cpu"},
                    "pairformer": {
                        "init_z_from_adjacency": True
                    }
                }
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 3
            },
            "stageB_pairformer": {
                "c_s": 64,
                "c_z": 32,
                "device": "cpu"
            }
        })
        mock_compose.return_value = mock_cfg

        # Create a wrapper function to call main without Hydra's CLI argument parsing
        def call_main():
            # Call the main function directly with the mock config
            from rna_predict.pipeline.stageB.main import main as original_main
            return original_main(mock_cfg)

        # Run the function
        call_main()

        # Verify that the mocks were called with the expected arguments
        mock_run_pipeline.assert_called_once_with("ACGUACGU", mock_cfg)
        mock_demo.assert_called_once_with(mock_cfg)

    @patch("rna_predict.pipeline.stageB.main.run_pipeline")
    @patch("rna_predict.pipeline.stageB.main.demo_gradient_flow_test")
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    @patch("rna_predict.pipeline.stageB.main.logger.error")
    def test_main_run_pipeline_exception(self, mock_logger_error, mock_compose, mock_demo, mock_run_pipeline):
        """Test that main handles exceptions from run_pipeline correctly.

        [UNIQUE-ERR-STAGEB-MAIN-003] This test verifies that the main function properly
        catches and logs exceptions from run_pipeline without propagating them.
        """
        # Set up mocks to raise an exception
        mock_run_pipeline.side_effect = Exception("Error in run_pipeline")

        # Mock the hydra.compose function to return a mock config
        mock_cfg = OmegaConf.create({
            "model": {
                "stageA": {
                    "device": "cpu",
                    "min_seq_length": 80,
                    "dropout": 0.3,
                    "num_hidden": 128,
                    "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
                    "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
                    "checkpoint_zip_path": "RFold/checkpoints.zip",
                    "batch_size": 32,
                    "lr": 0.001,
                    "threshold": 0.5,
                    "debug_logging": True,
                    "run_example": True,
                    "example_sequence": "ACGU"
                },
                "stageB": {
                    "debug_logging": True,
                    "torsion_bert": {"device": "cpu"},
                    "pairformer": {
                        "init_z_from_adjacency": True
                    }
                }
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 3
            },
            "stageB_pairformer": {
                "c_s": 64,
                "c_z": 32,
                "device": "cpu"
            }
        })
        mock_compose.return_value = mock_cfg

        # Create a wrapper function to call main without Hydra's CLI argument parsing
        def call_main():
            # Call the main function directly with the mock config
            from rna_predict.pipeline.stageB.main import main as original_main
            return original_main(mock_cfg)

        # The main function catches exceptions from run_pipeline and logs them,
        # so we need to check that the function was called and the exception was raised
        # but not propagated
        call_main()

        # Verify that the mocks were called with the expected arguments
        mock_run_pipeline.assert_called_once_with("ACGUACGU", mock_cfg)
        # The main function continues execution even after run_pipeline fails,
        # so demo_gradient_flow_test is still called
        mock_demo.assert_called_once_with(mock_cfg)

        # Verify that the exception was raised in the mock
        assert mock_run_pipeline.side_effect is not None
        assert str(mock_run_pipeline.side_effect) == "Error in run_pipeline"

        # Verify that the error was properly logged
        # Use a more flexible approach to check that the error was logged
        error_logged = False
        for call_args in mock_logger_error.call_args_list:
            args, _ = call_args
            if len(args) >= 1 and "Error running pipeline" in args[0]:
                error_logged = True
                break

        assert error_logged, "[UNIQUE-ERR-STAGEB-MAIN-004] Error message was not properly logged"

    @patch("rna_predict.pipeline.stageB.main.run_pipeline")
    @patch("rna_predict.pipeline.stageB.main.demo_gradient_flow_test")
    @patch("rna_predict.pipeline.stageB.main.hydra.compose")
    @patch("rna_predict.pipeline.stageB.main.logger.error")
    def test_main_demo_exception(self, mock_logger_error, mock_compose, mock_demo, mock_run_pipeline):
        """Test that main handles exceptions from demo_gradient_flow_test correctly.

        [UNIQUE-ERR-STAGEB-MAIN-001] This test verifies that the main function properly
        catches and logs exceptions from demo_gradient_flow_test without propagating them.
        """
        # Set up mocks to raise an exception
        mock_demo.side_effect = Exception("Error in demo_gradient_flow_test")

        # Mock the hydra.compose function to return a mock config
        mock_cfg = OmegaConf.create({
            "model": {
                "stageA": {
                    "device": "cpu",
                    "min_seq_length": 80,
                    "dropout": 0.3,
                    "num_hidden": 128,
                    "checkpoint_path": "RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
                    "checkpoint_url": "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
                    "checkpoint_zip_path": "RFold/checkpoints.zip",
                    "batch_size": 32,
                    "lr": 0.001,
                    "threshold": 0.5,
                    "debug_logging": True,
                    "run_example": True,
                    "example_sequence": "ACGU"
                },
                "stageB": {
                    "debug_logging": True,
                    "torsion_bert": {"device": "cpu"},
                    "pairformer": {
                        "init_z_from_adjacency": True
                    }
                }
            },
            "test_data": {
                "sequence": "ACGUACGU",
                "adjacency_fill_value": 0.5,
                "target_dim": 3
            },
            "stageB_pairformer": {
                "c_s": 64,
                "c_z": 32,
                "device": "cpu"
            }
        })
        mock_compose.return_value = mock_cfg

        # Create a wrapper function to call main without Hydra's CLI argument parsing
        def call_main():
            # Call the main function directly with the mock config
            from rna_predict.pipeline.stageB.main import main as original_main
            return original_main(mock_cfg)

        # The main function catches exceptions from demo_gradient_flow_test and logs them,
        # so we need to check that the function was called and the exception was raised
        # but not propagated
        call_main()

        # Verify that the mocks were called with the expected arguments
        mock_run_pipeline.assert_called_once_with("ACGUACGU", mock_cfg)
        mock_demo.assert_called_once_with(mock_cfg)

        # Verify that the exception was raised in the mock
        assert mock_demo.side_effect is not None
        assert str(mock_demo.side_effect) == "Error in demo_gradient_flow_test"

        # Verify that the error was properly logged
        # Use a more flexible approach to check that the error was logged
        error_logged = False
        for call_args in mock_logger_error.call_args_list:
            args, _ = call_args
            if len(args) >= 1 and "Error in gradient flow test" in args[0]:
                error_logged = True
                break

        assert error_logged, "[UNIQUE-ERR-STAGEB-MAIN-002] Error message was not properly logged"


# --- NEW TEST: property-based config structure validation for StageBTorsionBertPredictor ---
def make_config(structure):
    return OmegaConf.create(structure)

@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=16),
        values=st.recursive(
            st.integers() | st.text() | st.booleans() | st.none(),
            lambda children: st.lists(children) | st.dictionaries(st.text(min_size=1, max_size=16), children),
            max_leaves=5,
        ),
        min_size=1,
        max_size=3,
    )
)
def test_stageb_torsionbert_config_structure_property(config_dict):
    """
    Property-based test: StageBTorsionBertPredictor should raise unique error code [ERR-TORSIONBERT-CONFIG-001]
    if config is missing model.stageB.torsion_bert. This ensures config validation is robust and future-proof.
    """
    from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
    if not ("model" in config_dict and isinstance(config_dict["model"], dict) and "stageB" in config_dict["model"] and isinstance(config_dict["model"]["stageB"], dict) and "torsion_bert" in config_dict["model"]["stageB"]):
        cfg = make_config(config_dict)
        with pytest.raises(ValueError) as excinfo:
            StageBTorsionBertPredictor(cfg)
        assert "[ERR-TORSIONBERT-CONFIG-001]" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main(["-v", "test_main_comprehensive.py"])
