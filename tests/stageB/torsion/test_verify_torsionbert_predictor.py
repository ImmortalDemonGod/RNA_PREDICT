"""
Verification script for StageBTorsionBertPredictor component.

This script verifies the StageBTorsionBertPredictor component according to the
verification checklist:

1. Instantiation
   - Verify successful instantiation of StageBTorsionBertPredictor
   - Confirm TorsionBERT model loads correctly from specified path
   - Validate tokenizer loads properly
   - Ensure pre-trained weights load without errors
   - Check device configuration (CPU/GPU) is appropriate

2. Interface
   - Verify callable functionality via __call__(sequence, adjacency=None) method
   - Confirm acceptance of RNA sequence string input
   - Test optional adjacency matrix parameter handling

3. Output Validation
   - Confirm return value is a dictionary
   - Verify dictionary contains required "torsion_angles" key
   - Ensure value is a PyTorch Tensor
   - Validate tensor shape matches expected format [N, 14] for sin/cos pairs of 7 angles
   - Check all values fall within valid [-1, 1] range for sin/cos mode
   - Inspect for any NaN/Inf values

4. Functional Testing
   - Execute predictor with test sequence to verify end-to-end operation
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

# Import the component to test
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)


class TestStageBTorsionBertPredictorVerification:
    """
    Tests to verify the StageBTorsionBertPredictor component according to the checklist.
    """

    @pytest.fixture
    def mock_predictor(self):
        """
        Fixture to create a mock StageBTorsionBertPredictor that returns predictable outputs.
        """
        import torch
        from omegaconf import OmegaConf
        from unittest.mock import patch, MagicMock

        # Create a config
        cfg = OmegaConf.create({
            "stageB_torsion": {
                "model_name_or_path": "sayby/rna_torsionbert",
                "device": "cpu",
                "angle_mode": "sin_cos",
                "num_angles": 7,
                "max_length": 512
            }
        })

        # Dummy tokenizer class (not MagicMock)
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                # Simulate tokenizer output for a sequence of length N
                N = 8
                return {"input_ids": torch.zeros((1, N), dtype=torch.long), "attention_mask": torch.ones((1, N), dtype=torch.long)}

        dummy_tokenizer = DummyTokenizer()
        mock_model = MagicMock(name="MockModel")
        mock_model.to.return_value = mock_model

        with patch(
            "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained", lambda *a, **kw: dummy_tokenizer
        ), patch(
            "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained", lambda *a, **kw: mock_model
        ):
            predictor = StageBTorsionBertPredictor(cfg=cfg)
            # Patch predict_angles_from_sequence to return a valid tensor
            def dummy_predict_angles_from_sequence(seq, *args, **kwargs):
                n = len(seq)
                return torch.zeros(n, 14)  # 2*num_angles
            predictor.predict_angles_from_sequence = dummy_predict_angles_from_sequence
            return predictor

    @pytest.fixture
    def real_predictor(self):
        """
        Fixture to create a real StageBTorsionBertPredictor instance.
        This may fail if the model can't be loaded, so we'll handle that in the tests.
        """
        try:
            # Create a config
            cfg = OmegaConf.create({
                "stageB_torsion": {
                    "model_name_or_path": "sayby/rna_torsionbert",
                    "device": "cpu",
                    "angle_mode": "sin_cos",
                    "num_angles": 7,
                    "max_length": 512
                }
            })

            predictor = StageBTorsionBertPredictor(cfg=cfg)
            return predictor
        except Exception as e:
            pytest.skip(f"Failed to load real model: {e}")

    def test_instantiation(self, mock_predictor):
        """
        Verify successful instantiation of StageBTorsionBertPredictor.
        """
        # Check that the predictor was created successfully
        assert mock_predictor is not None
        assert isinstance(mock_predictor, StageBTorsionBertPredictor)

        # Check that the model attributes are set correctly
        assert mock_predictor.model_name_or_path == "sayby/rna_torsionbert"
        assert str(mock_predictor.device) == "cpu"
        assert mock_predictor.angle_mode == "sin_cos"
        assert mock_predictor.num_angles == 7
        assert mock_predictor.max_length == 512  # Default value

    def test_callable_interface(self, mock_predictor):
        """
        Property-based: Verify callable functionality via __call__(sequence, adjacency=None) method, with unique error codes.
        """
        from hypothesis import given, strategies as st
        import torch

        @given(
            sequence=st.text(alphabet="ACGU", min_size=1, max_size=16),
            use_adjacency=st.booleans()
        )
        def check_callable(sequence, use_adjacency):
            try:
                if use_adjacency and len(sequence) > 0:
                    adjacency = torch.zeros((len(sequence), len(sequence)))
                    result = mock_predictor(sequence, adjacency=adjacency)
                else:
                    result = mock_predictor(sequence)
            except Exception as e:
                import pytest
                pytest.fail(f"[UNIQUE-ERR-TORSIONBERT-CALLABLE-EXC] Exception in callable interface: {e}")
            assert isinstance(result, dict), "[UNIQUE-ERR-TORSIONBERT-CALLABLE-DICT] Output is not a dict"
            assert "torsion_angles" in result, "[UNIQUE-ERR-TORSIONBERT-CALLABLE-KEY] Missing 'torsion_angles' key"
            torsion_angles = result["torsion_angles"]
            assert isinstance(torsion_angles, torch.Tensor), "[UNIQUE-ERR-TORSIONBERT-CALLABLE-TENSOR] Output is not a tensor"
            expected_shape = (len(sequence), 14)
            assert torsion_angles.shape == expected_shape, f"[UNIQUE-ERR-TORSIONBERT-CALLABLE-SHAPE] Expected shape {expected_shape}, got {torsion_angles.shape}"
        check_callable()

    def test_output_validation(self, mock_predictor):
        """
        Property-based: Validate the output of the predictor with unique error codes.
        """
        from hypothesis import given, strategies as st
        import torch

        @given(st.text(alphabet="ACGU", min_size=1, max_size=16))
        def check_output(sequence):
            try:
                result = mock_predictor(sequence)
            except Exception as e:
                import pytest
                pytest.fail(f"[UNIQUE-ERR-TORSIONBERT-OUTPUT-EXC] Exception in output validation: {e}")
            assert isinstance(result, dict), "[UNIQUE-ERR-TORSIONBERT-OUTPUT-DICT] Output is not a dict"
            assert "torsion_angles" in result, "[UNIQUE-ERR-TORSIONBERT-OUTPUT-KEY] Missing 'torsion_angles' key"
            torsion_angles = result["torsion_angles"]
            assert isinstance(torsion_angles, torch.Tensor), "[UNIQUE-ERR-TORSIONBERT-OUTPUT-TENSOR] Output is not a tensor"
            expected_shape = (len(sequence), 14)
            assert torsion_angles.shape == expected_shape, f"[UNIQUE-ERR-TORSIONBERT-OUTPUT-SHAPE] Expected shape {expected_shape}, got {torsion_angles.shape}"
            if mock_predictor.angle_mode == "sin_cos":
                assert torch.all(torsion_angles >= -1.0), "[UNIQUE-ERR-TORSIONBERT-OUTPUT-RANGE] Values < -1"
                assert torch.all(torsion_angles <= 1.0), "[UNIQUE-ERR-TORSIONBERT-OUTPUT-RANGE] Values > 1"
            assert not torch.any(torch.isnan(torsion_angles)), "[UNIQUE-ERR-TORSIONBERT-OUTPUT-NAN] NaN in output"
            assert not torch.any(torch.isinf(torsion_angles)), "[UNIQUE-ERR-TORSIONBERT-OUTPUT-INF] Inf in output"
        check_output()

    def test_functional_end_to_end(self, mock_predictor):
        """
        Property-based: Execute predictor with test sequences to verify end-to-end operation, with unique error codes.
        """
        from hypothesis import given, strategies as st

        @given(st.lists(st.text(alphabet="ACGU", min_size=1, max_size=8), min_size=1, max_size=6))
        def check_end_to_end(sequences):
            for seq in sequences:
                try:
                    result = mock_predictor(seq)
                except Exception as e:
                    import pytest
                    pytest.fail(f"[UNIQUE-ERR-TORSIONBERT-END2END-EXC] Exception in end-to-end: {e}")
                assert isinstance(result, dict), "[UNIQUE-ERR-TORSIONBERT-END2END-DICT] Output is not a dict"
                assert "torsion_angles" in result, "[UNIQUE-ERR-TORSIONBERT-END2END-KEY] Missing 'torsion_angles' key"
                torsion_angles = result["torsion_angles"]
                assert torsion_angles.shape == (len(seq), 14), f"[UNIQUE-ERR-TORSIONBERT-END2END-SHAPE] Expected shape ({len(seq)}, 14), got {torsion_angles.shape}"
        check_end_to_end()

    def test_angle_mode_conversion(self):
        """
        Test that the angle_mode parameter correctly affects the output shape and values.

        This test verifies that the StageBTorsionBertPredictor correctly handles different angle modes
        and performs the appropriate conversions between sin/cos pairs and angle values in radians or degrees.

        # ERROR_ID: TORSIONBERT_ANGLE_MODE_CONVERSION
        """
        # Create a dummy model that won't trigger the MagicMock assertion
        from rna_predict.pipeline.stageB.torsion.torsionbert_inference import DummyTorsionBertAutoModel

        # Create dummy model and tokenizer
        dummy_model = DummyTorsionBertAutoModel(num_angles=7)
        dummy_tokenizer = MagicMock()
        # Configure the tokenizer mock to return a valid dictionary
        dummy_tokenizer.return_value = {"input_ids": torch.ones(1, 10, dtype=torch.long), "attention_mask": torch.ones(1, 10)}

        # We'll use patching for all three predictors to avoid actual model loading
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=dummy_tokenizer
            ),
            patch(
                "transformers.AutoModel.from_pretrained",
                return_value=dummy_model
            ),
        ):
            # Test with sin_cos mode
            cfg_sincos = OmegaConf.create({
                "stageB_torsion": {
                    "model_name_or_path": "dummy_path",
                    "device": "cpu",
                    "angle_mode": "sin_cos",
                    "num_angles": 7,
                    "max_length": 512
                }
            })
            predictor_sincos = StageBTorsionBertPredictor(cfg=cfg_sincos)

            # Patch the predict_angles_from_sequence method
            with patch.object(
                predictor_sincos, "predict_angles_from_sequence"
            ) as mock_method:
                mock_method.return_value = torch.ones(
                    (4, 14)
                )  # All ones for simplicity

                # Test with a simple sequence
                sequence = "ACGU"
                result_sincos = predictor_sincos(sequence)

                # Check the shape for sin_cos mode
                assert result_sincos["torsion_angles"].shape == (4, 14)

            # Test with radians mode
            cfg_radians = OmegaConf.create({
                "stageB_torsion": {
                    "model_name_or_path": "dummy_path",
                    "device": "cpu",
                    "angle_mode": "radians",
                    "num_angles": 7,
                    "max_length": 512
                }
            })
            predictor_radians = StageBTorsionBertPredictor(cfg=cfg_radians)

            # Patch the predict_angles_from_sequence method
            with patch.object(
                predictor_radians, "predict_angles_from_sequence"
            ) as mock_method:
                mock_method.return_value = torch.ones(
                    (4, 14)
                )  # All ones for simplicity

                result_radians = predictor_radians(sequence)

                # Check the shape for radians mode
                assert result_radians["torsion_angles"].shape == (4, 7)

                # Check that the values are converted correctly
                # For sin=1, cos=1, the angle should be π/4 radians
                assert torch.allclose(
                    result_radians["torsion_angles"],
                    torch.full((4, 7), torch.pi / 4),
                    atol=1e-5,
                )

            # Test with degrees mode
            cfg_degrees = OmegaConf.create({
                "stageB_torsion": {
                    "model_name_or_path": "dummy_path",
                    "device": "cpu",
                    "angle_mode": "degrees",
                    "num_angles": 7,
                    "max_length": 512
                }
            })
            predictor_degrees = StageBTorsionBertPredictor(cfg=cfg_degrees)

            # Patch the predict_angles_from_sequence method
            with patch.object(
                predictor_degrees, "predict_angles_from_sequence"
            ) as mock_method:
                mock_method.return_value = torch.ones(
                    (4, 14)
                )  # All ones for simplicity

                result_degrees = predictor_degrees(sequence)

                # Check the shape for degrees mode
                assert result_degrees["torsion_angles"].shape == (4, 7)

                # Check that the values are converted correctly
                # For sin=1, cos=1, the angle should be 45 degrees
                assert torch.allclose(
                    result_degrees["torsion_angles"],
                    torch.full((4, 7), 45.0),
                    atol=1e-5,
                )

    @pytest.mark.skip(reason="Test is hanging or taking too long to run. The real model may not be available or accessible. [ERR-TORSIONBERT-TIMEOUT-001]")
    def test_real_model_if_available(self, real_predictor):
        """
        Test with the real model if it's available.
        This test will be skipped if the model can't be loaded.

        Note: This test is currently skipped because it hangs or takes too long to run.
        The issue might be related to the TorsionBERT model availability or
        the test environment. Further investigation is needed.

        Possible issues:
        1. The model 'sayby/rna_torsionbert' is not available or accessible
        2. Network connectivity issues when trying to download the model
        3. Memory limitations when loading the model
        4. Timeout issues during model initialization

        [ERR-TORSIONBERT-TIMEOUT-001]
        """
        if real_predictor is None:
            pytest.skip("Real model not available")

        # Test with a simple RNA sequence
        sequence = "ACGU"
        try:
            result = real_predictor(sequence)

            # Check that the result is a dictionary
            assert isinstance(result, dict)

            # Check that the dictionary contains the required key
            assert "torsion_angles" in result

            # Check that the tensor shape matches the sequence length
            torsion_angles = result["torsion_angles"]
            assert torsion_angles.shape[0] == len(sequence)

            # The shape[1] depends on the angle_mode and the actual model output
            if real_predictor.angle_mode == "sin_cos":
                # For sin_cos mode, we expect an even number of columns (for sin/cos pairs)
                assert (
                    torsion_angles.shape[1] % 2 == 0
                ), "Expected even number of columns for sin/cos pairs"
                # Store the actual number of angles for informational purposes
                actual_num_angles = torsion_angles.shape[1] // 2
                print(
                    f"Model outputs {actual_num_angles} angles ({torsion_angles.shape[1]} columns)"
                )
            else:
                # For radians/degrees mode, we expect the number of columns to match the number of angles
                actual_num_angles = torsion_angles.shape[1]
                print(f"Model outputs {actual_num_angles} angles")

            # Check for NaN/Inf values
            assert not torch.any(torch.isnan(torsion_angles))
            assert not torch.any(torch.isinf(torsion_angles))

            # Check that the tensor shape matches the sequence length
            assert torsion_angles.shape[0] == len(sequence)

            print(f"Real model test passed with shape {torsion_angles.shape}")
        except Exception as e:
            pytest.fail(f"Real model test failed: {e}")
