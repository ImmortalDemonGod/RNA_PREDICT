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

import torch
import pytest
from unittest.mock import patch, MagicMock

# Import the component to test
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor


class TestStageBTorsionBertPredictorVerification:
    """
    Tests to verify the StageBTorsionBertPredictor component according to the checklist.
    """

    @pytest.fixture
    def mock_predictor(self):
        """
        Fixture to create a mock StageBTorsionBertPredictor that returns predictable outputs.
        """
        # Patch the TorsionBertModel and DummyTorsionModel to avoid actual model loading
        with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained") as mock_tokenizer:
            with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained") as mock_model:
                # Configure the mocks
                mock_tokenizer.return_value = MagicMock(name="MockTokenizer")
                mock_model.return_value = MagicMock(name="MockModel")
                mock_model.return_value.to.return_value = mock_model.return_value
                
                # Create the predictor
                predictor = StageBTorsionBertPredictor(
                    model_name_or_path="sayby/rna_torsionbert",
                    device="cpu",
                    angle_mode="sin_cos",
                    num_angles=7
                )
                
                # Patch the predict_angles_from_sequence method
                with patch.object(predictor.model, 'predict_angles_from_sequence') as mock_method:
                    def side_effect(seq_arg):
                        N = len(seq_arg)
                        # Create a tensor with values in the range [-1, 1] for sin/cos pairs
                        return torch.rand((N, 14)) * 2 - 1
                    
                    mock_method.side_effect = side_effect
                    
                    yield predictor

    @pytest.fixture
    def real_predictor(self):
        """
        Fixture to create a real StageBTorsionBertPredictor instance.
        This may fail if the model can't be loaded, so we'll handle that in the tests.
        """
        try:
            predictor = StageBTorsionBertPredictor(
                model_name_or_path="sayby/rna_torsionbert",
                device="cpu",
                angle_mode="sin_cos",
                num_angles=7
            )
            return predictor
        except Exception as e:
            pytest.skip(f"Failed to load real model: {e}")
            return None

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
        Verify callable functionality via __call__(sequence, adjacency=None) method.
        """
        # Test with a simple RNA sequence
        sequence = "ACGU"
        result = mock_predictor(sequence)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Test with adjacency matrix
        adjacency = torch.zeros((4, 4))
        result_with_adj = mock_predictor(sequence, adjacency=adjacency)
        
        # Check that the result is still a dictionary
        assert isinstance(result_with_adj, dict)

    def test_output_validation(self, mock_predictor):
        """
        Validate the output of the predictor.
        """
        sequence = "ACGU"
        result = mock_predictor(sequence)
        
        # Check that the result is a dictionary
        assert isinstance(result, dict)
        
        # Check that the dictionary contains the required "torsion_angles" key
        assert "torsion_angles" in result
        
        # Check that the value is a PyTorch Tensor
        assert isinstance(result["torsion_angles"], torch.Tensor)
        
        # Check the tensor shape
        torsion_angles = result["torsion_angles"]
        assert torsion_angles.shape == (4, 14)  # [N, 2*num_angles] for sin/cos pairs
        
        # Check that all values are within the valid range for sin/cos mode
        if mock_predictor.angle_mode == "sin_cos":
            assert torch.all(torsion_angles >= -1.0)
            assert torch.all(torsion_angles <= 1.0)
        
        # Check for NaN/Inf values
        assert not torch.any(torch.isnan(torsion_angles))
        assert not torch.any(torch.isinf(torsion_angles))
        
        # Check that the dictionary contains the "residue_count" key
        assert "residue_count" in result
        assert result["residue_count"] == 4

    def test_angle_mode_conversion(self):
        """
        Test that the angle_mode parameter correctly affects the output shape and values.
        """
        # We'll use patching for all three predictors to avoid actual model loading
        with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained"), \
             patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained"):
            
            # Test with sin_cos mode
            predictor_sincos = StageBTorsionBertPredictor(
                model_name_or_path="dummy_path",
                device="cpu",
                angle_mode="sin_cos",
                num_angles=7
            )
            
            # Patch the predict_angles_from_sequence method
            with patch.object(predictor_sincos.model, 'predict_angles_from_sequence') as mock_method:
                mock_method.return_value = torch.ones((4, 14))  # All ones for simplicity
                
                # Test with a simple sequence
                sequence = "ACGU"
                result_sincos = predictor_sincos(sequence)
                
                # Check the shape for sin_cos mode
                assert result_sincos["torsion_angles"].shape == (4, 14)
            
            # Test with radians mode
            predictor_radians = StageBTorsionBertPredictor(
                model_name_or_path="dummy_path",
                device="cpu",
                angle_mode="radians",
                num_angles=7
            )
            
            # Patch the predict_angles_from_sequence method
            with patch.object(predictor_radians.model, 'predict_angles_from_sequence') as mock_method:
                mock_method.return_value = torch.ones((4, 14))  # All ones for simplicity
                
                result_radians = predictor_radians(sequence)
                
                # Check the shape for radians mode
                assert result_radians["torsion_angles"].shape == (4, 7)
                
                # Check that the values are converted correctly
                # For sin=1, cos=1, the angle should be π/4 radians
                assert torch.allclose(result_radians["torsion_angles"], torch.full((4, 7), torch.pi/4), atol=1e-5)
            
            # Test with degrees mode
            predictor_degrees = StageBTorsionBertPredictor(
                model_name_or_path="dummy_path",
                device="cpu",
                angle_mode="degrees",
                num_angles=7
            )
            
            # Patch the predict_angles_from_sequence method
            with patch.object(predictor_degrees.model, 'predict_angles_from_sequence') as mock_method:
                mock_method.return_value = torch.ones((4, 14))  # All ones for simplicity
                
                result_degrees = predictor_degrees(sequence)
                
                # Check the shape for degrees mode
                assert result_degrees["torsion_angles"].shape == (4, 7)
                
                # Check that the values are converted correctly
                # For sin=1, cos=1, the angle should be 45 degrees
                assert torch.allclose(result_degrees["torsion_angles"], torch.full((4, 7), 45.0), atol=1e-5)

    def test_functional_end_to_end(self, mock_predictor):
        """
        Execute predictor with test sequences to verify end-to-end operation.
        """
        # Test with various RNA sequences
        sequences = ["A", "AC", "ACG", "ACGU", "ACGUA", "ACGUAC"]
        
        for seq in sequences:
            result = mock_predictor(seq)
            
            # Check that the result is a dictionary
            assert isinstance(result, dict)
            
            # Check that the dictionary contains the required keys
            assert "torsion_angles" in result
            assert "residue_count" in result
            
            # Check that the tensor shape matches the sequence length
            torsion_angles = result["torsion_angles"]
            assert torsion_angles.shape == (len(seq), 14)
            
            # Check that the residue_count matches the sequence length
            assert result["residue_count"] == len(seq)

    def test_real_model_if_available(self, real_predictor):
        """
        Test with the real model if it's available.
        This test will be skipped if the model can't be loaded.
        """
        if real_predictor is None:
            pytest.skip("Real model not available")
        
        # Test with a simple RNA sequence
        sequence = "ACGU"
        try:
            result = real_predictor(sequence)
            
            # Check that the result is a dictionary
            assert isinstance(result, dict)
            
            # Check that the dictionary contains the required keys
            assert "torsion_angles" in result
            assert "residue_count" in result
            
            # Check that the tensor shape matches the sequence length
            torsion_angles = result["torsion_angles"]
            assert torsion_angles.shape[0] == len(sequence)
            
            # The shape[1] depends on the angle_mode and the actual model output
            if real_predictor.angle_mode == "sin_cos":
                # For sin_cos mode, we expect an even number of columns (for sin/cos pairs)
                assert torsion_angles.shape[1] % 2 == 0, "Expected even number of columns for sin/cos pairs"
                # Store the actual number of angles for informational purposes
                actual_num_angles = torsion_angles.shape[1] // 2
                print(f"Model outputs {actual_num_angles} angles ({torsion_angles.shape[1]} columns)")
            else:
                # For radians/degrees mode, we expect the number of columns to match the number of angles
                actual_num_angles = torsion_angles.shape[1]
                print(f"Model outputs {actual_num_angles} angles")
            
            # Check for NaN/Inf values
            assert not torch.any(torch.isnan(torsion_angles))
            assert not torch.any(torch.isinf(torsion_angles))
            
            # Check that the residue_count matches the sequence length
            assert result["residue_count"] == len(sequence)
            
            print(f"Real model test passed with shape {torsion_angles.shape}")
        except Exception as e:
            pytest.fail(f"Real model test failed: {e}")