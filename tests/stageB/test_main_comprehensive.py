"""
Comprehensive tests for the main.py module in Stage B.

This module tests the functions in rna_predict/pipeline/stageB/main.py,
including run_pipeline, run_stageB_combined, and demo_gradient_flow_test.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from rna_predict.pipeline.stageB.main import (
    run_pipeline,
    run_stageB_combined,
    demo_gradient_flow_test,
    main,
)
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper


class TestRunPipeline:
    """Tests for the run_pipeline function."""

    @patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageCReconstruction")
    def test_run_pipeline_basic(self, mock_stageC, mock_stageB, mock_stageA):
        """Test that run_pipeline runs without errors and returns expected output."""
        # Set up mocks
        mock_stageA_instance = mock_stageA.return_value
        mock_stageA_instance.predict_adjacency.return_value = torch.ones((8, 8)).numpy()

        mock_stageB_instance = mock_stageB.return_value
        mock_stageB_instance.return_value = {"torsion_angles": torch.ones((8, 7))}

        mock_stageC_instance = mock_stageC.return_value
        mock_stageC_instance.return_value = {
            "coords": torch.ones((8, 10, 3)),
            "atom_count": 10,
        }

        # Run the function
        run_pipeline("ACGUACGU")

        # Verify that the mocks were called with the expected arguments
        mock_stageA.assert_called_once_with(config={})
        mock_stageA_instance.predict_adjacency.assert_called_once_with("ACGUACGU")

        mock_stageB.assert_called_once_with(
            model_name_or_path="sayby/rna_torsionbert",
            device="cpu",
            angle_mode="degrees",
            num_angles=7,
            max_length=512,
        )
        mock_stageB_instance.assert_called_once()

        mock_stageC.assert_called_once()
        mock_stageC_instance.assert_called_once()

    @patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageCReconstruction")
    def test_run_pipeline_empty_sequence(self, mock_stageC, mock_stageB, mock_stageA):
        """Test run_pipeline with an empty sequence."""
        # Set up mocks
        mock_stageA_instance = mock_stageA.return_value
        mock_stageA_instance.predict_adjacency.return_value = torch.zeros((0, 0)).numpy()

        mock_stageB_instance = mock_stageB.return_value
        mock_stageB_instance.return_value = {"torsion_angles": torch.zeros((0, 7))}

        mock_stageC_instance = mock_stageC.return_value
        mock_stageC_instance.return_value = {
            "coords": torch.zeros((0, 10, 3)),
            "atom_count": 0,
        }

        # Run the function
        run_pipeline("")

        # Verify that the mocks were called with the expected arguments
        mock_stageA.assert_called_once_with(config={})
        mock_stageA_instance.predict_adjacency.assert_called_once_with("")

    @patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageCReconstruction")
    def test_run_pipeline_invalid_sequence(self, _, __, mock_stageA):
        """Test run_pipeline with an invalid sequence."""
        # Set up mocks
        mock_stageA_instance = mock_stageA.return_value
        mock_stageA_instance.predict_adjacency.side_effect = ValueError("Invalid sequence")

        # Run the function and check that it raises the expected exception
        with pytest.raises(ValueError, match="Invalid sequence"):
            run_pipeline("XYZ")  # Invalid RNA sequence

        # Verify that the mocks were called with the expected arguments
        mock_stageA.assert_called_once_with(config={})
        mock_stageA_instance.predict_adjacency.assert_called_once_with("XYZ")

    @patch("rna_predict.pipeline.stageB.main.StageARFoldPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.main.StageCReconstruction")
    def test_run_pipeline_long_sequence(self, mock_stageC, mock_stageB, mock_stageA):
        """Test run_pipeline with a long sequence."""
        # Set up mocks
        mock_stageA_instance = mock_stageA.return_value
        mock_stageA_instance.predict_adjacency.return_value = torch.ones((100, 100)).numpy()

        mock_stageB_instance = mock_stageB.return_value
        mock_stageB_instance.return_value = {"torsion_angles": torch.ones((100, 7))}

        mock_stageC_instance = mock_stageC.return_value
        mock_stageC_instance.return_value = {
            "coords": torch.ones((100, 10, 3)),
            "atom_count": 10,
        }

        # Run the function
        run_pipeline("A" * 100)

        # Verify that the mocks were called with the expected arguments
        mock_stageA.assert_called_once_with(config={})
        mock_stageA_instance.predict_adjacency.assert_called_once_with("A" * 100)

        mock_stageB.assert_called_once_with(
            model_name_or_path="sayby/rna_torsionbert",
            device="cpu",
            angle_mode="degrees",
            num_angles=7,
            max_length=512,
        )
        mock_stageB_instance.assert_called_once()

        mock_stageC.assert_called_once()
        mock_stageC_instance.assert_called_once()


class TestRunStageBCombined:
    """Tests for the run_stageB_combined function."""

    def setup_method(self):
        """
        Initialize test fixture attributes for model testing.
        
        Sets up a sample RNA sequence and its corresponding 8x8 adjacency matrix.
        Creates mock instances for the torsion BERT predictor and pairformer wrapper with
        preset configurations to simulate model outputs, and assigns the computation
        device.
        """
        self.sequence = "ACGUACGU"
        self.adjacency_matrix = torch.ones((8, 8))
        self.torsion_bert_model = MagicMock(spec=StageBTorsionBertPredictor)
        self.torsion_bert_model.return_value = {"torsion_angles": torch.ones((8, 7))}
        self.pairformer_model = MagicMock(spec=PairformerWrapper)
        self.pairformer_model.c_s = 64
        self.pairformer_model.c_z = 32
        self.pairformer_model.return_value = (
            torch.ones((1, 8, 64)),  # s_up
            torch.ones((1, 8, 8, 32)),  # z_up
        )
        self.device = "cpu"

    def test_run_stageB_combined_basic(self):
        """Test that run_stageB_combined runs without errors and returns expected output structure."""
        result = run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency_matrix,
            torsion_bert_model=self.torsion_bert_model,
            pairformer_model=self.pairformer_model,
            device=self.device,
            init_z_from_adjacency=False,
        )

        # Check that all expected keys are present
        assert "torsion_angles" in result
        assert "s_embeddings" in result
        assert "z_embeddings" in result

        # Check shapes
        N = len(self.sequence)
        assert result["torsion_angles"].shape[0] == N
        assert result["s_embeddings"].shape == (N, self.pairformer_model.c_s)
        assert result["z_embeddings"].shape == (N, N, self.pairformer_model.c_z)

        # Verify that the torsion_bert_model was called
        self.torsion_bert_model.assert_called_once()
        self.pairformer_model.assert_called_once()

    def test_run_stageB_combined_with_adjacency_init(self):
        """Test run_stageB_combined with adjacency-based initialization."""
        result = run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency_matrix,
            torsion_bert_model=self.torsion_bert_model,
            pairformer_model=self.pairformer_model,
            device=self.device,
            init_z_from_adjacency=True,
        )

        # Check that all expected keys are present
        assert "torsion_angles" in result
        assert "s_embeddings" in result
        assert "z_embeddings" in result

        # Check shapes
        N = len(self.sequence)
        assert result["torsion_angles"].shape[0] == N
        assert result["s_embeddings"].shape == (N, self.pairformer_model.c_s)
        assert result["z_embeddings"].shape == (N, N, self.pairformer_model.c_z)

        # Verify that the torsion_bert_model was called
        self.torsion_bert_model.assert_called_once()
        self.pairformer_model.assert_called_once()

    def test_run_stageB_combined_empty_sequence(self):
        """Test run_stageB_combined with an empty sequence."""
        # Set up mocks for empty sequence
        empty_sequence = ""
        empty_adjacency_matrix = torch.zeros((0, 0))
        self.torsion_bert_model.return_value = {"torsion_angles": torch.zeros((0, 7))}
        self.pairformer_model.return_value = (
            torch.zeros((1, 0, 64)),  # s_up
            torch.zeros((1, 0, 0, 32)),  # z_up
        )

        # Run the function
        # Empty sequences should be handled gracefully
        result = run_stageB_combined(
            sequence=empty_sequence,
            adjacency_matrix=empty_adjacency_matrix,
            torsion_bert_model=self.torsion_bert_model,
            pairformer_model=self.pairformer_model,
            device=self.device,
            init_z_from_adjacency=False,
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
        """
        Verify that run_stageB_combined assigns the correct device to the models.
        
        This test ensures that both the torsion model's internal model and the pairformer model
        are moved to the specified device by calling their respective to() methods with the
        device argument.
        """
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
            torch.ones((1, 8, 64)),  # s_up
            torch.ones((1, 8, 8, 32)),  # z_up
        )

        # Run the function
        run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=self.adjacency_matrix,
            torsion_bert_model=torsion_model,
            pairformer_model=pairformer_model,
            device="cpu",  # Use the same device for testing
            init_z_from_adjacency=False,
        )

        # Verify that the models were moved to the correct device
        pairformer_model.to.assert_called_once_with("cpu")
        torsion_model.model.to.assert_called_once_with("cpu")

    def test_run_stageB_combined_shape_mismatch(self):
        """Test run_stageB_combined with a shape mismatch between sequence and adjacency."""
        # Set up mismatch
        mismatched_adjacency = torch.ones((10, 10))  # Sequence is length 8, adjacency is 10x10

        # The function should handle the mismatch gracefully
        result = run_stageB_combined(
            sequence=self.sequence,
            adjacency_matrix=mismatched_adjacency,
            torsion_bert_model=self.torsion_bert_model,
            pairformer_model=self.pairformer_model,
            device=self.device,
            init_z_from_adjacency=False,
        )

        # Check that the result has the expected structure
        assert "torsion_angles" in result
        assert "s_embeddings" in result
        assert "z_embeddings" in result

        # Check shapes - they should match the sequence length, not the adjacency matrix
        N = len(self.sequence)
        assert result["torsion_angles"].shape[0] == N
        assert result["s_embeddings"].shape == (N, self.pairformer_model.c_s)
        assert result["z_embeddings"].shape == (N, N, self.pairformer_model.c_z)


class TestDemoGradientFlowTest:
    """Tests for the demo_gradient_flow_test function."""

    @patch("rna_predict.pipeline.stageB.main.torch.device")
    @patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.main.PairformerWrapper")
    @patch("rna_predict.pipeline.stageB.main.run_stageB_combined")
    @patch("rna_predict.pipeline.stageB.main.torch.nn.Linear")
    @patch("rna_predict.pipeline.stageB.main.F.mse_loss")
    def test_demo_gradient_flow_test_basic(self, mock_loss, mock_linear, mock_run_stageB, mock_pairformer, mock_torsion, mock_device):
        """Test that demo_gradient_flow_test runs without errors and performs the expected operations."""
        # Set up device mock
        mock_device.return_value = "cpu"

        # Set up torsion model mock
        mock_torsion_instance = MagicMock()
        mock_torsion_instance.model = MagicMock()
        mock_torsion_instance.model.zero_grad = MagicMock()
        mock_torsion_instance.model.named_parameters = MagicMock(
            return_value=[("param1", torch.ones(1, requires_grad=True))]
        )
        mock_torsion.return_value = mock_torsion_instance

        # Set up pairformer mock
        mock_pairformer_instance = MagicMock()
        mock_pairformer_instance.to = MagicMock(return_value=mock_pairformer_instance)
        mock_pairformer_instance.zero_grad = MagicMock()
        mock_pairformer_instance.named_parameters = MagicMock(
            return_value=[("param1", torch.ones(1, requires_grad=True))]
        )
        mock_pairformer.return_value = mock_pairformer_instance

        # Set up run_stageB_combined mock
        mock_run_stageB.return_value = {
            "torsion_angles": torch.ones((8, 7), requires_grad=True),
            "s_embeddings": torch.ones((8, 64), requires_grad=True),
            "z_embeddings": torch.ones((8, 8, 32), requires_grad=True),
        }

        # Set up linear mocks
        linear_instances = [MagicMock() for _ in range(3)]
        for i, instance in enumerate(linear_instances):
            instance.to = MagicMock(return_value=instance)
            instance.zero_grad = MagicMock()
            instance.named_parameters = MagicMock(
                return_value=[("weight", torch.ones((3, 64 if i == 0 else 7 if i == 1 else 32), requires_grad=True)),
                              ("bias", torch.ones(3, requires_grad=True))]
            )
            instance.return_value = torch.ones((8, 3), requires_grad=True)
        mock_linear.side_effect = linear_instances

        # Set up loss mock
        loss_tensor = torch.tensor(1.0, requires_grad=True)
        loss_tensor.backward = MagicMock()
        mock_loss.return_value = loss_tensor

        # Run the function
        demo_gradient_flow_test()

        # Verify that the mocks were called with the expected arguments
        mock_torsion.assert_called_once()
        mock_pairformer.assert_called_once_with(n_blocks=2, c_z=32, c_s=64, dropout=0.1)
        mock_pairformer_instance.to.assert_called_once()
        mock_run_stageB.assert_called_once()
        assert mock_linear.call_count == 3
        mock_loss.assert_called_once()

        # Verify that zero_grad was called
        mock_torsion_instance.model.zero_grad.assert_called_once()
        mock_pairformer_instance.zero_grad.assert_called_once()
        for instance in linear_instances:
            instance.zero_grad.assert_called_once()

        # Verify that backward was called
        loss_tensor.backward.assert_called_once()

    @patch("rna_predict.pipeline.stageB.main.torch.device")
    @patch("rna_predict.pipeline.stageB.main.StageBTorsionBertPredictor")
    @patch("rna_predict.pipeline.stageB.main.PairformerWrapper")
    def test_demo_gradient_flow_test_exception_handling(self, mock_pairformer, mock_torsion, mock_device):
        """Test that demo_gradient_flow_test handles exceptions correctly."""
        # Set up device mock
        mock_device.return_value = "cpu"

        # Set up torsion model to raise an exception on first call, then succeed on second call
        mock_torsion.side_effect = [
            Exception("Could not load model"),  # First call raises exception
            MagicMock()  # Second call returns a mock
        ]

        # Set up the second mock instance that will be returned after the exception
        second_instance = MagicMock()
        second_instance.model = MagicMock()
        second_instance.model.zero_grad = MagicMock()
        second_instance.model.named_parameters = MagicMock(return_value=[])
        mock_torsion.return_value = second_instance

        # Set up pairformer mock
        mock_pairformer_instance = MagicMock()
        mock_pairformer_instance.to = MagicMock(return_value=mock_pairformer_instance)
        mock_pairformer_instance.zero_grad = MagicMock()
        mock_pairformer_instance.named_parameters = MagicMock(return_value=[])
        mock_pairformer.return_value = mock_pairformer_instance

        # Run the function with patched run_stageB_combined and other dependencies
        with patch("rna_predict.pipeline.stageB.main.run_stageB_combined") as mock_run_stageB, \
             patch("rna_predict.pipeline.stageB.main.torch.nn.Linear") as mock_linear, \
             patch("rna_predict.pipeline.stageB.main.F.mse_loss") as mock_loss:

            # Set up run_stageB_combined mock
            mock_run_stageB.return_value = {
                "torsion_angles": torch.ones((8, 7)),
                "s_embeddings": torch.ones((8, 64)),
                "z_embeddings": torch.ones((8, 8, 32)),
            }

            # Set up linear mocks
            linear_instances = [MagicMock() for _ in range(3)]
            for instance in linear_instances:
                instance.to = MagicMock(return_value=instance)
                instance.zero_grad = MagicMock()
                instance.return_value = torch.ones((8, 3))
            mock_linear.side_effect = linear_instances

            # Set up loss mock
            loss_tensor = torch.tensor(1.0)
            loss_tensor.backward = MagicMock()
            mock_loss.return_value = loss_tensor

            # Run the function
            demo_gradient_flow_test()

        # Verify that the mocks were called with the expected arguments
        assert mock_torsion.call_count == 2
        mock_torsion.assert_any_call(model_name_or_path="sayby/rna_torsionbert", device=mock_device.return_value)
        mock_torsion.assert_called_with(model_name_or_path="dummy_invalid_path", device=mock_device.return_value)


class TestMain:
    """Tests for the main function."""

    @patch("rna_predict.pipeline.stageB.main.run_pipeline")
    @patch("rna_predict.pipeline.stageB.main.demo_gradient_flow_test")
    def test_main_basic(self, mock_demo, mock_run_pipeline):
        """Test that main runs without errors."""
        # Run the function
        main()

        # Verify that the mocks were called with the expected arguments
        mock_run_pipeline.assert_called_once_with("ACGUACGU")
        mock_demo.assert_called_once()

    @patch("rna_predict.pipeline.stageB.main.run_pipeline")
    @patch("rna_predict.pipeline.stageB.main.demo_gradient_flow_test")
    def test_main_run_pipeline_exception(self, mock_demo, mock_run_pipeline):
        """Test that main handles exceptions from run_pipeline correctly."""
        # Set up mocks to raise an exception
        mock_run_pipeline.side_effect = Exception("Error in run_pipeline")

        # Run the function and check that it raises the expected exception
        with pytest.raises(Exception, match="Error in run_pipeline"):
            main()

        # Verify that the mocks were called with the expected arguments
        mock_run_pipeline.assert_called_once_with("ACGUACGU")
        mock_demo.assert_not_called()

    @patch("rna_predict.pipeline.stageB.main.run_pipeline")
    @patch("rna_predict.pipeline.stageB.main.demo_gradient_flow_test")
    def test_main_demo_exception(self, mock_demo, mock_run_pipeline):
        """Test that main handles exceptions from demo_gradient_flow_test correctly."""
        # Set up mocks to raise an exception
        mock_demo.side_effect = Exception("Error in demo_gradient_flow_test")

        # Run the function and check that it raises the expected exception
        with pytest.raises(Exception, match="Error in demo_gradient_flow_test"):
            main()

        # Verify that the mocks were called with the expected arguments
        mock_run_pipeline.assert_called_once_with("ACGUACGU")
        mock_demo.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", "test_main_comprehensive.py"])
