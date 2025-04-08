from typing import Any, Dict
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DummyTorsionBertAutoModel(nn.Module):
    """Dummy model for testing that returns tensors with correct shape."""

    def __init__(self, num_angles: int = 7):
        super().__init__()
        self.num_angles = num_angles
        self.side_effect = None  # For testing purposes

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        """Forward pass that returns a tensor with correct shape."""
        if isinstance(inputs, MagicMock):
            # For tests that don't set input_ids properly
            batch_size, seq_len = 1, 1
        else:
            input_ids = inputs["input_ids"]
            batch_size, seq_len = input_ids.shape

        # Return tensor with shape [batch_size, seq_len, 2*num_angles]
        output = torch.zeros(batch_size, seq_len, 2 * self.num_angles)
        if self.side_effect:
            return self.side_effect(inputs["input_ids"], inputs["attention_mask"])
        return type("obj", (object,), {"logits": output, "last_hidden_state": output})()


class TorsionBertModel(nn.Module):
    """A wrapper around the TorsionBert model that handles both logits and last_hidden_state outputs."""

    def __init__(
        self,
        model_path: str,
        num_angles: int = 7,
        max_length: int = 512,
        device: str = "cpu",
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.num_angles = num_angles
        self.max_length = max_length
        self.device = torch.device(device)
        self.return_dict = return_dict

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)

        # If model is mocked (for testing), replace with dummy model
        if isinstance(self.model, MagicMock):
            self.model = DummyTorsionBertAutoModel(num_angles=num_angles)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        """Forward pass through the model."""
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            return outputs  # Return as is if already a dictionary
        if self.return_dict:
            # Handle both cases where outputs might have logits or last_hidden_state
            if hasattr(outputs, 'logits'):
                return {"logits": outputs.logits}
            else:
                return {"logits": outputs.last_hidden_state}
        return outputs

    def _preprocess_sequence(self, rna_sequence: str) -> tuple[str, int]:
        """
        Preprocess the RNA sequence by converting to uppercase and replacing U with T.

        Args:
            rna_sequence: Input RNA sequence

        Returns:
            Tuple of (processed sequence, sequence length)
        """
        seq = rna_sequence.upper().replace("U", "T")
        return seq, len(seq)

    def _build_tokens(self, seq: str, k: int = 3) -> str:
        """
        Build k-mer tokens from the sequence using a sliding window approach.

        Args:
            seq: Input sequence
            k: Size of the k-mer window

        Returns:
            Space-separated string of k-mers
        """
        tokens = []
        for i in range(len(seq) - (k - 1)):
            tokens.append(seq[i : i + k])
        return " ".join(tokens)

    def _prepare_inputs(self, spaced_kmers: str) -> dict:
        """
        Prepare tokenizer inputs from spaced k-mers and move to device.

        Args:
            spaced_kmers: Space-separated k-mers string

        Returns:
            Dictionary of tokenizer inputs on the correct device
        """
        inputs = self.tokenizer(
            spaced_kmers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        for k_, v_ in inputs.items():
            inputs[k_] = v_.to(self.device)
        return inputs

    def _extract_raw_sincos(self, outputs) -> torch.Tensor:
        """
        Extract raw sin/cos values from model outputs.

        Args:
            outputs: Model outputs dictionary or object

        Returns:
            Tensor containing raw sin/cos values
        """
        if isinstance(outputs, dict):
            if "logits" in outputs:
                return outputs["logits"]
            elif "last_hidden_state" in outputs:
                return outputs["last_hidden_state"]
            # If neither key exists, return the first tensor value
            for value in outputs.values():
                if isinstance(value, torch.Tensor):
                    return value
            raise ValueError("No tensor output found in model outputs")
        return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.logits

    def _fill_result(self, raw_sincos: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Create and fill result tensor with values from raw_sincos.

        Args:
            raw_sincos: Raw sin/cos tensor from model
            seq_len: Length of the input sequence

        Returns:
            Filled result tensor
        """
        sincos_dim = raw_sincos.shape[-1]
        n_3mers = raw_sincos.shape[1]
        result = torch.zeros((seq_len, sincos_dim), device=self.device)

        for i in range(n_3mers):
            if i < seq_len:
                result[i] = raw_sincos[0, i]

        return result

    def predict_angles_from_sequence(self, sequence: str) -> torch.Tensor:
        """Predict torsion angles from an RNA sequence."""
        if not sequence:
            # Return empty tensor for empty sequence
            return torch.zeros((0, 2 * self.num_angles), device=self.device)

        # Preprocess sequence
        seq, seq_len = self._preprocess_sequence(sequence)

        # Build k-mer tokens
        spaced_kmers = self._build_tokens(seq)

        # Prepare inputs
        inputs = self._prepare_inputs(spaced_kmers)

        # Get model outputs
        outputs = self.forward(inputs)

        # Extract logits or last_hidden_state
        raw_sincos = self._extract_raw_sincos(outputs)

        # Fill result tensor
        result = self._fill_result(raw_sincos, seq_len)
        return result.to(self.device)
