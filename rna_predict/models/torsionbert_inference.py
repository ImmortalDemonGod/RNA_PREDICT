import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Optional

class TorsionBertModel(nn.Module):
    """
    A wrapper around a pre-trained TorsionBERT model that outputs
    backbone torsion angles (commonly as sin/cos pairs).
    """
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        num_angles: int = 7,
        max_length: int = 512
    ):
        """
        Args:
            model_name_or_path: HF Hub ID (e.g. "sayby/rna_torsionbert") or local path.
            device: torch.device object, e.g. torch.device("cpu" or "cuda").
            num_angles: number of backbone angles (commonly 7 for alpha..chi).
            max_length: tokenizer maximum length (often 512).
        """
        super().__init__()
        self.device = device
        self.num_angles = num_angles
        self.max_length = max_length

        # Load tokenizer & model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def forward(self, rna_sequence: str) -> torch.Tensor:
        """
        Forward pass: convert a raw RNA sequence to sin/cos angle predictions.
        Returns:
            A torch.Tensor of shape [N, 2 * num_angles].
            If the input length < 3, returns shape [N, 2 * num_angles] but zeros.
        """
        seq = rna_sequence.upper().replace("U", "T")
        seq_len = len(seq)
        if seq_len == 0:
            return torch.zeros((0, 2 * self.num_angles), device=self.device)

        # Build 3-mer tokens by sliding a window of size 3
        tokens = []
        k = 3
        for i in range(seq_len - (k - 1)):
            tokens.append(seq[i:i+k])
        spaced_kmers = " ".join(tokens)
        if not spaced_kmers:
            # For sequences with length 1 or 2
            return torch.zeros((seq_len, 2 * self.num_angles), device=self.device)

        # Tokenize and push inputs to device
        inputs = self.tokenizer(
            spaced_kmers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        # Model inference: output either 'logits' or last_hidden_state
        # Call the model with positional arguments instead of **inputs
        # so we don't pass 'input_ids' as a named argument:
        if "token_type_ids" in inputs:
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"]
            )
        else:
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        # By convention, TorsionBERT might store predictions in outputs["logits"]
        if "logits" in outputs:
            raw_sincos = outputs["logits"]  # shape [1, T, 2*num_angles]
        else:
            raw_sincos = outputs.last_hidden_state  # e.g. shape [1, T, 2*num_angles]
        # Map each 3-mer token to the middle residue: token i => residue (i+1)
        result = torch.zeros((seq_len, 2 * self.num_angles), device=self.device)
        for i in range(T):
            mid_idx = i + 1
            if mid_idx < seq_len:
                result[mid_idx] = raw_sincos[i]
        return result