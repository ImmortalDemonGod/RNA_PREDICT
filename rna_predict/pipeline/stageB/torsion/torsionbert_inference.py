import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



class TorsionBertModel(nn.Module):
    """
    A wrapper around a pre-trained TorsionBERT model that outputs
    backbone torsion angles (commonly as sin/cos pairs).
    The 'num_angles' constructor arg is mainly for reference or validation,
    but actual dimension might differ in the loaded model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        num_angles: int = 7,
        max_length: int = 512,
    ):
        """
        Args:
            model_name_or_path: HF Hub ID or local path, e.g. "sayby/rna_torsionbert"
            device: torch.device object
            num_angles: user-supplied guess or config
            max_length: max tokenizer length
        """
        super().__init__()
        self.device = device
        self.user_requested_num_angles = num_angles
        self.max_length = max_length

        # Load HF objects
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def forward(self, inputs):
        """
        Generic forward that calls self.model(inputs).
        Usually returns a dict with 'logits' or an object with .last_hidden_state
        """
        return self.model(inputs)

    def predict_angles_from_sequence(self, rna_sequence: str) -> torch.Tensor:
        """
        Convert an RNA seq to sin/cos angle pairs as a [N, sincos_dim] tensor,
        where sincos_dim = 2*NmodelAngles from the loaded model.
        If the seq is empty or has no valid k-mer tokens, returns a zero tensor of
        shape (seq_len, 2 * self.user_requested_num_angles).

        We do a partial fill of the result, row i => raw_sincos[0, i], if i < seq_len.
        """
        seq = rna_sequence.upper().replace("U", "T")
        seq_len = len(seq)
        if seq_len == 0:
            # Return a zero-length result
            return torch.zeros(
                (0, 2 * self.user_requested_num_angles), device=self.device
            )

        # Build 3-mers by sliding window of 3
        k = 3
        tokens = []
        for i in range(seq_len - (k - 1)):
            tokens.append(seq[i : i + k])
        spaced_kmers = " ".join(tokens)

        if not spaced_kmers:
            # No valid 3-mer => no data
            return torch.zeros(
                (seq_len, 2 * self.user_requested_num_angles), device=self.device
            )

        # Prepare inputs
        inputs = self.tokenizer(
            spaced_kmers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        # Move to device
        for k_, v_ in inputs.items():
            inputs[k_] = v_.to(self.device)

        # Inference
        outputs = self.forward(inputs)

        if "logits" in outputs:
            raw_sincos = outputs["logits"]
        else:
            raw_sincos = outputs.last_hidden_state

        # raw_sincos => shape [batch=1, n_3mers, sincos_dim]
        # We'll create [seq_len, sincos_dim]
        sincos_dim = raw_sincos.shape[-1]
        n_3mers = raw_sincos.shape[1]

        result = torch.zeros((seq_len, sincos_dim), device=self.device)

        # Fill row i => raw_sincos[0, i], if i < seq_len
        for i in range(n_3mers):
            if i < seq_len:
                result[i] = raw_sincos[0, i]

        return result
