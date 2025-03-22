import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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

    def forward(self, inputs):
        """
        Forward pass that passes 'inputs' as a single dictionary to the remote-coded TorsionBERT model.
        This avoids the 'unexpected keyword argument' error from passing named parameters.
        """
        return self.model(inputs)

    def predict_angles_from_sequence(self, rna_sequence: str) -> torch.Tensor:
        """
        Custom method that takes a raw RNA sequence,
        tokenizes into 3-mers, and calls the Hugging Face model.
        Returns a [seq_len, 2 * self.num_angles] tensor of sin/cos pairs
        or zeros if the sequence length < 1.
        """
        seq = rna_sequence.upper().replace("U", "T")
        seq_len = len(seq)
        if seq_len == 0:
            return torch.zeros((0, 2 * self.num_angles), device=self.device)

        # Build 3-mer tokens by sliding a window of size 3
        tokens = []
        k = 3
        for i in range(seq_len - (k - 1)):
            tokens.append(seq[i : i + k])
        spaced_kmers = " ".join(tokens)

        if not spaced_kmers:
            return torch.zeros((seq_len, 2 * self.num_angles), device=self.device)

        inputs = self.tokenizer(
            spaced_kmers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        # Move tokenizer outputs to the appropriate device
        for key_, val_ in inputs.items():
            inputs[key_] = val_.to(self.device)

        # Now we call the model by passing the entire dictionary as a single argument
        outputs = self.forward(inputs)

        # By convention, TorsionBERT might store predictions in outputs["logits"]
        # If not, we fall back to outputs.last_hidden_state
        if "logits" in outputs:
            raw_sincos = outputs["logits"]
        else:
            raw_sincos = outputs.last_hidden_state

        # Allocate space for the final sin/cos angles (size [seq_len, 2 * num_angles])
        result = torch.zeros((seq_len, 2 * self.num_angles), device=self.device)
        # Fill each residue row with the corresponding output
        for i in range(raw_sincos.shape[1]):
            mid_idx = i + 1
            if mid_idx < seq_len:
                result[mid_idx] = raw_sincos[0, i]

        return result