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
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.user_requested_num_angles = num_angles
        self.max_length = max_length
        self.num_angles = num_angles

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
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        return outputs.last_hidden_state

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

    def predict_angles_from_sequence(self, rna_sequence: str) -> torch.Tensor:
        """
        Convert an RNA seq to sin/cos angle pairs as a [N, sincos_dim] tensor,
        where sincos_dim = 2*NmodelAngles from the loaded model.
        If the seq is empty or has no valid k-mer tokens, returns a zero tensor of
        shape (seq_len, 2 * self.user_requested_num_angles).

        We do a partial fill of the result, row i => raw_sincos[0, i], if i < seq_len.
        """
        seq, seq_len = self._preprocess_sequence(rna_sequence)
        if seq_len == 0:
            return torch.zeros(
                (0, 2 * self.user_requested_num_angles), device=self.device
            )

        spaced_kmers = self._build_tokens(seq)
        if not spaced_kmers:
            return torch.zeros(
                (seq_len, 2 * self.user_requested_num_angles), device=self.device
            )

        inputs = self._prepare_inputs(spaced_kmers)
        outputs = self.forward(inputs)
        raw_sincos = self._extract_raw_sincos(outputs)
        return self._fill_result(raw_sincos, seq_len)
