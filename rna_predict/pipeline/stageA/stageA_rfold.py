import torch
import numpy as np

# Adjust these imports based on where your code is located.
# For example, if RFold_Model is in rna_predict/pipeline/model.py, do:
# from rna_predict.pipeline.model import RFold_Model
# Or if it's in rna_predict.models.rfold.model, adapt accordingly.
from rna_predict.pipeline.model import RFold_Model

# If row_col_argmax, constraint_matrix, process_seqs are in colab_utils or rfold.py, update as needed:
from rna_predict.scripts.colab_utils import (
    process_seqs,
    row_col_argmax,
    constraint_matrix
)

class StageARFoldPredictor:
    """
    Stage A: RFold-based predictor for RNA secondary structure.
    Loads a pre-trained RFold model and provides predict_adjacency().
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        :param checkpoint_path: path to the trained checkpoint for RFold
        :param device: "cuda" or "cpu"
        """
        # Example: if you need certain hyperparams, adapt as needed:
        # self.model = RFold_Model(num_hidden=128, dropout=0.3, etc.)
        # Or if you load from JSON config, do so here.
        self.model = RFold_Model(...)  # Insert appropriate constructor args
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load trained weights
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_data)
        print(f"[StageARFoldPredictor] Loaded checkpoint from {checkpoint_path}")

    def predict_adjacency(self, rna_sequence: str) -> np.ndarray:
        """
        Convert an RNA sequence string -> adjacency matrix [N, N] via RFold.
        :param rna_sequence: e.g. "AUGCUAG..."
        :return: adjacency matrix (np.ndarray) of shape [N, N].
        """
        # 1) Preprocess the sequence into numeric form + optional padding
        nseq, nseq_one_hot, seq_len = process_seqs(rna_sequence, self.device)
        # nseq : [1, padded_len]
        # nseq_one_hot : [1, padded_len, 4]

        # 2) Forward pass through the RFold model
        with torch.no_grad():
            raw_pred = self.model(nseq)  # shape [1, padded_len, padded_len]

        # 3) row/col factorization + constraints
        # raw_pred[0] => [padded_len, padded_len]
        pred_bin = row_col_argmax(raw_pred[0]) * constraint_matrix(nseq_one_hot)

        # 4) Slice back to original seq_len
        final_adj = pred_bin[:seq_len, :seq_len].cpu().numpy()

        return final_adj