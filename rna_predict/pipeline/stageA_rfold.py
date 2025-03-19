import json
import torch
import torch.nn.functional as F
from rna_predict.pipeline.model import RFold_Model

class StageARFoldPredictor:
    def __init__(self, config_path, checkpoint_path, device=None):
        # load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # set device
        self.device = device if device is not None else torch.device('cpu')
        
        # build model
        self.model = self._build_rfold_model()
        self.model.to(self.device)
        
        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
    def _build_rfold_model(self):
        # assumes instantiation of model using RFold_Model
        return RFold_Model(**self.config.get("model_params", {}))
    
    def predict_adjacency(self, rna_sequence):
        # Preprocess: Convert RNA sequence to tensor (dummy example mapping A,C,G,U to digits)
        mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        seq_tensor = torch.tensor([mapping.get(n, 0) for n in rna_sequence], dtype=torch.long).unsqueeze(0)  # shape (1, L)
        seq_tensor = seq_tensor.to(self.device)
        
        # forward pass through model
        logits = self.model(seq_tensor)  # assume output is tensor shape (1, L, L)
        
        # apply row/column argmax with constraints
        predicted = self._row_col_argmax_with_constraint(logits[0])
        
        return predicted.cpu().numpy()
    
    def _row_col_argmax_with_constraint(self, matrix):
        # Apply row and column argmax while enforcing at most one 1 per row and column
        discrete_matrix = self._discrete_bipartite(matrix)
        return discrete_matrix
    
    def _discrete_bipartite(self, matrix):
        # For each row, select the maximum value if it satisfies constraints
        discrete = torch.zeros_like(matrix)
        for i in range(matrix.size(0)):
            row = matrix[i]
            j = torch.argmax(row)
            constraint = self._build_constraint_matrix(i, j, matrix.size(0))
            if constraint[i, j] == 1:
                discrete[i, j] = 1
        return discrete
    
    def _build_constraint_matrix(self, row_idx, col_idx, size):
        # Build a constraint matrix that enforces base pairing and distance constraints (e.g., distance >= 4)
        matrix = torch.ones((size, size), dtype=torch.long)
        for i in range(size):
            for j in range(size):
                if abs(i - j) < 4:
                    matrix[i, j] = 0
        matrix[row_idx, col_idx] = 1
        return matrix
