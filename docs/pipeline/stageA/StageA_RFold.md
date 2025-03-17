Technical Documentation: Integrating RFold into a Multi-Stage RNA 3D Pipeline
Focusing on Stage A (2D Structure/Adjacency)

⸻

1. Overview

This document outlines how to incorporate RFold—a K-Rook-based RNA secondary structure predictor—into Stage A of a multi-stage RNA 3D pipeline. In this pipeline, Stage A is responsible for predicting base pairs (2D adjacency or contact maps) from an RNA sequence. The subsequent stages (e.g., torsion angle prediction and 3D reconstruction) use these adjacency maps as inputs. By using RFold in Stage A, you can guarantee valid secondary structures and streamline the overall workflow.

⸻

2. Why Use RFold for Stage A?

Key Reasons
	1.	Validity Guarantee: RFold treats RNA folding as a symmetric K-Rook matching problem, ensuring each nucleotide pairs at most once and obeys base-type/distance constraints. Other neural methods often require extra steps or iterative relaxations to approximate constraints.
	2.	High Accuracy: RFold achieves superior F1 scores across multiple RNA datasets (e.g., RNAStralign, ArchiveII, bpRNA), particularly excelling in precision.
	3.	Fast Inference: RFold typically runs in a fraction of the time compared to many other tools (approx. 0.02 s per sequence on GPU). This is highly beneficial if you need to fold large numbers of RNAs quickly.
	4.	Straightforward Integration: RFold’s output is simply an N×N adjacency (or probability) matrix. You can feed this directly into your Stage B (torsion prediction), then Stage C (3D reconstruction).

⸻

3. Stage A: 2D Structure / Adjacency

Recall: In a multi-stage RNA pipeline, Stage A receives an RNA sequence of length N and outputs a contact map (an N×N matrix, M[i,j] = 1 for a predicted pair).

<details>
<summary>Constraints typically enforced</summary>


	•	(a) Base-type: Only A–U, G–C, or G–U pairs permitted.
	•	(b) Minimum Loop Length: E.g., no pairs between indices i, j if |i−j| < 4.
	•	(c) One Pair per Nucleotide: Each row/column in the adjacency matrix can have at most one “1.”

</details>


RFold inherently obeys these constraints by viewing base pairs as non-attacking rooks on a chessboard, guaranteeing a valid adjacency matrix M.

⸻

4. Implementation Steps

Below is a recommended approach for integrating RFold as your Stage A module:

4.1 Files & Class Structure
	1.	rna_predict/models/stageA_2d.py
	•	Class: RFold2DPredictor
	•	Responsibility:
	•	Implements the core RFold logic (Seq2map attention, U-Net, row/column factorization).
	•	Exposes predict_2d_structure(sequence) -> np.ndarray that returns a binary adjacency or a probability map for base pairs.
	2.	rna_predict/dataset/dataset_loader.py (optional)
	•	If you unify data loading, you can place code here that handles tokenization or any needed data pre-processing for RFold.

Minimal Example Sketch:

# rna_predict/models/stageA_2d.py
import torch
import torch.nn as nn

class RFold2DPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1) Token embeddings + Seq2map attention
        # 2) U-Net backbone
        # 3) Row/col softmax heads
        # 4) Possibly load pre-trained weights from 'rfold_checkpoint.pt'

    def forward(self, seq_str: str):
        """
        Args:
          seq_str: e.g., 'AUGC...'
        Returns:
          adjacency_matrix: [N, N] with 1/0 indicating paired or unpaired
        """
        # A) Convert seq_str -> token embeddings
        # B) seq2map -> raw pairwise map
        # C) U-Net => produce hidden matrix H
        # D) Symmetrize & mask impossible pairs => xH
        # E) row-softmax, col-softmax => R, C
        # F) combine row/col => final adjacency
        # G) return adjacency
        pass

def build_rfold_predictor(config):
    model = RFold2DPredictor(config)
    # Optional: model.load_state_dict(torch.load("rfold_checkpoint.pt"))
    return model



⸻

4.2 Stage A Integration

Define a simple “extractor” class or function that uses RFold2DPredictor under the hood, so your pipeline can call this Stage A with minimal fuss:

class StageA2DExtractor:
    def __init__(self, config):
        self.model = build_rfold_predictor(config)
    def run(self, sequence: str) -> np.ndarray:
        adjacency = self.model(sequence)  # shape [N, N]
        return adjacency

Later in your main pipeline:

# pipeline.py
def run_pipeline(seq: str):
    # Stage A
    adjacency = stageA2D.run(seq)  # returns [N, N]
    # Stage B
    angles = torsionPredictor.forward(seq, adjacency)
    # Stage C
    coords = forwardKinematics.run(angles)
    ...



⸻

5. Notes on Usage and Configuration
	1.	Pretrained Weights
	•	RFold usually requires training on a known dataset (RNAStralign, bpRNA, etc.). You can either replicate their training or load any existing checkpoint from the RFold authors.
	2.	Symmetry & Masking
	•	Internally, the model will force H -> (H * H^T) * mask, thus ignoring invalid base-type pairs or short loops.
	•	Row- and column-softmax yield R and C, which are combined (e.g., by (R + C)/2) to produce a final adjacency probability.
	3.	Binary Output
	•	If you want a final 0/1 adjacency matrix, apply argmax or threshold (≥ 0.5) to those probabilities.
	•	This threshold ensures valid row and column constraints, effectively placing a single “rook” per row or column if the row-col max is unique.
	4.	Performance Gains
	•	Empirically, RFold is quite fast—on the order of 0.02s/sequence for ~100–200 nt. If you handle very large RNAs (e.g., thousands of nucleotides), watch out for memory usage in the N×N step, but the authors note it scales better than many alternatives.
	5.	Possible Soft Version
	•	If your dataset has rare exceptions to standard constraints, you can adopt a “soft” approach that slightly relaxes the one-pair-per-nucleotide rule. (See the “soft-RFold” discussion in the paper.)

⸻

6. Benefits & Summary

By adopting RFold in Stage A:
	•	Full Constraint Satisfaction: The adjacency is guaranteed to have valid base pairs, no short loops, and at most one pairing per row/column.
	•	Higher Precision: RFold rarely predicts spurious pairs thanks to its row/column factorization.
	•	Efficient: Its row-col decomposition is straightforward, resulting in a speed advantage over iterative or relaxation-based folding approaches.

Hence, RFold is an ideal choice if your pipeline demands fast, accurate, and strictly valid adjacency predictions before downstream torsion-angle and 3D reconstruction steps.

⸻

References
	•	Chen, et al. “RNA Secondary Structure Prediction by Learning Unrolled Algorithms.” ICLR, 2019.
	•	Fu, et al. “UFold: Fast and Accurate RNA Secondary Structure Prediction with Deep Learning.” NAR, 2022.
	•	Tan, et al. “Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective.” arXiv:2212.14041, 2024.
	•	… (See Document Above for Further Citations)

⸻

End of Document