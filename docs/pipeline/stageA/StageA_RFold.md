## 🚀 Technical Documentation: Integrating **RFold** into a Multi-Stage RNA 3D Pipeline

### 🎯 Focusing on Stage A (2D Structure/Adjacency)

---

### 📌 1. Overview

This document outlines how to incorporate **RFold**—a K-Rook-based RNA secondary structure predictor—into **Stage A** of a multi-stage RNA 3D pipeline. Stage A is responsible for predicting base pairs (2D adjacency or contact maps) from an RNA sequence. Subsequent stages (torsion angle prediction and 3D reconstruction) use these adjacency maps as inputs. By integrating **RFold** into Stage A, valid secondary structures are guaranteed, and the overall workflow is streamlined.

---

### ✅ 2. Why Use RFold for Stage A?

- **Validity Guarantee** 🔒
  - RFold formulates RNA folding as a symmetric K-Rook matching problem, ensuring nucleotides pair at most once and adhere strictly to base-type/distance constraints.
  - Other neural methods often require iterative relaxations or additional steps to approximate constraints.

- **High Accuracy** 🎯
  - RFold demonstrates superior F1 scores across RNA datasets (**RNAStralign**, **ArchiveII**, **bpRNA**), notably excelling in precision.

- **Fast Inference** ⚡
  - Runs approximately **0.02 s per sequence** (on GPU), ideal for rapidly processing large RNA datasets.

- **Straightforward Integration** 🔗
  - Produces an **N×N adjacency (or probability) matrix** easily fed into subsequent stages (torsion prediction and 3D reconstruction).

---

### 🛠️ 3. Stage A: 2D Structure / Adjacency

In Stage A, an RNA sequence of length **N** is used to generate a contact map (**adjacency matrix** `M[i,j] = 1` indicates paired nucleotides).

<details>
<summary>📐 Constraints Typically Enforced</summary>

- **Base-type**: Only permitted pairs are **A–U**, **G–C**, or **G–U**.
- **Minimum Loop Length**: No pairs allowed between indices `i, j` if `|i−j| < 4`.
- **One Pair per Nucleotide**: Each row/column in the adjacency matrix can have at most one "1."

</details>

**RFold** inherently adheres to these constraints, representing base pairs as non-attacking rooks on a chessboard, thus guaranteeing a valid adjacency matrix.

---

### 🔧 4. Implementation Steps

#### 📂 4.1 Files & Class Structure

- **`rna_predict/models/stageA_2d.py`**
  - Class: `RFold2DPredictor`
  - Responsibilities:
    - Implements RFold logic (**Seq2map attention**, **U-Net backbone**, **row/column softmax factorization**).
    - Method: `predict_2d_structure(sequence)` → returns adjacency/probability map.

- **`rna_predict/dataset/dataset_loader.py`** *(optional)*
  - Handles unified data loading, tokenization, and necessary preprocessing.

#### 💻 Minimal Example Sketch

```python
# rna_predict/models/stageA_2d.py
import torch
import torch.nn as nn

class RFold2DPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1) Token embeddings + Seq2map attention
        # 2) U-Net backbone
        # 3) Row/col softmax heads
        # 4) Load pre-trained weights ('rfold_checkpoint.pt')

    def forward(self, seq_str: str):
        """
        Args:
          seq_str: e.g., 'AUGC...'
        Returns:
          adjacency_matrix: [N, N] binary adjacency
        """
        pass

def build_rfold_predictor(config):
    model = RFold2DPredictor(config)
    # model.load_state_dict(torch.load("rfold_checkpoint.pt"))
    return model
```

---

#### 🧩 4.2 Stage A Integration

Define an extractor class for seamless integration:

```python
class StageA2DExtractor:
    def __init__(self, config):
        self.model = build_rfold_predictor(config)

    def run(self, sequence: str) -> np.ndarray:
        adjacency = self.model(sequence)  # [N, N]
        return adjacency
```

Pipeline usage example:

```python
# pipeline.py
def run_pipeline(seq: str):
    # Stage A
    adjacency = stageA2D.run(seq)  # [N, N]
    # Stage B
    angles = torsionPredictor.forward(seq, adjacency)
    # Stage C
    coords = forwardKinematics.run(angles)
```

---

### ⚙️ 5. Notes on Usage & Configuration

- **Pretrained Weights** 📦
    - RFold typically requires training on datasets (**RNAStralign**, **bpRNA**). Checkpoints from the RFold authors can be utilized.

- **Symmetry & Masking** 🎭
    - Internally employs `(H * H^T) * mask` to remove invalid pairs.
    - Combines row/column-softmax outputs to form final adjacency probabilities.

- **Binary Output** 🔘
    - Apply thresholding (≥0.5) or argmax on probabilities for binary adjacency matrices, maintaining valid row-column constraints.

- **Performance Gains** 🚅
    - Rapid inference (~0.02 s per sequence, for RNA lengths 100–200 nt).
    - Better scalability than iterative/relaxation-based methods.
    - For extremely long RNAs (1000+ nucleotides), monitor memory usage in the N×N computation step.

- **Soft Version** 🌥️
    - Slight constraint relaxation for datasets with rare exceptions (see the “soft-RFold” discussion in original publications).

---

### 🎉 6. Benefits & Summary

Adopting RFold in Stage A provides:

- **Constraint Satisfaction** ✅
    - Guaranteed valid base pairs, proper loop lengths, and strict nucleotide pairing constraints.

- **Precision** 📌
    - Minimal spurious pair predictions, leveraging row-column factorization.

- **Efficiency** ⏱️
    - Substantial speed advantage, ideal for high-throughput RNA folding applications.

Thus, **RFold** is optimal for pipelines requiring fast, accurate, and constraint-valid adjacency predictions prior to downstream torsion-angle estimation and 3D reconstruction.

---

### 📚 References

- Chen, et al. *“RNA Secondary Structure Prediction by Learning Unrolled Algorithms.”* **ICLR**, 2019.
- Fu, et al. *“UFold: Fast and Accurate RNA Secondary Structure Prediction with Deep Learning.”* **NAR**, 2022.
- Tan, et al. *“Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective.”* **arXiv**, 2024.

---
