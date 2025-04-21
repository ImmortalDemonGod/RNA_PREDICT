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


**Note on Configuration (Hydra)**

Stage A's behavior is controlled via [Hydra](https://hydra.cc/). The default parameters are set in:
`rna_predict/conf/model/stageA.yaml`

Here's a snippet illustrating the structure and some key default values:

```yaml
# rna_predict/conf/model/stageA.yaml

# --- Top-Level Parameters ---
num_hidden: 128       # Corresponds to query_key_dim in Seq2Map? (Check RFold code)
dropout: 0.3          # General dropout rate
min_seq_length: 80    # Minimum sequence length for padding
device: "cuda"        # Target device ("cuda" or "cpu")
checkpoint_path: "RFold/checkpoints/RNAStralign_trainset_pretrained.pth"
checkpoint_url: "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1"
batch_size: 32        # Batch size (if applicable during inference/training)
lr: 0.001             # Learning rate (for training/fine-tuning)
threshold: 0.5        # Threshold for converting probabilities to binary map

# --- Nested Configurations ---
visualization:
  enabled: true
  varna_jar_path: "tools/varna-3-93.jar" # Default path to VARNA jar
  resolution: 8.0

model:
  # U-Net Architecture
  conv_channels: [64, 128, 256, 512]
  residual: true
  c_hid: 32 # Hidden channels in U-Net bottleneck

  # Seq2Map Component
  seq2map:
    attention_heads: 8
    attention_dropout: 0.1
    # ... other seq2map params ...

  # Decoder Component
  decoder:
    skip_connections: true
    # ... other decoder params ...
```
*(**Note:** Refer to `rna_predict/conf/config_schema.py` for the complete, typed definition of all parameters and their defaults.)*

**Overriding Defaults via Command Line:**

You can easily override any parameter without editing the YAML file using command-line arguments. Since the main `default.yaml` loads `model/stageA`, these parameters are accessed under the `stageA` group:

* **Change Dropout:**
    ```bash
    python rna_predict/pipeline/stageA/run_stageA.py stageA.dropout=0.1
    ```
* **Use CPU:**
    ```bash
    python rna_predict/pipeline/stageA/run_stageA.py stageA.device=cpu
    ```
* **Change Nested Parameter (U-Net hidden channels):**
    ```bash
    python rna_predict/pipeline/stageA/run_stageA.py stageA.model.c_hid=64
    ```
* **Change Deeper Nested Parameter (Seq2Map attention heads):**
    ```bash
    python rna_predict/pipeline/stageA/run_stageA.py stageA.model.seq2map.attention_heads=12
    ```
* **Disable Visualization:**
    ```bash
    python rna_predict/pipeline/stageA/run_stageA.py stageA.visualization.enabled=false
    ```

Combine multiple overrides as needed. This allows for flexible experimentation.
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
