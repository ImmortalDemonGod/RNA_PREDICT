# 📌 rna_predict

[![codecov](https://codecov.io/gh/ImmortalDemonGod/RNA_PREDICT/graph/badge.svg?token=dY9KCmU95B)](https://codecov.io/gh/ImmortalDemonGod/RNA_PREDICT)

[![CI](https://github.com/ImmortalDemonGod/RNA_PREDICT/actions/workflows/main.yml/badge.svg)](https://github.com/ImmortalDemonGod/RNA_PREDICT/actions/workflows/main.yml)

[![CodeScene general](https://codescene.io/images/analyzed-by-codescene-badge.svg)](https://codescene.io/projects/65039)

[![CodeScene Average Code Health](https://codescene.io/projects/65039/status-badges/average-code-health)](https://codescene.io/projects/65039)

[![CodeScene Hotspot Code Health](https://codescene.io/projects/65039/status-badges/hotspot-code-health)](https://codescene.io/projects/65039)


**Awesome `rna_predict` created by ImmortalDemonGod** 🚀

---

### 📦 Installation

Install from PyPI:

```bash
pip install rna_predict
```

---

### 🚩 Usage

**Basic Python Usage:**

```python
from rna_predict import BaseClass
from rna_predict import base_function

BaseClass().base_method()
base_function()
```

**Command Line Interface:**

```bash
python -m rna_predict
# or simply
rna_predict
```

---

### 🛠️ Development

Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

---

### 📚 RNA 3D Structure Prediction

#### 🎯 Introduction & Motivation

RNA molecules fold into intricate 3D structures critically determining their biological functions. This repository provides a pipeline:

- **Stage A:** RNA 2D adjacency (secondary structure).
- **Stage B:** Neural torsion-angle prediction.
- **Stage C:** Forward kinematics from angles to 3D coordinates, plus optional energy-based refinement.

Advanced modules include diffusion-based refinement (AlphaFold 3 inspired), isosteric base substitutions, and potential HPC integration (Kaggle competitions).

---

### 🧩 Pipeline Overview

The pipeline, inspired by AlphaFold but specialized for RNA, includes:

#### Stage A: RNA 2D Structure (Adjacency)
- Predict or import base-pair matrix (e.g., RFold, ViennaRNA, RNAfold).
- Detailed documentation: `StageA_RFold.md`, `RFold_paper.md`.

#### Stage B: Torsion Angle Prediction
- Neural approaches: `AtomAttentionEncoder` (`atom_encoder.py`, `atom_transformer.py`, `block_sparse.py`) or `RNA-TorsionBERT` (`torsionBert_full_paper.md`, `torsionBert.md`).
- Predicts torsion angles \(\alpha,\beta,\gamma,\delta,\epsilon,\zeta,\chi\).
- Benchmarks (`rna_predict/benchmarks/benchmark.py`) GPU latency and memory use.

#### Stage C: Forward Kinematics & 3D Reconstruction
- Stepwise conversion of torsion angles into 3D coordinates (pseudo-algorithm provided in `Stage_C.md`, `Stage_C_Refinement_Plan.md`).
- Optional energy minimization or molecular dynamics (MD) via GROMACS, AMBER, OpenMM.

#### Stage D (Advanced):
- Diffusion-based refinement (`s4_diffusion.md`, `AlphaFold3_progress.md`).
- Isosteric base-substitution logic for redesign (`RNA_isostericity.md`).

---

### 🌀 Detailed Pipeline Stages

#### Stage A: RNA 2D (Adjacency)
- Uses RFold or external tools. No single "StageA" Python file; adjacency computed externally.
- Outputs: `[N × N]` adjacency or partial contact probabilities.

#### Stage B: Torsion-Angle Prediction
- Approaches: AtomAttentionEncoder (local adjacency), RNA-TorsionBERT (sequence-only).
- Outputs: `[N_res, 7]` angles or `[N_res, 2×7]` sin/cos representations.

#### Stage C: 3D Reconstruction & Refinement
- Forward kinematics pseudo-algorithm detailed clearly.
- Optional MD refinements recommended (short minimization steps).

---

### 🔬 Advanced Methods

#### Diffusion-Based Refinement
- Iterative denoising, inspired by AlphaFold 3 (`AngleDiffusionModule`).

#### Isosteric Base Substitutions
- Sequence redesign preserving geometry, detailed logic provided.

#### Kaggle Integration
- Competitive scenarios explained (`kaggle_competition.md`).

---

### 🗂️ Code Organization

#### Key Directories:
- `rna_predict/models/attention/`
- `rna_predict/models/encoder/`
- `rna_predict/scripts/`
- `rna_predict/benchmarks/`

#### Running Demos:

```bash
cd rna_predict
python main.py
```

---

### 📐 Performance Considerations

- **Local Block-Sparse Optimization:** Significantly reduces GPU memory/time complexity.
- Benchmark specifics provided for large RNA handling.
- Explicit recommendation for chunking or dimension reduction for very large RNA sequences.

---

### 📖 Theoretical Highlight: Forward Kinematics

- Clearly defined pseudo-algorithmic logic.
- Sugar pucker handling: Standard (`C3′-endo`) or predicted angles.

```pseudo
coords[0] = place_first_residue(torsion_angles[0])
for i in range(1, N_res):
    anchor = coords[i-1]
    angles = torsion_angles[i]
    coords[i] = build_next_residue(anchor, angles, standard_geom)
```

---

### 🚀 Recommendations for Future Development

- Implement full pipeline (`rna_predict/pipeline.py`) combining stages explicitly.
- Create explicit `forward_kinematics.py`.
- Add small MLP torsion-head (`torsion_head.py`).
- Partial diffusion refinement module and confidence metrics.

---

🌟 **Conclusion:**

Structured, MkDocs-friendly documentation, explicitly detailed with filenames, pipeline stages, algorithmic insights, and clear performance guidelines to enhance readability and comprehensive understanding.

