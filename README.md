# 📌 rna_predict

[![codecov](https://codecov.io/gh/ImmortalDemonGod/RNA_PREDICT/branch/main/graph/badge.svg?token=dY9KCmU95B)](https://codecov.io/gh/ImmortalDemonGod/RNA_PREDICT) [![CI](https://github.com/ImmortalDemonGod/RNA_PREDICT/actions/workflows/main.yml/badge.svg)](https://github.com/ImmortalDemonGod/RNA_PREDICT/actions/workflows/main.yml) [![CodeScene general](https://codescene.io/images/analyzed-by-codescene-badge.svg)](https://codescene.io/projects/65039) [![CodeScene Average Code Health](https://codescene.io/projects/65039/status-badges/average-code-health)](https://codescene.io/projects/65039) [![CodeScene Hotspot Code Health](https://codescene.io/projects/65039/status-badges/hotspot-code-health)](https://codescene.io/projects/65039)


**Awesome `rna_predict` created by ImmortalDemonGod** 🚀

---

### 🚀 RNA_PREDICT Inference Pipeline: Running Predictions

#### 🔥 Recommended Command (with Explanation)

```bash
uv run rna_predict/predict.py \
  input_csv=rna_predict/dataset/examples/kaggle_minimal_index.csv \
  checkpoint_path=outputs/2025-04-28/16-07-58/outputs/checkpoints/last.ckpt \
  output_dir=outputs/predict_M2_test/ \
  fast_dev_run=true \
  > dev_run_output.txt 2>&1
```

- **uv run rna_predict/predict.py**: Runs the main prediction pipeline using the project's preferred Python runner (never use `python` directly).
- **input_csv=...**: Path to the CSV file listing RNA sequences for prediction.
- **checkpoint_path=...**: Path to the model checkpoint to use for inference.
- **output_dir=...**: Where to save all prediction outputs (CSV, PDB, and .pt files).
- **fast_dev_run=true**: (Optional) Runs a single sequence for quick debugging.
- **> dev_run_output.txt 2>&1**: Redirects all output (including errors) to a log file for later inspection.

#### 🗂️ Output Files
- **prediction_{i}.csv**: Atom-level coordinates (atom name, residue index, x, y, z) for each prediction.
- **prediction_{i}.pdb**: Standard PDB file for molecular visualization.
- **prediction_{i}.pt**: (Optional) PyTorch dictionary for internal use.
- **summary.csv**: Atom counts and summary for all predictions.

#### 💡 Why these formats?
- **CSV**: Easy to inspect, analyze, or import into data tools.
- **PDB**: Standard for 3D structure visualization (PyMOL, Chimera, VMD, etc).
- **.pt**: For advanced PyTorch workflows/debugging.

#### 📝 Notes
- Always use `uv run` for all project scripts for correct environment handling.
- You can customize `input_csv`, `checkpoint_path`, and `output_dir` as needed.
- For full pipeline or batch prediction, set `fast_dev_run=false` (or omit).
- All outputs are saved in the specified `output_dir`.

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
python runners/demo_entry.py
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
