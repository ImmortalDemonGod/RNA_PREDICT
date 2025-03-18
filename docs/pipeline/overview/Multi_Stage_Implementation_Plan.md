# 📐 Multi-Stage RNA 3D Prediction Pipeline

This consolidated technical plan merges the extensive details from Version 1 with the visually structured and accessible layout of Version 2. It provides comprehensive explanations, practical implementation guidance, and actionable debugging strategies.

---

## 📌 Overview of the Multi-Stage Architecture

The RNA 3D prediction pipeline consists of four clearly defined stages (**A, B, C**, and optionally **D**). Each stage is modular, independently testable, and replaceable:

### 🧬 Stage A: 2D Predictor
- **Goal:** Predict RNA secondary structure (base-pair adjacency, contact maps).
- **Input:** Raw RNA sequence (`N` nucleotides).
- **Output:** Adjacency matrix (`adjacency ∈ ℝ^(N×N)` or multi-channel features, `ℝ^(N×N×c₂ᴅ)`).

### 📐 Stage B: Torsion-Angle Predictor
- **Goal:** Predict nucleotide backbone torsion angles using adjacency and sequence.
- **Input:** `adjacency` from Stage A and RNA sequence or embeddings.
- **Output:** Torsion angles per nucleotide (`α, β, γ, δ, ε, ζ, χ`) in array form (`ℝ^(N×n_angles)`).

### 🔧 Stage C: Forward Kinematics (3D Reconstruction)
- **Goal:** Generate precise 3D atom coordinates from torsion angles.
- **Input:** Torsions from Stage B and standard RNA bond geometry.
- **Output:** Cartesian coordinates (`ℝ^(N_atoms×3)`).

### 🌟 Optional Stage D: AF3-Inspired Refinement
- **Goal:** Enhance prediction accuracy using Pairformer/diffusion methods.
- **Input:** Partial 3D coordinates, torsion angles, adjacency, or embeddings.
- **Output:** Refined angles/coordinates, optional confidence metrics (e.g., pLDDT, PDE).

> 📌 **Optional expansions:** Consider additional modules (MSA, confidence heads, template embeddings) after the basic pipeline is operational.

---

## 🛠️ Detailed Stage-by-Stage Design

### 🔬 Stage A: 2D Structure/Adjacency
- **Inputs:** Nucleotide sequence (e.g., "AUGC…").
- **Processing Options:**
	- External prediction tools (e.g., ViennaRNA).
	- Minimal neural models (LSTM/Transformer) for base-pair probability prediction.
- **Outputs:** Contact map (`[N, N]`), optionally multi-channel with probabilities and entropies (`[N, N, c₂ᴅ]`).
- **Implementation Notes:**
	- Files: `rna_predict/dataset/dataset_loader.py` or `rna_predict/models/stageA_2d.py`.
	- Recommended class wrapper: `StageA2DExtractor`.

### 📏 Stage B: Torsion-Angle Predictor
- **Inputs:** Adjacency from Stage A, RNA sequence, or residue embeddings.
- **Model Architecture:** GNN or Transformer (recommended); simpler MLP if necessary (less optimal for large N).
- **Outputs:** Torsion angles (`ℝ^(N×n_angles)`), `[α, β, γ, δ, ε, ζ, χ]`.
- **Implementation Notes:**
  - File: `rna_predict/models/encoder/torsion_predictor.py`.
  - Debugging: initially use trivial or "A-form average" predictor.
  - Ensure consistent indexing across stages.

### 🛠️ Stage C: Forward Kinematics (3D Build)
- **Inputs:** Torsion angles from Stage B, standard RNA geometry.
- **Core Logic:**
  - Place the first residue in reference orientation.
  - Iteratively apply torsions, rotating local coordinate frames.
  - Handle sugar puckers (initial backbone-only recommended).
- **Outputs:** Atom coordinates (`ℝ^(N_atoms×3)`).
- **Implementation Notes:**
  - File: `rna_predict/scripts/forward_kinematics.py`.
  - Start simply by placing phosphate-sugar backbone atoms only.

---

### 🌟 Optional Stage D: Pairformer & Diffusion

#### 🔹 Pairformer (AF3-Trunk)
- **Purpose:** Global residue-pair context.
- **Modules:** TriangleMultiplication, TriangleAttention (AF3).
- **Implementation:** `rna_predict/models/trunk/pairformer_stack.py`.

#### 🌬️ Diffusion Refinement
- **Purpose:** Iteratively denoise angles/coordinates.
- **Algorithm:** Angle-based, iterative denoising referencing Pairformer embeddings.
- **Implementation:** `rna_predict/models/diffusion/angle_diffusion.py`.
  - Start single-step; expand as time permits.

#### 📈 Confidence Heads (Optional)
- **Purpose:** Prediction confidence estimation (pLDDT, PAE, PDE).
- **Implementation:** Classifiers post-Pairformer trunk.

---

## 📅 Recommended Development Phases

### 🚩 Phase 1: Minimal Pipeline (Days 1–7)
- **Days 1–3:** Data parsing, adjacency pipeline.
- **Days 3–5:** Basic torsion predictor.
- **Days 5–7:** Forward kinematics.
- **Result:** Functional pipeline by Day 7.

### 🚩 Phase 2: AF3 Enhancements (Days 8–20)
- Days 8–14: Pairformer trunk implementation.
- Days 14–20: Angle-based diffusion refinement.
- Optional: Confidence/MSA modules as resources allow.

### 🚩 Phase 3: Integration & Submission
- Multi-seed predictions per sequence.
- Geometry refinement with external tools (e.g., PyRosetta).
- Memory optimization.
- Confidence-based ranking.

---

## 📂 Implementation Notes & File Layout

```
rna_predict/
  dataset/
    dataset_loader.py        # Stage A (sequence + adjacency)
  models/
    encoder/
      torsion_predictor.py   # Stage B (torsion angles)
    trunk/
      pairformer_stack.py    # Optional Pairformer
    diffusion/
      angle_diffusion.py     # Stage D diffusion
  scripts/
    forward_kinematics.py    # Stage C (3D coordinates)
```

- Maintain modular code with clear input/output interfaces.
- Integrate optional MSA modules clearly.

---

## ⚠️ Key Risks & Mitigations
- **Residue Indexing:** Centralize and consistently enforce indexing.
- **Overfitting:** Cross-validation, partial datasets, A-form priors.
- **Embedding Complexity:** Optimize memory via chunking.
- **Time Constraints:** Prioritize minimal system functionality.

---

## 🎯 Conclusion
- **Core Pipeline:** Quickly establish Stages A→C.
- **Optional Enhancements:** Stage D refinements, MSA, confidence heads.
- **Recommended Timeline:** Balances quality and deadlines.
- **Modular Structure:** Enables incremental development and testing.

