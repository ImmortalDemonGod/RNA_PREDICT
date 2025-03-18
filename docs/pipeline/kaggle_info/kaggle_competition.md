# Comprehensive Overview: Stanford RNA 3D Folding Competition 🧬

---

## 1. Competition Goal 🎯

### High-Level Objective

Accurately predict the 3D coordinates (**x, y, z**) of each nucleotide’s **C1′ atom** in an RNA chain from sequence alone. Competitors must provide **five distinct structural predictions per RNA target**. The primary evaluation metric is the **TM-score** (0 to 1; higher scores indicate superior predictions), widely accepted for RNA/protein structure comparisons.

### Challenges

- **RNA Flexibility:** RNA frequently adopts multiple conformations.
- **Existing Limitations:** Automated RNA prediction lags behind expert-guided manual modeling.
- **Competition Ambition:** Outperform manual expert predictions and advance RNA modeling frontiers.

---

## 2. Detailed Data Overview 📂

### Primary Files

- **train_sequences.csv** (~844 sequences)

	- `target_id`: Unique identifier (e.g., pdbid_chain).

	- `sequence`: RNA nucleotide sequence (A, C, G, U, plus rare alternatives).

	- `temporal_cutoff`: Date of sequence/structure publication; ensures compliance with chronological data-use constraints.

	- `description`: Context about RNA (source, ligands, etc.).

	- `all_sequences`: FASTA-format sequences of all experimental structure chains.

- **train_labels.csv** (Experimental coordinates)

	- `ID`: Combination of target ID and residue number (e.g., 101D_1).

	- `resname`: Nucleotide (A, C, G, U).

	- `resid`: Residue index (1-based).

	- Coordinates (`x_1, y_1, z_1, x_2, y_2, z_2, …`): Multiple conformations or PDB depositions.

- **validation_sequences.csv / validation_labels.csv**

	- ~12 RNA targets from prior CASP15 challenges, intended for local validation.

	- Often considered "burned" after initial leaderboard tuning.

- **test_sequences.csv**
	- Public leaderboard test set, periodically updated; no labels provided.

- **sample_submission.csv**
	- Submission format example:
    ```
    ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
    ```

- **MSA/ folder**
	- Multiple Sequence Alignments (FASTA), valuable for evolutionary conservation signals.

### Additional Resources

- **Synthetic RNA Dataset:** 400,000+ structures from RFdiffusion available for model augmentation.
- **Public PDB Data:** Allowed with strict adherence to temporal cutoff rules.
- **External Advanced Models:** Permitted if temporally compliant and Kaggle-offline.

---

## 3. Data Utilization and Structure 🔍

### Sequence-to-3D Coordinate Mapping

- Models take RNA sequences/MSAs and output coordinates of the C1′ atom.
- Training set teaches sequence-structure relationships; validation refines performance.
- Kaggle evaluates predictions against hidden labels using automated TM-score calculation.

### Handling Multiple Conformations

- Datasets may include multiple conformations per RNA; Kaggle selects best TM-score alignment automatically.

### Five Predictions Requirement

- Generate five structural predictions per residue, either from single models (multiple seeds) or distinct models.
- Kaggle scoring uses only the best of these five predictions per RNA target.

---

## 4. Scoring: TM-score 📏

### Calculation Formula

```
TM-score = max(1/L_ref × Σ[1/(1+(d_i/d_0)²)])
```

- `L_ref`: Number of residues in the reference structure.
- `d_i`: Distance between aligned residues (C1′ atoms).
- Automated sequence-independent alignment via **US-align**.

### Alignment Specifics

- Automated, sequence-independent alignment ensures optimal structural comparison.
- Final TM-score averaged over all test targets, using your best-of-five predictions.

---

## 5. Competition Timeline and Phases 📅

| Phase                           | Date                     | Description                                                  |
|---------------------------------|--------------------------|-------------------------------------------------------------|
| **Start Date**                  | February 27, 2025        | Release of training data and initial test sequences         |
| **Leaderboard Refresh**         | April 23, 2025           | New sequences added; some test sequences moved to training; leaderboard reset; early-sharing prizes awarded |
| **Final Submission Deadline**   | May 29, 2025             | Final submissions for private leaderboard ranking           |
| **Future Data Phase**           | June – September 2025    | Evaluate generalization on up to 40 new RNA structures      |

---

## 6. Frequently Asked Questions & Insights ❓

- **Training Set Size:** Smaller to focus on direct, experimentally validated 3D structures rather than indirect data.
- **Multiple Predictions:** Required due to RNA’s propensity for multiple valid conformations.
- **Use of External Tools:** Permitted with offline capabilities, compliance with temporal cutoffs.
- **Temporal Cutoff Purpose:** Prevents "future data leakage"; structural information post-cutoff date disallowed.
- **Multiple Reference Structures:** Kaggle selects best conformational alignment automatically.
- **Future Data Evaluation:** Validates methods' genuine generalization capabilities.

---

## 7. Practical Steps for Effective Modeling 🚧

### Data Preparation

- Handle duplicates and multiple entries.
- Decide treatment for multiple conformations (single vs. multi-target).
- Optionally augment data with synthetic RNA or public structures.

### Model Strategies

- **Neural Networks:** Graph neural networks, equivariant networks, diffusion models.
- **Language Models:** Fine-tune large language models with structural prediction heads.
- **Hybrid Approaches:** Combine existing predictors (RiboNanzaNet, RhoFold) with energy-based refinement.
- **Manual/Heuristic Methods:** Utilize known RNA motifs (A-minor motifs, base-pair geometry constraints).

### Submission Guidelines

- Provide five distinct predictions per residue.
- Follow exact submission format provided by Kaggle.
- Observe Kaggle's runtime constraints (≤ 8 hours).

### Validation & Optimization

- Employ cross-validation, particularly across temporal splits.
- Optionally replicate Kaggle’s TM-score calculation locally using US-align.

---

## 8. Key Takeaways 🗝️

- **Primary Goal:** Predict RNA 3D coordinates (C1′ atoms).
- **Provided Data:** Moderately sized dataset (~844 structures), CASP15 validation set, evolving test sequences.
- **Five-Prediction Requirement:** Encourages robust exploration of RNA structural variability.
- **Scoring Metric:** TM-score (US-align automated alignment).
- **Timeline:** Frequent data and leaderboard updates; final evaluation includes unseen future structures.
- **Opportunities and Challenges:**
	- Complex RNA folding dynamics.
	- Advanced modeling techniques encouraged (deep learning, diffusion).
	- Strict adherence to data-use temporal constraints critical.
	- Collaborative environment with incentives for transparency and early result sharing.

---