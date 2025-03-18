# rna_predict

[![codecov](https://codecov.io/gh/ImmortalDemonGod/RNA_PREDICT/branch/main/graph/badge.svg?token=RNA_PREDICT_token_here)](https://codecov.io/gh/ImmortalDemonGod/RNA_PREDICT)
[![CI](https://github.com/ImmortalDemonGod/RNA_PREDICT/actions/workflows/main.yml/badge.svg)](https://github.com/ImmortalDemonGod/RNA_PREDICT/actions/workflows/main.yml)

Awesome rna_predict created by ImmortalDemonGod

## Install it from PyPI

```bash
pip install rna_predict
```

## Usage

```py
from rna_predict import BaseClass
from rna_predict import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m rna_predict
#or
$ rna_predict
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Below is the fully updated, merged write-up that incorporates the best features from V1–V4 while addressing their criticisms. It is verbose enough for thorough technical documentation, covering all pipeline stages (A–C), advanced methods (diffusion, isosteric substitutions), direct references to code in rna_predict/, and moderate theoretical detail for forward kinematics. In addition, it clarifies performance and usage aspects—combining the synergy of V3 with some theoretical expansions from V4 but without overshadowing the pipeline approach.

⸻

RNA 3D Structure Prediction: A Unified Comprehensive Document

1. Introduction & Motivation

RNA molecules fold into intricate 3D structures that critically determine their biological functions. Accurately predicting these structures from sequence remains a grand challenge. Our repository tackles it with a multi-stage pipeline combining:
	•	Stage A: RNA 2D adjacency (secondary structure).
	•	Stage B: Neural torsion-angle prediction.
	•	Stage C: Forward kinematics from angles to 3D coordinates, plus optional energy-based refinement.

Beyond these foundational stages, advanced modules explore diffusion-based denoising (inspired by AlphaFold 3), isosteric base substitutions for redesign, and potential HPC integration for large-scale challenges (like Kaggle competitions). This document merges all references from our docs (docs/) and the code (rna_predict/) into one place, so new contributors or advanced developers can see the entire pipeline in detail—the how, the why, and the code references.

⸻

2. Pipeline Overview

We adopt a pipeline reminiscent of AlphaFold but specialized for RNA:
	1.	Stage A: 2D structure (Adjacency)
	•	Predict or import the base-pair matrix (which nucleotides pair) — or partial contact constraints.
	•	Tools like RFold can yield a valid adjacency for any input RNA sequence.
	2.	Stage B: Torsion Angle Prediction
	•	A neural approach to produce backbone torsions \{\alpha,\beta,\gamma,\delta,\epsilon,\zeta,\chi\} (and possibly sugar pucker angles).
	•	Implemented either via AtomAttentionEncoder + local attention or a TorsionBERT that uses BERT-like embeddings from the raw sequence alone.
	3.	Stage C: Forward Kinematics & 3D
	•	Convert torsion angles to 3D coordinates by applying standard bond lengths/angles + rotating around each bond axis for each torsion.
	•	Optionally run a short local energy minimization or MD to fix small geometry errors.
	•	Output a final 3D model in, say, PDB format.

Advanced:
	•	A final “Stage D” might do diffusion-based refinement (AlphaFold 3–style) or local annealing to refine coordinates further.
	•	The code references isosteric base-substitution logic for rational redesign while preserving local geometry.

In the docs:
	•	We have a separate .md for each stage (StageA_RFold.md, Stage_B, Stage_C, etc.) plus advanced_torsionBert docs, diffusion notes, isosteric_substitutions guidelines, Kaggle usage.
	•	The present doc merges them for a cohesive reference.

⸻

3. Stage A: RNA 2D (Adjacency)

3.1 Purpose & Implementation

We want a base-pair adjacency matrix [N \times N] that indicates which nucleotides are predicted to pair. This adjacency can come from:
	•	RFold: The doc in docs/pipeline/stageA/StageA_RFold.md explains how “RFold” uses a K-rook matching perspective to ensure each position pairs at most once, respecting base-type constraints (A–U, G–C, G–U) and minimum loop length.
	•	External Tools: Alternatively, you could run ViennaRNA, RNAfold, or else. The code only requires adjacency as an input if you want adjacency-based angle prediction.

In rna_predict/:
	•	We do not see a single “StageA” Python file. Instead, the adjacency is typically stored or computed externally.
	•	The user can store adjacency in a [N,N] Tensor or partial adjacency for noncanonical pairs.

3.2 Usage

One might do:

# Suppose you have an adjacency from an external fold tool
adj_matrix = run_rfold(sequence)  # or from a dot-bracket
# Then feed it into Stage B if needed

Docs:
	•	docs/pipeline/stageA/StageA_RFold.md details the usage of RFold2DPredictor.
	•	RFold_paper.md in the same folder for theoretical underpinnings.

Outputs:
	•	[N \times N] adjacency or a partial contact probability array for each sequence.
	•	If you skip adjacency, you can rely on TorsionBERT (no adjacency) or partial adjacency from 3D if known.

⸻

4. Stage B: Torsion-Angle Prediction

4.1 Approaches

We provide two main strategies:
	1.	AtomAttentionEncoder (Local adjacency-based approach)
	•	Found in rna_predict/models/encoder/atom_encoder.py.
	•	Processes per-atom features: (pos, charge, element, etc.) plus pairwise embeddings from adjacency or partial 3D distances.
	•	Feeds them into AtomTransformer (in atom_transformer.py), a local multi-head self-attention with optional block-sparse optimization (block_sparse.py).
	•	Aggregates atoms → tokens via scatter_mean (see scatter_utils.py).
	•	Typically we get a [N_res, c_token] embedding. We can add a small MLP to map each token to 7 torsion angles if we want. The doc in Stage_B.md references how this might be done.
	2.	RNA-TorsionBERT (Sequence-based approach)
	•	Discussed in docs/pipeline/stageB/torsionBert_full_paper.md and torsionBert.md.
	•	Fine-tunes a BERT-like model (like DNABERT) on RNA sequences to predict angles (\alpha,\dots,\zeta,\chi).
	•	Ignores adjacency—purely sequence → angle.
	•	Has an optional scoring function TB-MCQ to evaluate angle-based geometry accuracy.

Either method yields torsion angles \{\alpha,\beta,\gamma,\delta,\epsilon,\zeta,\chi\} per nucleotide plus possible sugar pucker angle or pseudorotation if the doc addresses it.

4.2 Code Details & Local Attention

Local block-sparse:
	•	In block_sparse.py, we have LocalBlockSparseAttentionNaive or an optimized BlockSparseAttentionOptimized.
	•	The user can set use_optimized=True in AtomTransformerBlock to accelerate large system training.
	•	The doc in docs/advanced_methods/diffusion/s4_diffusion.md references “liquid-s4” expansions but that’s more advanced.

4.3 Performance & Benchmarks

rna_predict/benchmarks/benchmark.py:
	•	benchmark_input_embedding(): times forward/backward on random synthetic data for various [N_atom, N_token].
	•	benchmark_decoding_latency_and_memory(): logs GPU memory usage on bigger inputs.

In docs:
	•	We mention local vs. global attention trade-offs, HPC usage, etc.

4.4 Output

Angles:

A typical shape: [N_res, 7] if 7 angles/res. The user can store them or pass them to Stage C. If the code uses TorsionBERT, you might see a direct [N_res, 2 x 7] for sin/cos representations.

⸻

5. Stage C: 3D Reconstruction & Refinement

5.1 Forward Kinematics

Motivation:
	•	In RNA, the main degrees of freedom are torsion angles. We can treat each residue as a link in a chain, with known bond lengths and angles but variable torsions.
	•	Forward kinematics (FK) means we apply each torsion angle as a rotation around its bond, building the chain from 5′ to 3′.

Docs:
	•	docs/pipeline/stageC/Stage_C.md and Stage_C_Refinement_Plan.md explain:
	•	Stepwise approach: place first residue in a reference orientation, then for each subsequent residue i, we place P(i), O5’(i), sugar ring, etc. by rotating around the previous bonds.
	•	We might fix standard lengths from a reference geometry library (like AMBER or standard RNA tables).
	•	If sugar pucker is predicted, incorporate ring closure or do a partial fallback to a standard C3′-endo.

Implementation:

Currently, no single “forward_kinematics.py” in rna_predict/. Instead, we have:
	•	Pseudocode in docs:
	•	We’d define coords[0] for the first residue, then for i in [1..N], apply a local transform = rotation(torsion_i) + translations from bond lengths.
	•	Possibly do ring closure for the sugar.
	•	The user can do a final check or short local MD to fix small ring tension or base stacking.

5.2 Local Refinement (Energy Minimization or MD)

Why: Torsion predictions might be approximate, or the sugar ring might be left slightly open. A short minimization can fix bond strains or steric clashes.

Doc:
	•	Stage_C_Refinement_Plan.md enumerates the usage of GROMACS or AMBER:
	1.	Minimize geometry (a few thousand steps).
	2.	If still needed, do a short 100 ps MD at low temperature or with weak restraints.

In the code:
	•	We have no direct calls to GROMACS or Amber. The doc suggests a shell approach or a partial Python integration with OpenMM.
	•	Could produce a final PDB that is physically realistic.

⸻

6. Advanced Methods & Modules

6.1 Diffusion-Based Refinement

Concept:
	•	Inspired by AlphaFold 3: after you get an initial coordinate from Stage C, a “diffuser” module can do iterative denoising (like “angle-based diffusion” or “coordinate-based”).
	•	The doc in docs/advanced_methods/diffusion/s4_diffusion.md and AlphaFold3_progress.md explains a potential approach that sees geometry as a 1D sequence of atoms, applying an S4-like approach for large T steps.

Implementation:
	•	Not fully coded in rna_predict/. We have partial references to “AngleDiffusionModule” but it’s mostly conceptual.

Value:
	•	Could unify adjacency from Stage A, trunk embeddings from Stage B, and do repeated small updates to refine geometry further.
	•	Potential for HPC usage if you want 10k diffusion steps.

6.2 Isosteric Base Substitutions

Goal:
	•	Redesign an existing 3D structure’s sequence while preserving the geometry. E.g., mutate A–U → G–U or G–C if they remain isosteric.

Docs:
	•	docs/advanced_methods/isosteric_substitutions/RNA_isostericity.md details:
	•	Identify base pairs in the 3D structure, classify them (Leontis–Westhof families).
	•	Gather possible isosteric or near-isosteric replacements from known tables.
	•	Filter by bridging water or environment constraints.
	•	Use a constraint solver or backtracking to produce mutated sequences.

Implementation:
	•	Also partially doc-based, not fully integrated. The approach references a “pairwise constraint approach” but doesn’t exist as a final .py.

6.3 Kaggle Competition Integration

Doc:
	•	docs/pipeline/kaggle_info/kaggle_competition.md explains typical scoring with TM-score, the multi-output approach (5 structures / target), and timeline.
	•	The pipeline can produce multiple random seeds or small variations in angles (Stage B) or final coordinates (Stage C) to get 5 candidate solutions.

⸻

7. Code Organization & Usage

7.1 Key Directories
	1.	rna_predict/models/attention/
	•	block_sparse.py, atom_transformer.py (local attention block).
	2.	rna_predict/models/encoder/
	•	atom_encoder.py (AtomAttentionEncoder), input_feature_embedding.py (merges token-level features).
	3.	rna_predict/scripts/
	•	Torsion angle calculations (mdanalysis), etc.
	4.	rna_predict/benchmarks/benchmark.py
	•	Times the embedding forward/back pass, memory usage, local vs. naive attention.
	5.	rna_predict/main.py
	•	Demonstrations:
	•	demo_run_input_embedding() → synthetic data → InputFeatureEmbedder → prints shape.
	•	demo_stream_bprna(), show_full_bprna_structure(), demo_compute_torsions_for_bprna().

7.2 Running Basic Demos

Example:

cd rna_predict
python main.py

You’ll see logs about shapes, example data from the bprna-spot dataset, or partial torsion calculations.
For performance, run python benchmarks/benchmark.py.

7.3 Missing Full E2E Pipeline

While each stage is well-documented, a single “rna_predict/pipeline.py” that calls Stage A → B → C in a single run is not present. You must manually handle adjacency creation, angle inference, then forward kinematics code (found as doc-based pseudocode). Similarly, local MD refinement is an external step.

⸻

8. Performance Considerations
	1.	Local Block-Sparse
	•	Drastically reduces memory/time from \mathcal{O}(N^2) to \mathcal{O}(N \times \text{window-size}).
	•	benchmark.py logs forward/back times.
	2.	Potential GPU Gains
	•	If block_sparse_attn is installed, you can do the optimized path. If not, fallback is naive.
	3.	Large RNAs
	•	Could still push GPU memory. Users might chunk the sequence or reduce hidden dims.
	4.	Refinement
	•	Short minimizations typically run fast (seconds to minutes).
	•	Diffusion with large T might be CPU/GPU costly, but is explained mostly in docs, not code.

⸻

9. Theoretical Highlight: Forward Kinematics (Moderate Depth)

Definition:
Forward kinematics is how we place 3D coordinates from internal angles. Each residue is a link with fixed bond lengths/angles, but variable dihedrals.

Algorithm (short pseudo-logic):

coords[0] = place_first_residue(torsion_angles[0])
for i in range(1, N_res):
    anchor = coords[i-1]  # e.g. O3'(i-1)
    angles = torsion_angles[i]
    coords[i] = build_next_residue(anchor, angles, standard_geom)
# Optionally refine sugar ring or do partial local minimization

Sugar Pucker:
We can fix as C3′-endo or incorporate a predicted pseudorotation angle.
Environment:
If we want strict ring closure, we do a small local constraint or post-run geometry fix.
In the docs:
Stage_C.md plus the heavier theoretical expansions in advanced “NeRF-like approach” from certain doc references.

⸻

10. Conclusion & Next Steps

Our pipeline provides:
	1.	Modular Stage A for adjacency (or skip if you have adjacency).
	2.	Stage B: a flexible neural approach for torsion angles (AtomAttention or TorsionBERT).
	3.	Stage C: forward kinematics to 3D, optional local refinement.
	4.	(Advanced) Diffusion or isosteric design, Kaggle usage.

Strengths:
	•	Detailed documentation bridging each step.
	•	Code synergy with local block-sparse attention for large RNAs.
	•	Performance benchmarks included.
	•	Exploratory advanced modules.

Limitations:
	•	Not all stages are in a single pipeline script; some remain conceptual or doc-based (esp. forward_kinematics, diffusion).
	•	No direct full AF3 trunk with triangular updates or MSA row attention.
	•	Sugar ring closure or synergy with MD is user-driven.

Recommended further expansions:
	1.	Implement a full “forward_kinematics.py” for Stage C, referencing docs’ pseudocode.
	2.	Provide a small MLP “torsion_head.py” or integrated angle predictor after InputFeatureEmbedder.
	3.	Potentially unify in a single pipeline script with command-line usage.
	4.	If advanced: integrate a partial diffusion refinement module, plus a confidence head akin to pLDDT for final structure quality.

By following the steps in this doc and the more specialized instructions scattered across docs/, you can fully replicate or adapt the pipeline to your needs—from adjacency-based angle prediction to building and refining 3D RNA structures. This system forms a strong basis for future expansions, bridging advanced AI concepts (like S4-based diffusion) with classical geometry-based approaches (forward kinematics, energy minimization) to tackle RNA structural modeling and design challenges.

⸻

That concludes the integrated, verbose reference doc merging the best qualities of Versions 1–4 while addressing each version’s critiques.