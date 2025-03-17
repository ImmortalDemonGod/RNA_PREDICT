# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
=======
# Documentation Overview

Below you’ll find a document-by-document analysis of the files in this repository’s `docs/` folder. Each section outlines the purpose, structure, and key takeaways of the respective document, helping newcomers and collaborators quickly grasp where to look for specific information.

---

## 1. `docs/AlphaFold3_progress.md`

**Purpose & Context**  
- A comprehensive status report on re-implementing or extending an AlphaFold 3 (AF3)–style system for RNA structure prediction.  
- Compares official AF3 pipeline components to what’s already built, highlighting missing parts.

**Structure & Key Points**  
1. **Introduction**: Explains the goal—an AF3-like pipeline specializing in RNA.  
2. **Summary of Achievements**:
   - Data & Feature Prep: Streaming with Hugging Face’s `bprna-spot` dataset, synthetic feature dictionaries, etc.
   - Model Components: Implementation of atom attention, local block-sparse code, placeholders for trunk recycling.
   - Benchmarks & Organization: GPU memory tracking scripts, modular code structure.
3. **Comparison to Official AF3**: Identifies which major modules (multi-dataset training, MSA module, Pairformer stack, diffusion head, confidence heads) aren’t yet implemented.  
4. **Action Items**: Detailed next steps—data pipeline expansions, MSA/pairformer blocks, multi-stage training routines, etc.

**Notable Nuances**  
- You’ll see exactly how your current code lines up against the official AF3 pipeline.  
- It doubles as a roadmap for continuing the AF3-like implementation, showing precisely where to focus next (e.g., diffusion module, confidence heads).

**Takeaway**  
A crucial document if you’re actively building an AF3-inspired pipeline for RNA. It’s effectively both a progress summary and a to-do list.

---

## 2. `docs/RNA_papers.md`

**Purpose & Context**  
- Compares three versions (V1, V2, V3) of a reference list on RNA 3D prediction methods, intended for a Stanford RNA 3D Folding Kaggle context.

**Structure & Key Points**  
1. **Side-by-Side Comparison**: V1 is detailed and enumerates 10 references, V2 is more concise, and V3 is thematically organized.  
2. **Highlights & Differences**: Explains how each version handles deep learning vs. physics-based approaches, motif references, benchmarking mention, etc.  
3. **Which Papers Are Most Useful**: Calls out RhoFold+, NuFold, Vfold, CASP15 assessment, RNA-Puzzles, foundation models—likely the top references to consult for any RNA 3D competition.

**Nuances**  
- It’s somewhat “meta”: an internal commentary on three alternative lists.  
- If you need a single master reference doc, the final section identifies the overlapping essential references.

**Takeaway**  
Use this file to decide which version (V1, V2, or V3) best fits your style—most comprehensive or more streamlined—and to see which papers are universally recommended for RNA 3D structure tasks.

---

## 3. `docs/core_framework.md`

**Purpose & Context**  
- Lays out a 3-step pipeline for RNA structure prediction:
  1. Sequence → 2D structure
  2. 2D structure → Torsion angles
  3. Torsion angles → 3D coordinates

**Structure & Key Points**  
1. **Model Breakdown**:
   - (A) Predict base-pairing (2D) plus relevant stats from sequence.
   - (B) Convert 2D + stats into backbone torsion angles.
   - (C) Convert torsion angles to full 3D Cartesian coordinates.  
2. **Training Plan**: Each stage can be trained separately, enabling a modular approach.  

**Nuances**  
- Emphasizes a clean, hierarchical design where each output feeds the next.  
- Perfect for teams wanting to partition responsibilities (e.g., one team on 2D structure, another on torsion modeling).

**Takeaway**  
A high-level conceptual flow for an RNA pipeline, clarifying how each sub-model fits together to produce final 3D predictions.

---

## 4. `docs/s4_diffusion.md`

**Purpose & Context**  
- Describes Liquid-S4 (an extension of S4 state-space models) and how it performs on long-sequence benchmarks. Also shows how it might integrate into an AlphaFold3-like pipeline.

**Structure & Key Points**  
1. **Main Contributions**: Introduces Liquid-S4’s “liquid” convolution kernel, highlights empirical successes (LRA tasks, speech, sCIFAR).  
2. **Experimental Results**: Tables comparing Liquid-S4 to Transformers, Reformer, S4 variants, etc.  
3. **Kernel Computation**: Offers pseudocode for the “Power-of-B” approach in JAX-like style.  
4. **Hyperparameters**: A table listing typical per-task configurations.  
5. **Addendum**: Deeper notes on S4 math, LTC, S5. Also an “AF3 bridging” note— how to use big unrolls at test time in diffusion without re-checking adjacency.

**Nuances**  
- Half a direct research summary of Liquid-S4, half a set of pointers for adapting AF3’s diffusion stage.  
- Great if you’re exploring state-space or diffusion-based methods for large-scale RNA structure tasks.

**Takeaway**  
A reference doc for any advanced user wanting to integrate S4-based layers, possibly in an AlphaFold-like trunk or in a diffusion stage for RNA or protein structure modeling.

---

## 5. `docs/test_time_scaling.md`

**Purpose & Context**  
- Explains how diffusion models let you adjust the number of denoising steps at inference, trading speed vs. sample quality.

**Structure & Key Points**  
1. **Why This Works**: Basic diffusion overview, skipping steps or adopting bigger intervals for faster but potentially lower-quality outputs.  
2. **Practical Tips**: Mentions discrete vs. continuous solvers, flexible step counts, adaptive error checks.

**Nuances**  
- A short doc focusing solely on the concept of controlling “T” in diffusion-based generation.  
- Reinforces that you don’t have to re-train if you want fewer or more test steps—common in image and molecular diffusion approaches.

**Takeaway**  
A quick read for deciding your sampling strategy in diffusion—fewer steps for speed or more for fidelity. Helpful if you’re implementing a diffusion-based RNA pipeline and want to experiment with test-time hyperparameters.

---

## 6. `docs/torsion_angles.md`

**Purpose & Context**  
- A “mini-guide” on RNA torsion angles: definitions, calculation methods, relevant software, theoretical approaches, and advanced considerations.

**Structure & Key Points**  
1. **Basic Intro**: Names each torsion (α–ζ, χ, sugar pucker).  
2. **Algorithm for Calculation**: Step-by-step dihedral angle formula with cross products, `atan2`, sign determination.  
3. **Software Tools**: PyMOL, Chimera, 3DNA, DSSR, Barnaba, MD packages (Amber, GROMACS, MDAnalysis), RNAtango, etc.  
4. **Theoretical Approaches**: Rotamer libraries, pseudo-torsions (η/θ), sugar pucker pseudorotation.  
5. **Advanced Tech Details**: Numeric stability, boundary conditions, ring closure constraints.

**Nuances**  
- Very thorough: from the geometry basics to potential correlation frameworks (rotamers, rarely used puckers).  
- Cites practical tools for everyday tasks (like batch torsion analysis, MD trajectory extraction).

**Takeaway**  
If you’re new to RNA geometry or building an RNA structure predictor that manipulates torsion angles, this is the ultimate reference for angles: the “why” and “how” plus tool suggestions.

---

## 7. `docs/torsion_angle_Latent_Manifold_Representation.md`

**Purpose & Context**  
- Explores whether you can go beyond standard torsion angles to an even lower-dimensional “latent manifold” for RNA conformation. Think: autoencoders or VAEs that compress the molecule’s shape into a handful of latent variables.

**Structure & Key Points**  
1. **Proposed LLMR**: Argues that real RNA structure might lie on a much lower-dimensional manifold than even the 7–10 angles/residue.  
2. **Comparison**: Cartesian vs. Torsion vs. Learned Latent. Summaries of parameter count, memory usage, etc.  
3. **Use Cases**: Potentially large speedups for big RNAs, easier global sampling if the model learns typical fold constraints.

**Nuances**  
- The doc acknowledges you must still decode from latent code → physically valid 3D (hence a robust decoder).  
- Great for advanced researchers—would require significant data to train such a manifold-based approach well.

**Takeaway**  
A forward-looking perspective on compressing RNA geometry. Could be a powerful approach if you have extensive data and want minimal degrees of freedom for large-scale tasks.

---

## 8. `docs/kaggle_competition.md`

**Purpose & Context**  
- Distills the structure and rules of the Stanford RNA 3D Folding challenge on Kaggle. Explains the dataset, submission format, timeline, and scoring with TM-score.

**Structure & Key Points**  
1. **Competition Goal**: Predict the C1′ coordinates for each residue from the raw RNA sequence (five submissions per residue, best-of-5 used).  
2. **Data**: `train_sequences.csv` / `train_labels.csv` (844 RNAs), `validation_*`, `test_sequences.csv`, plus MSAs and possibly synthetic expansions.  
3. **Scoring**: TM-score alignment, best-of-5 approach. The average across all test targets forms your leaderboard score.  
4. **Timeline**: Start date, public leaderboard refresh, final submission, future data phase.  
5. **Common Questions**: E.g., multi-conformation usage, temporal cutoff rules, how to handle real vs. synthetic data.

**Nuances**  
- Mentions a multi-structure reference scenario in training data (some RNAs have multiple known conformations).  
- Encourages or allows external data sources if they respect cutoff constraints.

**Takeaway**  
This doc is crucial if you’re actively competing or training a model for the Kaggle challenge. It explains how to handle the dataset’s intricacies, the 5-model submission, and compliance with competition rules.

---

## Overall Synergies and Key Observations

1. **Comprehensive RNA 3D Resource**  
   - These docs collectively form a knowledge base covering fundamental geometry (torsion angles) through advanced approaches (Liquid-S4, learned latent manifolds), plus a practical competition use-case (the Kaggle challenge).

2. **Implementation Roadmaps**  
   - Documents like `AlphaFold3_progress.md` and `s4_diffusion.md` offer partial engineering steps or references for advanced modeling approaches (AF3 pipeline expansions, S4-based diffusion, etc.).

3. **Competition Integration**  
   - The Kaggle competition doc anchors these ideas in a real challenge scenario, clarifying how to combine stepwise frameworks (`core_framework.md`) or references (`RNA_papers.md`) to create a top-scoring solution.

4. **Advanced Torsion & Manifold Strategies**  
   - For geometry enthusiasts, `torsion_angles.md` and `torsion_angle_Latent_Manifold_Representation.md` show how to either stick to classical dihedrals or push further with data-driven dimensionality reduction.

---

### Where to Go from Here

- **If You’re New**: Skim `core_framework.md` to see the overall pipeline, then use `torsion_angles.md` for geometry basics, `kaggle_competition.md` for challenge details, and `AlphaFold3_progress.md` to understand the AF3-like approach status.
- **If You Want Cutting-Edge**: Check `s4_diffusion.md` for Liquid-S4 or `torsion_angle_Latent_Manifold_Representation.md` for manifold-based compression.
- **If You Need References**: `RNA_papers.md` helps you find crucial papers; `AlphaFold3_progress.md` links your progress to official AF3 design.

---

*We hope this guide clarifies each document’s purpose and interconnections. Happy exploring!*
===
V2:
# RNA_PREDICT Documentation

Welcome to the RNA_PREDICT documentation homepage! This page provides a structured, detailed overview of all documentation files within the `docs/` directory, organized to help users quickly navigate and understand the resources available in this project. Each document is summarized to clarify its context, content structure, key points, and usage scenarios.

---

## 📌 Project Commands Overview

To quickly manage and interact with this MkDocs documentation, utilize the following commands:

- **`mkdocs new [dir-name]`**: Create a new MkDocs project in the specified directory.
- **`mkdocs serve`**: Launch a live-reloading local documentation server for rapid editing and review.
- **`mkdocs build`**: Generate a static HTML documentation website for deployment.
- **`mkdocs -h`**: Display a help message outlining available commands and usage.

---

## 📁 Documentation File Breakdown

### 📖 Pipeline Documentation

- **`AlphaFold3_progress.md`**:
  - Tracks the implementation progress of an RNA-specific pipeline inspired by AlphaFold 3 (AF3).
  - Lists implemented components, pending modules, and future action steps clearly.
  - Essential for those involved in pipeline development and AF3 model adaptation.

- **`Multi_Stage_Implementation_Plan.md`**:
  - Details the technical architecture and phased rollout strategy for the RNA 3D prediction pipeline.
  - Useful for technical leads overseeing the project’s structural evolution.

- **Stage-specific Documentation**:
  - **StageA_RFold.md**: Details Stage A, focused on RNA folding.
  - **Stage_B.md**: Covers intermediate torsion angle generation.
  - **Stage_C.md**: Describes final Cartesian coordinate generation.
  - Ideal for understanding modular responsibilities and interdependencies.

- **`core_framework.md`**:
  - Outlines a structured, three-step pipeline (sequence → 2D structure → torsion angles → 3D structure).
  - Ideal for team onboarding and understanding modular task assignments.

### 📚 Reference and Research Resources

- **Torsion Angle Documentation**:
  - **`torsion_angles.md`**:
    - Comprehensive guide covering definitions, computational methods, tools, theoretical frameworks, and advanced considerations for RNA torsion angles.
    - Recommended as a foundational resource for researchers and developers.

  - **`torsion_angle_Latent_Manifold_Representation.md`**:
    - Proposes innovative methods for RNA conformation representation using lower-dimensional latent spaces (e.g., autoencoders, VAEs).
    - Aimed at advanced researchers exploring cutting-edge representation strategies.

- **External Literature**:
  - **`RNA_papers.md`**:
    - Analyzes multiple reference list versions for RNA 3D structure prediction methods.
    - Highlights essential papers (e.g., NuFold, CASP15, RNA-Puzzles) critical for competitive RNA prediction.

  - **`2d_structure_prediction_papers.md`**:
    - Curates literature specifically on RNA secondary (2D) structure prediction methodologies.

  - **`RNA_STRUCTURE_PREDICTION_Categorized.csv`**:
    - Categorized dataset offering structured references for RNA prediction literature, facilitating efficient literature review.

  - **`ConnectedPapers-for-RNA-secondary-structure-prediction-using-an-ensemble-of-two_20dimensional-deep-neural-networks-and-transfer-learning.txt`**:
    - Captures insights from Connected Papers related to ensemble and transfer-learning-based RNA secondary prediction.

- **Isostericity Reference**:
  - **`RNA_isostericity.md`**:
    - Explores RNA isostericity, detailing the theory, significance, and practical implications for RNA modeling.

### ⚙️ Advanced Methods and Techniques

- **Diffusion Models**:
  - **`s4_diffusion.md`**:
    - Introduces Liquid-S4 state-space models for diffusion, highlighting experimental outcomes, pseudocode, and integration with AF3-inspired pipelines.

  - **`test_time_scaling.md`**:
    - Discusses adjustable inference strategies in diffusion models, balancing computation speed and result quality.

- **AlphaFold Adaptation**:
  - **`AF3_paper.md`**:
    - Summarizes foundational principles and innovations introduced by AlphaFold 3.
    - Essential reference for those adapting AF3 methodologies to RNA prediction.

### 🎯 Competition and Application Context

- **`kaggle_competition.md`**:
  - Provides comprehensive details about the Stanford RNA 3D Folding challenge on Kaggle.
  - Covers competition goals, datasets, scoring metrics, submission guidelines, and common FAQs.
  - Crucial for competitors preparing submissions and strategizing model training.

---

## 🌐 Inter-document Synergies

- **Pipeline Integration**: Documents like `core_framework.md` and `AlphaFold3_progress.md` articulate clear interfaces between theoretical insights (e.g., `torsion_angles.md`) and practical applications (`kaggle_competition.md`).

- **Cutting-edge Techniques**: Advanced documentation (`s4_diffusion.md`, `torsion_angle_Latent_Manifold_Representation.md`) points toward future pipeline enhancements and novel research directions.

- **Comprehensive Reference Set**: The combination of external literature documentation provides a robust framework for researchers seeking authoritative references to benchmark or validate RNA structure prediction methods.

---

## 🚩 Recommended Next Steps

- **For Newcomers**:
  - Start with `core_framework.md` for foundational understanding, proceed to `torsion_angles.md` for geometrical basics, and use `kaggle_competition.md` for practical application context.

- **For Advanced Practitioners**:
  - Explore innovative representation methods in `torsion_angle_Latent_Manifold_Representation.md` or integrate S4-based models described in `s4_diffusion.md`.

- **For Research-oriented Users**:
  - Use external literature docs (`RNA_papers.md`, `2d_structure_prediction_papers.md`) to align your work with current scientific standards and benchmarks.

---

This structured overview is intended to optimize your engagement with the RNA_PREDICT documentation, accelerating both onboarding and deep technical engagement. Happy exploring and contributing!

=====
V3:
Below is a suggested organizational structure for your document set, grouping them by pipeline stage, supporting references, and advanced methods. The goal is to make it easy for collaborators (or your future self) to locate the right file for each step of the RNA 3D pipeline, as well as any extra in-depth or next-gen resources.

⸻

1. Main Pipeline Files

(A) Stage A – 2D Predictor
	•	StageA_RFold.md
	•	Integrates the K-rook-based RFold approach, guaranteeing valid base pairs.
	•	Proposed new location: pipeline/stageA/StageA_RFold.md.

(B) Stage B – Torsion-Angle Predictor
	•	Stage_B.md
	•	Detailed design for predicting backbone torsion angles using adjacency from Stage A + GNN/Transformer.
	•	Proposed location: pipeline/stageB/Stage_B.md.

(C) Stage C – Forward Kinematics to 3D
	•	Stage_C.md
	•	Consolidated plan for converting torsion angles → 3D coordinates, including sugar pucker or local minimization.
	•	Proposed location: pipeline/stageC/Stage_C.md.

(D) Multi-Stage Overviews
	•	core_framework.md
	•	High-level 1→2→3 steps (2D→torsion→3D).
	•	Proposed location: pipeline/overview/core_framework.md.
	•	Multi_Stage_RNA3D_Pipeline_Technical_Architecture&Implementation_Plan.md
	•	Comprehensive blueprint that merges older “versions.”
	•	Proposed location: pipeline/overview/Multi_Stage_Implementation_Plan.md.

(E) Competition Context
	•	kaggle_competition.md
	•	Summaries of competition structure, data usage, 5-model submission format, TM-score, etc.
	•	Proposed location: pipeline/kaggle_info/kaggle_competition.md.

These are your main practical docs for each stage, plus the big overview references.

⸻

2. Supporting Materials & In-Depth Guides

2.1 Torsion Angles & 2D→3D Tools
	•	torsion_angles.md
	•	Thorough explanation of how to compute α..ζ, χ, sugar pucker, referencing standard software (3DNA, PyMOL, etc.).
	•	Proposed location: reference/torsion_calculations/torsion_angles.md.
	•	torsion_angle_Latent_Manifold_Representation.md
	•	Argues for a data-driven latent approach beyond classical torsions.
	•	Proposed location: reference/advanced_geom/torsion_angle_Latent_Manifold_Representation.md.

2.2 Isostericity & Sequence Preservation
	•	RNA_isostericity.md
	•	Algorithm for base-pair substitutions preserving 3D geometry without MSAs.
	•	Proposed location: reference/isosteric_substitutions/RNA_isostericity.md.

2.3 Summaries of External References / Papers
	•	RNA_papers.md
	•	Compares multiple “versions” of an RNA references list and identifies which are best for Kaggle.
	•	Proposed location: reference/external_lit/RNA_papers.md.

⸻

3. Advanced or Next-Generation Approaches

3.1 Diffusion & State-Space Models
	•	s4_diffusion.md
	•	Liquid-S4 or S4-based approach for large-step angle or coordinate diffusion.
	•	Proposed location: advanced_methods/diffusion/s4_diffusion.md.
	•	test_time_scaling.md
	•	Explains how diffusion steps can be a test-time knob.
	•	Proposed location: advanced_methods/diffusion/test_time_scaling.md.

3.2 AlphaFold 3–Style Systems
	•	AlphaFold3_progress.md
	•	Merges V1–V3 reviews into a single doc, highlighting partial re-implementation progress.
	•	Proposed location: advanced_methods/af3/AlphaFold3_progress.md.
	•	AF3_paper.md
	•	The large “supplementary style” doc with entire pipeline for AlphaFold 3 (templates, trunk, diffusion, confidence heads).
	•	Proposed location: advanced_methods/af3/AF3_paper.md.

⸻

4. Illustrative Folder/Directory Tree

Here’s what it might look like in practice:

docs/
 ├── pipeline/
 │    ├── overview/
 │    │    ├── core_framework.md
 │    │    └── Multi_Stage_Implementation_Plan.md
 │    ├── stageA/
 │    │    └── StageA_RFold.md
 │    ├── stageB/
 │    │    └── Stage_B.md
 │    ├── stageC/
 │    │    └── Stage_C.md
 │    └── kaggle_info/
 │         └── kaggle_competition.md
 ├── reference/
 │    ├── torsion_calculations/
 │    │    └── torsion_angles.md
 │    ├── advanced_geom/
 │    │    └── torsion_angle_Latent_Manifold_Representation.md
 │    ├── isosteric_substitutions/
 │    │    └── RNA_isostericity.md
 │    └── external_lit/
 │         └── RNA_papers.md
 └── advanced_methods/
      ├── diffusion/
      │    ├── s4_diffusion.md
      │    └── test_time_scaling.md
      └── af3/
           ├── AlphaFold3_progress.md
           └── AF3_paper.md

This structure groups everything by stage (A, B, C) plus a top-level overview folder (for pipeline wide docs) and reference vs. advanced methods subfolders.

⸻

5. Conclusion & Benefits
	•	Clear Stage Flow: The pipeline folder focuses on your day-to-day “build the pipeline” docs: Stage A, Stage B, Stage C, plus the overview.
	•	Supporting “Reference”: Torsion angle tutorials, isostericity design, or external references.
	•	Advanced: If you want to incorporate diffusion or an AlphaFold 3–like approach, jump to advanced_methods/.

With this reorganization, collaborators can easily see how the 2D→Torsion→3D pipeline is constructed (pipeline folder), what optional advanced expansions exist (advanced_methods folder), and how to handle specialized references or knowledge (reference folder).
===