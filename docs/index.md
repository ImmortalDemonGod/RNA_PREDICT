# RNA_PREDICT Documentation

Welcome to the RNA_PREDICT documentation homepage! This comprehensive guide provides detailed navigation and insights into all documentation files within the `docs/` directory. Organized clearly, this resource aims to quickly familiarize users and collaborators with the documentation, ensuring efficient and targeted exploration of topics essential for RNA 3D prediction workflows.

---

## 📌 Quickstart Project Commands

Easily manage your MkDocs documentation using these essential commands:

- **`mkdocs new [dir-name]`**: Initialize a new MkDocs project.
- **`mkdocs serve`**: Launch a local live-reloading documentation server.
- **`mkdocs build`**: Generate static HTML documentation for deployment.
- **`mkdocs -h`**: Display help information for MkDocs.

---

## 📂 Comprehensive Documentation Overview

### 🧬 RNA Prediction Pipeline

- **`AlphaFold3_progress.md`**

    - Tracks progress in adapting AlphaFold 3 methodologies specifically for RNA.

    - Highlights implemented components, remaining modules, and clearly defined next steps.

    - Recommended for developers and researchers working on AlphaFold-inspired RNA prediction.

- **`Multi_Stage_Implementation_Plan.md`**
    - Details the technical architecture and phased rollout plan of the RNA 3D prediction pipeline.
    - Ideal for technical architects and project managers.

- **Stage-specific Documentation:**

    - **StageA_RFold.md**: Covers RNA folding stage (Stage A) using the RFold approach.

    - **Stage_B.md**: Describes the intermediate torsion-angle generation from 2D structures.

    - **Stage_C.md**: Explains final generation of Cartesian coordinates from torsion angles.

    - Useful for clearly understanding the modular responsibilities and dependencies within the pipeline.

- **`core_framework.md`**
    - Outlines the pipeline’s three-stage process clearly:
        1. Sequence → 2D structure
        2. 2D structure → Torsion angles
        3. Torsion angles → 3D coordinates
    - Recommended for new team members and collaborators for a clear pipeline overview.

### 📑 Reference and Research Materials

#### 📐 Torsion Angles & Geometric Calculations

- **`torsion_angles.md`**
    - Comprehensive overview of RNA torsion angles, calculation methodologies, software tools (e.g., PyMOL, Chimera), and theoretical considerations.
    - Essential for researchers and developers working with RNA geometry.

- **`torsion_angle_Latent_Manifold_Representation.md`**
    - Proposes innovative methods using lower-dimensional latent manifolds (autoencoders, VAEs) for RNA conformation representation.
    - Targeted at advanced researchers exploring next-gen dimensionality reduction techniques.

#### 📖 External Literature & References

- **`RNA_papers.md`**
    - Compares different curated reference lists, highlighting critical RNA 3D structure prediction papers relevant to competitions and practical applications.

- **`2d_structure_prediction_papers.md`**
    - Specialized compilation of references specifically focusing on RNA secondary (2D) structure prediction methods.

- **`RNA_STRUCTURE_PREDICTION_Categorized.csv`**
    - Structured, categorized references to facilitate efficient literature review and method benchmarking.

- **`ConnectedPapers-for-RNA-secondary-structure-prediction-using-an-ensemble-of-two_20dimensional-deep-neural-networks-and-transfer-learning.txt`**
    - Provides curated insights from Connected Papers about RNA secondary prediction using advanced ensemble and transfer-learning methods.

#### 🔄 Isostericity and Sequence Conservation

- **`RNA_isostericity.md`**
    - Details principles of RNA isostericity, substitution algorithms, and their significance in structural modeling and preservation.

### 🚀 Advanced Modeling and Techniques

#### 🌊 Diffusion and State-Space Modeling

- **`s4_diffusion.md`**
    - Introduces Liquid-S4 state-space models and their empirical performance advantages in long-sequence modeling.
    - Offers integration strategies within AlphaFold-inspired pipelines.

- **`test_time_scaling.md`**
    - Discusses practical strategies to optimize inference speed and quality using flexible step-count adjustments in diffusion-based models.

#### 🧠 AlphaFold Adaptation

- **`AF3_paper.md`**
  - Summarizes AlphaFold 3 foundational concepts, pipeline structure, key innovations, and applicability to RNA prediction.

### 🎯 Competition Context and Practical Applications

- **`kaggle_competition.md`**
    - Comprehensive overview of the Stanford RNA 3D Folding Kaggle challenge.
    - Explains datasets, submission criteria, scoring metrics, and critical FAQs.
    - Indispensable resource for competitors and teams preparing for Kaggle submissions.

---

## 🌐 Interconnections and Recommended Usage

### 🔗 Pipeline Integration
- Core pipeline documents (`core_framework.md`, stage-specific files) articulate clear progression paths from theoretical concepts (`torsion_angles.md`) to practical competitive applications (`kaggle_competition.md`).

### 🌟 Cutting-edge Explorations
- Advanced documentation (`s4_diffusion.md`, `torsion_angle_Latent_Manifold_Representation.md`) outlines future enhancements and novel research trajectories, ensuring the project remains aligned with frontier research.

### 📘 Comprehensive Reference Resources
- External literature documents form a robust base for validating methodologies, guiding strategic implementation, and establishing scientific rigor in RNA structure prediction.

---

## 🚩 Recommended Exploration Pathways

- **New Users & Team Members:**
    - Begin with the pipeline overview (`core_framework.md`), progress to geometric foundational knowledge (`torsion_angles.md`), and conclude with practical competition context (`kaggle_competition.md`).

- **Advanced Practitioners & Researchers:**
    - Explore cutting-edge representation and modeling methods in `torsion_angle_Latent_Manifold_Representation.md` and state-space diffusion strategies outlined in `s4_diffusion.md`.

- **Research-oriented Users:**
    - Leverage curated external literature resources (`RNA_papers.md`, `2d_structure_prediction_papers.md`) for academic rigor, benchmarking, and method validation.

---

This enhanced documentation overview aims to optimize user engagement, foster efficient navigation, and encourage collaborative contribution to the RNA_PREDICT project. Happy exploring and contributing!

