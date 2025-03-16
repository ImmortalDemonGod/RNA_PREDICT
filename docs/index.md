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