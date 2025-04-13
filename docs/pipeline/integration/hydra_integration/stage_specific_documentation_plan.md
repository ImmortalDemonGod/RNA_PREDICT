# **HYDRA & STAGE-SPECIFIC DOCUMENTATION: MASTER PLAN**

Below is a **unified, in-depth reference** that merges your **existing Hydra Integration Plan** with a **systematic approach** to generating **specialized documentation** for each pipeline stage (A–D) and the Unified Latent Merger. It aims to ensure that when you refactor each stage, you'll have thorough reference materials connecting code logic, Hydra configuration, algorithmic details, and integration points.

---

## **1. Comprehensive Feature Documentation (Hydra) — Already Done**

You already have a "Hydra Integration Master Document" that:
- Explains why Hydra is beneficial (centralization, easy overrides, reproducibility).
- Details how to structure `conf/` with YAML files, typed dataclasses in `config_schema.py`, and how to refactor each stage to read `cfg` instead of inline defaults.
- Includes acceptance criteria ensuring backward compatibility, successful CLI overrides, and default param matching.

That **comprehensive Hydra document** covers the feature's high-level design, scope, acceptance criteria, step-by-step implementation plan, and references to all pipeline stages. It's your **central** guide for implementing Hydra across the entire codebase.

## **2. General Considerations (Hydra + Documentation)**

You've also enumerated **general considerations** such as:

- Code directory structure in `rna_predict/conf/`.
- Logging HPC environment or partial checkpoint loading.
- The need for iterative refinement of docstrings and minimal HPC notes, etc.

These ensure your Hydra-based approach remains robust over time. 

## **3. Plan to Generate Specialized Documentation Per Stage**

Now, we focus on **task-specific documentation** for each stage and the unified latent merger. The goal is to create a dedicated reference for each major component in your pipeline, clarifying how **Hydra-managed parameters** align with the stage's logic and function calls.

### **3.1 Overall Methodology**

1. **Automated Extraction**  
   - Use a doc generator (Sphinx, PyDoctor, or similar) to produce base-level references from docstrings, class definitions, and module docblocks.
   - This yields an initial skeleton capturing function/class signatures, parameters, and any existing docstring descriptions.

2. **Manual Enrichment**  
   - Conduct a structured code review for each stage. Identify:
     - "Magic numbers," hidden assumptions, or specialized data transformations,
     - Key integration points (e.g., adjacency from Stage A feeding into Pairformer in Stage B),
     - Known edge cases or fallback logic (like Stage C legacy reconstruction).
   - Merge these findings into your doc generator output to create in-depth references that go beyond auto-generated docs.

3. **Documentation Outline**  
   - For each stage (A–D plus the Unified Latent Merger), follow a consistent template:
     - **Purpose & Overview**  
       \- Short paragraph explaining the stage's function and motivation (e.g., "Stage A: Generate a contact matrix for RNA base pairs.").
     - **Inputs & Outputs**  
       \- Data shapes, types, and any assumptions (sin/cos angles, adjacency dimensions, etc.).  
       \- Expected Hydra config parameters that might affect these shapes or the stage's logic.
     - **Key Classes & Methods**  
       \- Summaries of the top-level classes and methods (like `StageARFoldPredictor`) with bullet-point commentary about their roles.  
       \- Focus on lines or subroutines that are vital for understanding how Hydra's config drives them.
     - **Configuration Parameters**  
       \- A listing of the YAML fields relevant to this stage (e.g., `stageA.num_hidden`, `stageA.dropout`, etc.), including defaults, usage, and any synergy with upstream or downstream parameters.  
       \- Example of how these values appear in the YAML (`conf/model/stageA.yaml`) and how they map to the code.
     - **Integration & Data Flow**  
       \- Which module feeds into this stage, and which stage(s) rely on its outputs?  
       \- If adjacency from Stage A is used by TorsionBERT or Pairformer, note that explicitly.
     - **Edge Cases & Error Handling**  
       \- Mention what the code does when inputs are malformed, config is out-of-range, or external files (e.g., checkpoint paths) are missing.
     - **References & Dependencies**  
       \- Links to external docs (like the RFold paper) or any specialized library usage (like LoRA injection modules, MP-NeRF references).

4. **Iterate & Validate**  
   - Circulate these specialized docs among developers for correctness.  
   - Update them in tandem with code changes or as part of the PR (pull request) process to keep them fresh.

---

## **4. Stage-by-Stage Documentation Breakdown**

This section sketches an **ideal, final** content structure for each specialized doc. You can store them either in separate Markdown files (one per stage) or unify them under one "RNA_PREDICT Stage Docs" directory. The essential idea is consistent coverage.

### **4.1 Stage A: 2D Adjacency Prediction (via RFold)**

**Title**: *StageA_2D_Adjacency.md*

**Sections**:  
1. **Purpose & Background**  
   - Summarize how adjacency prediction is the first step in the pipeline, providing base-pair contacts.  
   - Reference any relevant paper or logic used (like K-rook approach from RFold).
2. **Inputs & Outputs**  
   - Input: Single RNA sequence (string), possible multi-line FASTA, etc.  
   - Output: NxN adjacency matrix (probabilistic or binary).  
   - Possible shapes or thresholding methods described.
3. **Key Classes & Scripts**  
   - `StageARFoldPredictor` class  
     - Fields: `num_hidden`, `dropout`, `checkpoint_path`, device usage.  
     - Notable methods: `_get_cut_len()`, `predict_adjacency()`.
   - `run_stageA.py` entry logic: If it downloads missing checkpoints or calls visualization with VARNA.
4. **Hydra Configuration**  
   - List `stageA.num_hidden`, `stageA.dropout`, etc., from `stageA.yaml`.  
   - Provide short code snippet: "In `stageA.yaml`, we define `num_hidden: 128`, etc."
   - Show how `StageARFoldPredictor` loads these config values.
5. **Integration**  
   - Upstream: none. (Stage A is the pipeline start for structure.)  
   - Downstream: Stage B's TorsionBERT or Pairformer might read adjacency.  
   - Additional references: If adjacency is visualized with `varna_jar_path`.
6. **Edge Cases & Logging**  
   - If the sequence is < 4 nucleotides, if the checkpoint is missing/corrupted, if a mismatch occurs in dimension, etc.
7. **References**  
   - Link to the original RFold paper or relevant GitHub repos.  
   - Mention if it depends on a certain PyTorch version or other library for adjacency calculation.

---

### **4.2 Stage B: TorsionBERT (LoRA optional) & Pairformer (LoRA optional)**

**Title**: *StageB_TorsionBERT_and_Pairformer.md*

**Sections**:  
1. **Overview & Role**  
   - This stage outputs local angles (TorsionBERT) and pair embeddings (Pairformer) for each residue/residue pair.  
   - Quick rationale: angles feed partial 3D builds, pair embeddings capture global context.
2. **Inputs & Outputs**  
   - Inputs: Possibly adjacency matrix from Stage A, raw sequence, or other features.  
   - Outputs: Torsion angles shaped `[N, K or 2K]` if using sin/cos, single embeddings `[N, c_s]`, pair embeddings `[N, N, c_z]`.
3. **Key Classes**  
   - `StageBTorsionBertPredictor`, `PairformerWrapper`, plus any support code.  
   - If LoRA is used, identify injection points in attention or feed-forward layers.
4. **Hydra Config**  
   - TorsionBert parameters from `stageB_torsion.yaml` (like `model_name_or_path`, `lora.enabled`, `lora.r`).  
   - Pairformer parameters from `stageB_pairformer.yaml` (like `c_s`, `c_z`, `use_checkpoint`).  
   - Show examples of toggling LoRA or specifying target modules in YAML.
5. **Integration**  
   - Upstream: adjacency or sequence from Stage A.  
   - Downstream: Stage C uses angles for partial 3D. Pair embeddings might pass to the Unified Merger. 
6. **Edge Cases & Error Handling**  
   - Sequence indexing issues, mismatch in adjacency dimension, or missing LoRA checkpoint.  
   - Potential timeouts or memory constraints if `n_blocks` or `c_z` is large.
7. **References**  
   - If your TorsionBERT is based on an existing HF model or a custom paper, link it.  
   - If Pairformer references a particular paper or approach for pairwise embeddings, mention it.

---

### **4.3 Stage C: 3D Reconstruction (MP-NeRF or Fallback)**

**Title**: *StageC_3D_Reconstruction.md*

**Sections**:  
1. **Objective & Methods**  
   - Outline how torsion angles become backbone coordinates, referencing MP-NeRF or the trivial fallback.  
   - Possibly mention sugar pucker default "C3'-endo" and ring closure logic.
2. **Inputs & Outputs**  
   - Input: Torsion angles from Stage B, optionally adjacency or sequence.  
   - Output: 3D coords shaped `[N, #atoms, 3]` or `[N * #atoms, 3]`.
3. **Core Classes & Functions**  
   - `run_stageC_rna_mpnerf(...)`, `StageCReconstruction` fallback.  
   - Describe how angles are clipped or expanded if user sets a different `expected_torsion_count`.
4. **Hydra Config**  
   - `stageC.yaml` entries: `method`, `do_ring_closure`, `place_bases`, `sugar_pucker`, etc.  
   - Show how code picks the method string to decide on MP-NeRF or fallback. 
5. **Integration & Data Flow**  
   - Outputs partial 3D coords for the Unified Latent Merger or Stage D directly (depending on your pipeline design).  
   - Mention memory usage, potential HPC constraints for large sequences.
6. **Edge Cases & Error Handling**  
   - If angles are incomplete or if user tries "legacy" fallback.  
   - Negative or out-of-range angles if TorsionBERT is uninitialized, etc.

---

### **Unified Latent Merger**

**Title**: *UnifiedLatentMerger.md*

**Sections**:  
1. **Overview & Importance**  
   - This merges adjacency, angles, partial coords, single/pair embeddings into a single "conditioning latent" for Stage D.  
   - Ensures synergy between local angle features and global pair embeddings.
2. **Input Structures**  
   - For example, `[N, dim_angles]` for angles, `[N, c_s]` single embeddings, `[N, N, c_z]` pair embeddings, `[N, #atoms, 3 or N* #atoms, 3]` partial coords.  
   - Possibly row-wise pooling of pair embeddings to get `[N, c_z]`.
3. **Merger Architecture**  
   - If it's a simple MLP (like `SimpleLatentMerger`), specify hidden layers, activation, dropout.  
   - If you might adopt a small transformer in the future, mention that possibility. 
4. **Hydra Config**  
   - From `latent_merger.yaml`: `dim_angles`, `dim_s`, `dim_z`, `dim_out`, etc.  
   - How you handle dimension mismatches or expansions (`z_pooled = z.mean(dim=1)`).
5. **Integration**  
   - Upstream: results from Stage A/B/C.  
   - Downstream: Stage D diffusion uses the merged latent as a conditioning input.
6. **Edge Cases**  
   - If the shapes from upstream do not match config expectations. Possibly mention logging or shape asserts.

---

### **4.4 Stage D: Diffusion-based Refinement & Optional Energy Minimization**

**Title**: *StageD_Diffusion_and_Minimization.md*

**Sections**:  
1. **Purpose**  
   - Final step refining partial 3D coords or random initialization, possibly using advanced diffusion techniques.  
   - Optional: short local MD or energy minimization for finishing.
2. **Input & Output**  
   - Input: Partial coords from Stage C or random noise, plus the unified latent.  
   - Output: final 3D structure [N, #atoms, 3], possibly an ensemble from multiple diffusion seeds.
3. **Core Modules**  
   - `run_stageD.py`, `protenix_diffusion_manager.py`, memory optimization logic.  
   - If using LoRA for large diffusion models, show how it's toggled in `cfg.stageD.lora`.
4. **Hydra Config**  
   - `stageD_diffusion.yaml`: fields like `sigma_data`, `gamma0`, `n_steps`, `c_atom`, `c_token`, etc.  
   - `memory_optimization.enable`: if it modifies the forward pass or chunking.  
   - `energy_minimization.enabled`, `steps`, `method`.
5. **Integration**  
   - Receives the unified latent from the Merger.  
   - May produce final coords used for post-processing or output files.
6. **Edge Cases**  
   - Divergent diffusion steps, missing partial coords, HPC memory constraints.  
   - Minimization method not installed or recognized (e.g., OpenMM missing).

---

## **5. Iterative Refinement & Maintenance**

1. **Peer-Review Documentation**  
   - Encourage multiple contributors to add clarifications or correct minor oversights as the pipeline evolves.  

2. **Continuous Update Process**  
   - Each time you add a parameter in a stage's code, reflect that in the specialized doc and the Hydra YAML (`.yaml`).

3. **Versioning**  
   - If the pipeline undergoes major changes (like replacing MP-NeRF with a new method), increment doc versions or date-stamp them.

---

## **6. Summary of the Whole Approach**

1. **Comprehensive Hydra Doc**: *Already done.*  
   - Explains how to implement and maintain Hydra across all stages, ensuring consistent usage and backward-compatible defaults.

2. **General Considerations**:  
   - HPC environment, partial checkpoint loading, shape validations, logging, advanced overrides, etc. *Already enumerated.*

3. **Specialized Stage Docs**: *In Progress.*  
   - The plan above ensures each stage (A–D + Merger) is documented with:
     - **Purpose** & **High-level algorithm**  
     - **Core classes & Hydra config** references  
     - **Inputs/outputs** (shapes, types)  
     - **Integration** with prior/following pipeline stages  
     - **Edge cases & error handling**  
     - **LoRA toggles** or specialized features (like ring closure in Stage C or memory fixes in Stage D)  

With these three pillars (hydra doc, general considerations, specialized stage references), your pipeline refactoring becomes well-documented and maintainable. As you proceed to implement or update each stage with Hydra's parameters, you can rely on the specialized doc as a "living blueprint," ensuring no hidden logic or shape mismatch remains undocumented.

> **Next Steps**:  
> 1. Generate **base auto-documentation** (via Sphinx or similar).  
> 2. Conduct **manual code reviews** to annotate each stage's key logic.  
> 3. Draft **specialized docs** per stage using the structure above.  
> 4. Link these specialized docs with your **Hydra config references** (in `conf/`) so that every parameter is well-explained and mapped to code usage.  
> 5. Keep them updated as the pipeline evolves and new features or LoRA adapters get integrated.

This final approach merges your high-level Hydra integration strategy with a systematic plan for creating **specialized documentation**—ensuring that each stage is thoroughly described and that the Hydra-based configuration references are never out of date. 