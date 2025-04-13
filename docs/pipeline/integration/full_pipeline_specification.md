Unified RNA 3D Prediction Pipeline: “Best-of-All-Worlds” Technical Documentation

Below is a comprehensive, high-level-to-concrete technical guide that merges and improves upon the previous versions (V1–V5). It addresses criticisms from each, consolidates their strengths, and incorporates a cohesive plan for seamlessly integrating LoRA, Pairformer embeddings, a Unified Latent Merger, and Diffusion (Stage D). It is verbose and detailed—designed to serve as a robust blueprint for developers working on an end-to-end RNA structure prediction pipeline.

⸻

1. Introduction & Objectives

This “best-of-all-worlds” architecture aims to unify the entire multi-stage RNA 3D pipeline:
	1.	Stage A: 2D adjacency predictor (e.g., RFold or an external method).
	2.	Stage B:
	•	TorsionBERT (with LoRA) → local angles (α, β, γ, δ, ε, ζ, χ, …).
	•	Pairformer (with LoRA) → global pairwise embeddings \mathbf{z}_{ij} + single embeddings \mathbf{s}_i.
	3.	(Optional) Stage C: MP-NeRF or partial 3D geometry reconstruction from angles.
	4.	Unified Latent Merger: Merges adjacency, angles, partial coords, Pairformer outputs into a single “unified latent” representation.
	5.	Stage D: Diffusion-based refinement (with LoRA optional), which takes the unified latent as conditioning to generate final 3D coordinates.
	6.	(Optional) Energy Minimization: A post-hoc step for short local refinement in a force field.

Key motivations:
	•	Bring local angle predictions and global pair embeddings into a single representation.
	•	Support LoRA to minimize GPU memory usage in large pretrained models (TorsionBERT, Pairformer).
	•	Achieve synergy by letting the Diffusion step see both adjacency-based local constraints and Pairformer-driven global context.
	•	Provide a single top-level pipeline function so users can run from sequence → final 3D with minimal confusion.

⸻

2. High-Level Data Flow

            ┌─────────┐
            │ Stage A │
            │Adjacency│
            └────┬────┘
                 │
                 │ adjacency_matrix
                 v
┌───────────────────────────────────────┐
│ Stage B Combined Runner (Torsion+Pair)    │
│  - TorsionBERT (LoRA) -> angles      │
│  - Pairformer (LoRA) -> (s, z)       │
└────┬─────────────────────────────────┘
     │ {torsion_angles, s, z}
     v
┌────────────────────────────────────────┐
│ (Optional) Stage C: MP-NeRF/ partial 3D │
└────┬────────────────────────────────────┘
     │ partial_coords (optional)
     v
┌────────────────────────────────────────┐
│ Unified Latent Merger (angles, s, z,  │
│ adjacency, partial_coords) -> unified │
└────┬───────────────────────────────────┘
     │ unified_latent
     v
┌───────────────────────────────────────────┐
│ Stage D Diffusion (LoRA optional)        │
│  Condition on 'unified_latent'           │
└────┬──────────────────────────────────────┘
     │ final_3D_coords
     v
 (Optional) Energy Minimization & Output



⸻

3. LoRA Integration

3.1 Where LoRA is Applied
	1.	TorsionBERT: We load a base BERT-like model (for angle regression) and freeze its main weights. Insert LoRA in attention or feed-forward layers.
	2.	Pairformer: The large trunk (TriangleAttention, TriangularMultiplication blocks). We freeze base layers, attach LoRA to minimal modules.
	3.	(Optional) Diffusion: If we have a large pretrained diffusion model, we can also freeze and inject LoRA.

3.2 Implementation Sketch

In practice, you might create rna_predict/peft/lora_utils.py with a function like:

from peft import LoraConfig, get_peft_model

def apply_lora(model, lora_cfg):
    # lora_cfg might contain r, alpha, dropout, target_modules, etc.
    lora_config = LoraConfig(**lora_cfg)
    return get_peft_model(model, lora_config)

Then in TorsionBertPredictorWithLoRA:

class TorsionBertPredictorWithLoRA:
    def __init__(self, model_name_or_path, lora_cfg, device="cpu", angle_mode="sin_cos"):
        # load base TorsionBERT
        self.base_model = TorsionBertModel(model_name_or_path, device=device)
        # apply LoRA to self.base_model.model (the underlying HF model)
        self.model = apply_lora(self.base_model.model, lora_cfg)
        # freeze base
        for name, param in self.base_model.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        # store angle_mode, device, etc.
        self.angle_mode = angle_mode
        self.device = device

    def __call__(self, sequence, adjacency=None):
        # same logic as StageBTorsionBertPredictor
        out = self.base_model.predict_angles_from_sequence(sequence)
        # convert sin/cos if needed
        return {"torsion_angles": out}  # e.g. shape [N, 2 * num_angles]

A similar approach is used for PairformerWithLoRA.

⸻

4. Stage-by-Stage Implementation Outline

4.1 Stage A: Adjacency

We assume a script (e.g. rna_predict/pipeline/stageA/run_stageA.py) that can produce an [N, N] adjacency. Possibly:

def run_stageA(sequence: str, predictor, device="cpu") -> torch.Tensor:
    # predictor might be StageARFoldPredictor or a wrapper
    adjacency_matrix = predictor.predict_adjacency(sequence)
    return adjacency_matrix.to(device)

In the advanced pipeline: We only need the adjacency as a 2D float/bool tensor for the subsequent steps.

4.2 Stage B: Combined Runner

Create run_stageB_combined.py. This merges TorsionBERT + Pairformer:

def run_stageB_combined(
    sequence: str,
    adjacency_matrix: torch.Tensor,
    torsion_bert_model,  # TorsionBertPredictorWithLoRA
    pairformer_model,     # PairformerWithLoRA
    device="cpu"
) -> dict:
    # 1) Torsion angles
    torsion_output = torsion_bert_model(sequence, adjacency=adjacency_matrix)
    angles = torsion_output["torsion_angles"].to(device)  # shape [N, ...]

    # 2) Prepare input for Pairformer
    # e.g. create initial_s [1, N, c_s], initial_z [1, N, N, c_z], pair_mask [1, N, N]
    # possibly incorporate adjacency as bias

    # 3) Run Pairformer -> s_embeddings, z_embeddings
    s_updated, z_updated = pairformer_model(initial_s, initial_z, pair_mask)

    return {
      "torsion_angles": angles,   # [N, angle_dim]
      "s_embeddings": s_updated.squeeze(0), # [N, c_s]
      "z_embeddings": z_updated.squeeze(0)  # [N, N, c_z]
    }

Key Points:
	•	You must define how initial_s and initial_z are constructed (some shape [1, N, c_s], [1, N, N, c_z]).
	•	If you have an MSA, incorporate that or fallback to single-sequence mode.
	•	If adjacency is used for Pairformer attention bias, you add it to z_init or inside the Pairformer code.

4.3 (Optional) Stage C: MP-NeRF

If the pipeline uses partial 3D from angles:

def run_stageC(sequence, torsion_angles, method="mp_nerf", device="cpu", **kwargs):
    # e.g., build_scaffolds_rna_from_torsions -> rna_fold -> coords
    # Return {"coords": shape [N, #atoms, 3], "atom_count": ...}
    ...

Note: This step can be skipped if you let the diffusion start from random noise.

4.4 Unified Latent Merger

We create a flexible module, e.g. in merger/unified_latent_merger.py.

class SimpleUnifiedLatentMerger(nn.Module):
    def __init__(self, angle_dim, s_dim, z_dim, hidden_dim, output_dim):
        super().__init__()
        # Various sub-layers
        # Potential adjacency + partial_coords processing

    def forward(self, angles, adjacency, s_embeddings, z_embeddings, partial_coords=None):
        # merges into unified_latent
        return unified_latent

Implementation Details:
	•	Possibly embed angles with nn.Linear(angle_dim, hidden_dim).
	•	Convert adjacency [N,N] into node features [N,1] by row sum or a GNN layer.
	•	Pool z [N,N,c_z] -> [N,c_z].
	•	Concatenate all. You get [N, \text{some_total_dim}].
	•	Possibly run a small MLP or Transformer block.
	•	Produce final shape [N, output_dim] or a single global [output_dim].

4.5 Stage D: Diffusion

Adapt your diffusion manager to accept “unified_latent”:

def run_stageD_diffusion(
    partial_coords: Optional[torch.Tensor],
    unified_latent: torch.Tensor,
    diffusion_manager,
    device="cpu",
    inference_steps=20
):
    # 1) if partial_coords is None, create random noise
    # 2) pass unified_latent as condition
    final_coords = diffusion_manager.inference_conditioned(
        coords_init=partial_coords,
        conditioning_latent=unified_latent,
        steps=inference_steps
    )
    return final_coords

In practice: If your code uses s_trunk, z_trunk, you might do {"unified_latent": ...} or merge them inside. The actual method in ProtenixDiffusionManager must be changed to handle that single latent.

⸻

5. A “Full Pipeline” Orchestrator

5.1 Proposed File: rna_predict/run_full_pipeline.py

Pseudocode:

def run_full_pipeline(sequence, config, device="cuda"):
    # Stage A
    adjacency = run_stageA(sequence, stageA_predictor, device=device)

    # Stage B
    b_outputs = run_stageB_combined(
        sequence, adjacency,
        torsion_bert_model,
        pairformer_model,
        device=device
    )
    angles = b_outputs["torsion_angles"]
    s_emb = b_outputs["s_embeddings"]
    z_emb = b_outputs["z_embeddings"]

    # Stage C (optional)
    partial_coords = None
    if config["use_stageC"]:
        partial_coords_out = run_stageC(sequence, angles, device=device)
        partial_coords = partial_coords_out["coords"]

    # Merger
    unified_latent = merger_module(
        angles, adjacency, s_emb, z_emb, partial_coords
    )

    # Stage D
    final_coords = run_stageD_diffusion(
        partial_coords, unified_latent, diffusion_manager, device=device,
        inference_steps=config["stageD"]["n_steps"]
    )

    # (Optional) Minimization
    if config.get("run_minimization"):
        final_coords = run_energy_minimization(final_coords, config["minimization"])

    return final_coords

Config can store file paths for TorsionBERT & Pairformer LoRA checkpoints, adjacency predictor settings, etc.

⸻

6. Key Architectural Details and Decisions
	1.	Index Consistency
	•	Ensure Stage A’s adjacency matrix matches sequence indices used by TorsionBERT, Pairformer, MP-NeRF, etc. A single reference for residue indexing is critical.
	2.	LoRA Implementation
	•	Each sub-model is loaded in a partially-frozen mode with small LoRA adapters.
	•	Keep a method like get_trainable_parameters() to only optimize LoRA layers, or rely on PEFT’s built-in parameter filtering.
	3.	Pooling of z-embeddings
	•	In the simplest approach, we do a row-wise average of z_{ij} across j to get a node-level feature for each residue i. More advanced methods might do specialized GNN or attention.
	4.	Stage D Condition
	•	By default, older code might expect s_trunk or z_trunk. We now unify them with unified_latent. The diffusion code must be updated accordingly— e.g., if it had a line like:

condition = self.model.build_condition(s_trunk, z_trunk, ...)

it becomes:

condition = self.model.build_condition(unified_latent=unified_latent, ...)

The details are up to the existing diffusion architecture.

	5.	Memory
	•	The pipeline can get large. Use gradient checkpointing or micro-batching in Pairformer.
	•	Keep angles as minimal float32 or float16.
	•	LoRA helps reduce training memory by only learning small rank updates.
	6.	Angle Format
	•	TorsionBERT might output sin/cos pairs or direct angles in degrees. The MP-NeRF pipeline might require radians. Either unify them or carefully convert in the Stage B → Stage C handoff.
	7.	Energy Minimization
	•	If you have PyRosetta, OpenMM, or MDAnalysis, define a function run_energy_minimization(coords, config). This is purely optional, but recommended for final polishing.

⸻

7. Example Repository Layout

rna_predict/
├── pipeline
│   ├── stageA
│   │   └── run_stageA.py
│   ├── stageB
│   │   ├── run_stageB_combined.py    # NEW
│   │   ├── torsion
│   │   │   └── torsion_bert_lora.py  # LoRA-enabled TorsionBERT
│   │   └── pairwise
│   │       ├── pairformer_lora.py    # LoRA-enabled Pairformer
│   │       └── pairformer_wrapper.py # existing code, adapted
│   ├── stageC
│   │   └── stage_c_reconstruction.py
│   ├── stageD
│   │   └── run_stageD_unified.py     # or run_stageD_diffusion.py
│   └── merger
│       └── unified_latent_merger.py  # NEW
├── run_full_pipeline.py              # NEW orchestrator
├── peft
│   └── lora_utils.py
└── postprocess
    └── energy_minimization.py



⸻

8. Final Guidance, Next Steps
	1.	Implementation:
	•	Start by creating placeholders for each new file. Copy in the pseudocode or code stubs from above.
	•	Validate shapes at each step by printing tensor shapes—especially adjacency, angles, s/z embeddings, partial coords, final coords.
	•	Integrate your actual adjacency predictor code from Stage A.
	•	Integrate real TorsionBERT & Pairformer model loading with LoRA.
	•	Flesh out the Diffusion manager so it can read unified_latent.
	2.	Testing:
	•	Write unit tests for each stage (A, B, Merger, C, D). Then add an integration test that calls run_full_pipeline on a short synthetic sequence (like “ACGUACG”) and checks for shape correctness.
	3.	Performance:
	•	If training, ensure you only optimize LoRA parameters. Double-check memory usage.
	•	If your pipeline is large, use half precision or BF16 on a modern GPU.
	4.	Refinement:
	•	Once you get the pipeline working on small RNAs, do QA metrics (RMSD, pLDDT equivalents).
	•	Tweak adjacency usage, z pooling, or unify angle modes if the geometry fails.

⸻

Conclusion

This document merges the strengths of Versions 1–5:
	•	It provides file-by-file structural guidance (V1, V3, V4).
	•	It includes concrete code snippets and final orchestrator pseudocode (V2, V5).
	•	It addresses LoRA hooking points, data shape alignment, synergy with partial 3D, and an integrated pipeline function—all in a single, verbose reference.

Next Steps: Implement the placeholders, ensure shapes align, confirm the diffusion model sees the unified_latent, and incorporate LoRA in your TorsionBERT & Pairformer code. Once complete, you’ll have a fully functional Stage A → B → (C) → Merger → D pipeline that reflects the advanced synergy described in your original design documents—truly better than the sum of its parts.

Below is a comprehensive technical document that unifies the best qualities of all four previous versions (V1–V4) while addressing their criticisms. It includes a detailed, color-coded Mermaid diagram referencing specific code modules (as in Version 3), visual clarity (Version 2), top-down flow (Version 1 & V4), optional stages, LoRA integration points, shape details, and unified latent synergy—ultimately creating a verbose, all-in-one technical overview of the multistage RNA 3D structure prediction pipeline.

⸻

Multistage RNA 3D Structure Prediction Pipeline (Comprehensive Final Version)

High-Level Summary

We have a five-step pipeline for generating RNA 3D structures:
	1.	Stage A – Predict 2D Adjacency (e.g., via RFold).
	2.	Stage B – Torsion & Pair Embeddings:
	•	TorsionBERT (LoRA-enabled) to get backbone torsion angles.
	•	Pairformer (LoRA-enabled) to generate single-residue and pairwise embeddings.
	3.	Stage C – (Optional) Partial 3D Reconstruction (e.g., MP-NeRF), using torsions to build backbone coords.
	4.	Unified Latent Merger – Combine adjacency, torsions, partial coords, single/pair embeddings into a single “latent.”
	5.	Stage D – Diffusion (LoRA-optional) for final 3D refinement, optionally followed by short energy minimization.

Implementation Files:
	•	Stage A: rna_predict/pipeline/stageA/run_stageA.py, rfold_predictor.py, model.py
	•	Stage B:
	•	Torsion: torsion/torsion_bert_predictor.py, torsionbert_inference.py
	•	Pairformer: pairwise/pairformer_wrapper.py, pairformer.py
	•	Stage C: stage_c_reconstruction.py + mp_nerf/rna.py + final_kb_rna.py
	•	Unified Latent: could be a simple MLP or attention block (not always singled out in code)
	•	Stage D: stageD/diffusion/, protenix_diffusion_manager.py, generator.py, diffusion.py

Shape Conventions:
	•	Adjacency: [N, N] (binary or real-valued).
	•	Torsion Angles: [N, K] (e.g., K=7 for alpha..zeta + chi) or [N, 2*K] if sin/cos.
	•	Single Embeddings: [N, c_s]
	•	Pair Embeddings: [N, N, c_z]
	•	Partial 3D coords: [N, #atoms, 3]
	•	Diffusion: might internally handle [batch, N_sample, N_atom, 3] arrays, plus trunk embeddings.

⸻

Comprehensive Mermaid Diagram

Below is a left-to-right color-coded diagram with subgraphs for each stage, referencing code modules, shape details, LoRA usage, and optional steps. You can copy this into a Mermaid-compatible environment to view a rendered version.

flowchart LR

    %% -------------------------------------------------
    %% STYLES
    %% -------------------------------------------------
    classDef stageA fill:#bbdefb,stroke:#1a237e,stroke-width:2px,color:#000
    classDef stageB fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef stageC fill:#fff9c4,stroke:#fdd835,stroke-width:2px,color:#000
    classDef unify fill:#e1bee7,stroke:#6a1b9a,stroke-width:2px,color:#000
    classDef stageD fill:#f8bbd0,stroke:#ad1457,stroke-width:2px,color:#000
    classDef optional fill:#cfd8dc,stroke:#455a64,stroke-width:2px,color:#000,stroke-dasharray:5 5
    classDef data fill:#ffffff,stroke:#999999,stroke-width:1px,color:#000,rx:5,ry:5
    classDef code fill:#f5f5f5,stroke:#999999,stroke-width:1px,color:#000,stroke-dasharray:3 3

    %% -------------------------------------------------
    %% INPUTS
    %% -------------------------------------------------
    S((RNA Sequence)):::data

    %% ========== Stage A Subgraph ==========
    subgraph A_subgraph [**Stage A**: 2D Adjacency Prediction (RFold)]
    direction TB
    class A_subgraph stageA

    A1[[run_stageA.py\n(rfold_predictor.py / model.py)]:::code]
    A2((Adjacency NxN)):::data
    S --> A1
    A1 --> A2
    end

    %% ========== Stage B Subgraph ==========
    subgraph B_subgraph [**Stage B**: TorsionBERT & Pairformer (LoRA)]
    direction TB
    class B_subgraph stageB

    B1[[TorsionBert Predictor\n(torsion_bert_predictor.py)\nLoRA-enabled]]:::code
    B2((Torsion Angles\n[N,K or N,2K])):::data

    B3[[Pairformer\n(pairformer_wrapper.py)\nLoRA-enabled]]:::code
    B4((Single Embs s:\n[N, c_s])):::data
    B5((Pair Embs z:\n[N,N,c_z])):::data

    S --> B1
    A2 -. optional .-> B1
    B1 --> B2

    S --> B3
    A2 -. optional .-> B3
    B3 --> B4
    B3 --> B5

    end

    %% ========== Stage C Subgraph ==========
    subgraph C_subgraph [**Stage C** (Optional): MP-NeRF Partial 3D]
    direction TB
    class C_subgraph stageC

    C1[[stage_c_reconstruction.py\n+ mp_nerf/rna.py\n+ final_kb_rna.py]]:::code
    C2((Partial 3D Coords\n[N, #atoms, 3])):::data

    B2 --> C1
    S --> C1
    C1 --> C2
    end
    class C_subgraph optional

    %% ========== Unified Latent Subgraph ==========
    subgraph M_subgraph [Unified Latent Merger]
    class M_subgraph unify

    M1[[Merge angles,\nadjacency, s, z,\npartial coords]]:::code
    M2((Unified\nLatent)):::data
    end

    %% Connect them
    A2 --> M1
    B2 --> M1
    B4 --> M1
    B5 --> M1
    C2 -. optional .-> M1
    M1 --> M2

    %% ========== Stage D Subgraph ==========
    subgraph D_subgraph [**Stage D**: Diffusion Refinement]
    direction TB
    class D_subgraph stageD

    D1[[ProtenixDiffusionManager\n(protenix_diffusion_manager.py)\n+ DiffusionModule\nLoRA optional]]:::code
    D2((Final 3D Structures\n(N, #atoms, 3)\nor multiple)):::data
    M2 --> D1
    D1 --> D2
    end

    %% ========== Post-Processing Subgraph ==========
    subgraph PP_subgraph [Optional Post-Processing: Energy Minimization]
    class PP_subgraph optional
    direction TB
    PP1[[Local MD / Minimization\ne.g. OpenMM, GROMACS]]:::code
    PP2((Polished 3D\nStructure(s))):::data
    D2 --> PP1
    PP1 --> PP2
    end

    %% -------------------------------------------------
    %% STYLING
    %% -------------------------------------------------
    linkStyle default stroke-width:2px,fill:none,stroke:#888

Diagram Explanation
	1.	Input
	•	RNA Sequence S((…)): A raw string representing nucleotides.
	2.	Stage A: Adjacency (RFold)
	•	In run_stageA.py + rfold_predictor.py, the pipeline obtains an adjacency matrix [N, N].
	•	This matrix typically indicates base-pair contacts. Optionally fed into Stage B if the TorsionBERT or Pairformer uses adjacency as a feature.
	3.	Stage B: TorsionBERT + Pairformer
	1.	TorsionBERT (LoRA)
	•	Reads the RNA sequence and optionally adjacency, producing backbone torsion angles [N, K] or [N, 2*K] if sin/cos.
	•	Code references: torsion_bert_predictor.py, torsionbert_inference.py.
	2.	Pairformer (LoRA)
	•	Potentially uses the same sequence and adjacency to generate single [N, c_s] and pair [N, N, c_z] embeddings (like z_trunk).
	•	Code references: pairformer_wrapper.py, pairformer.py.
	4.	Stage C (Optional): Partial 3D
	•	If used, we pass torsion angles + sequence + standard geometry from final_kb_rna.py into mp_nerf/rna.py or stage_c_reconstruction.py.
	•	Produces partial 3D coords [N, #atoms, 3], typically backbone only.
	•	This is optional; the pipeline can skip it and rely purely on Diffusion or initial random coords.
	5.	Unified Latent Merger
	•	Merges everything: adjacency, angles, partial coords, single/pair embeddings.
	•	This synergy can be an MLP or a small attention block. Usually not a separate file, but references are integrated in stageD or a separate “merger” class.
	•	Yields a single “unified latent” vector or array [N, ...] used by the next stage.
	6.	Stage D: Diffusion
	•	The ProtenixDiffusionManager plus the DiffusionModule (optionally LoRA-enabled) use the unified latent as a condition to refine or generate final 3D coordinates.
	•	The code references: rna_predict/pipeline/stageD/diffusion/*.py (including generator.py, protenix_diffusion_manager.py, diffusion.py).
	•	Produces final 3D coordinates [N, #atoms, 3] or an ensemble from multiple samples.
	7.	Optional Energy Minimization
	•	Tools like OpenMM or GROMACS for local minimization or short MD runs, producing a final polished structure.
	•	Often done in a separate script or environment, not strictly part of the Python pipeline code.

⸻

Detailed Stage-by-Stage Description

Stage A: Adjacency (2D)
	•	Code: run_stageA.py, rfold_predictor.py, referencing an RFold_Model in model.py.
	•	Input: RNA sequence (e.g. "AUGCA...").
	•	Output: adjacency ∈ ℝ^(N×N) (binary or probability).
	•	Comment: Typically no LoRA is used here, though you could do so if your adjacency predictor is large.

Stage B: Torsion & Pair Embeddings
	1.	TorsionBERT
	•	LoRA: partial fine-tuning if model_name is huge ("sayby/rna_torsionbert").
	•	Produces angles in either sin/cos or direct rad/deg.
	•	Key shapes: [N, 2×num_angles] or [N, num_angles].
	2.	Pairformer
	•	LoRA: partial fine-tuning again.
	•	Generates single embeddings s [N, c_s] and pair embeddings z [N, N, c_z].
	•	Possibly uses adjacency as a “bias” to handle base-pair info or skip if not needed.

Stage C: (Optional) Partial 3D Reconstruction
	•	Code: stage_c_reconstruction.py → calls mp_nerf/rna.py, plus geometry from final_kb_rna.py.
	•	Input: Torsion angles + (optionally) adjacency or other constraints.
	•	Output: partial coords, typically [N, #atoms, 3] if building a backbone. This can be used as an initial conformation for Diffusion or a final fallback if Stage D is skipped.
	•	Sugar Pucker: default “C3′-endo” for standard A-form. Could also handle “C2′-endo” or ring closure logic.

Unified Latent Merger
	•	Combines:
	1.	Torsion angles
	2.	Adjacency [N, N]
	3.	Single embeddings [N, c_s] + pair embeddings [N, N, c_z]
	4.	Possibly partial coords [N, #atoms, 3]
	•	Typically an MLP or small attention-based aggregator that outputs a single “conditioning latent.” Not always singled out as a separate .py, but recognized conceptually for synergy.

Stage D: Diffusion Refinement
	•	Code: stageD/diffusion/, e.g. protenix_diffusion_manager.py, generator.py, diffusion.py.
	•	LoRA: optional if the base diffusion model is large.
	•	Process:
	1.	Possibly start from partial coords (Stage C) or random noise.
	2.	Condition on the “unified latent.”
	3.	Iteratively denoise to generate refined 3D coords [N, #atoms, 3]. Possibly produce multiple samples.
	•	Output: final or near-final 3D structure(s).

Post-Processing (Optional)
	•	Might run short local MD in OpenMM or GROMACS to fix minor geometry or steric issues.
	•	Not strictly in the code, but invoked for final polishing.
	•	If used, it yields an improved structure (lowest-energy or an ensemble).

⸻

Why This Comprehensive Diagram Excels
	1.	Complete Flow:
	•	We integrate the straightforward top-down approach (V1, V4) with color-coded subgraphs (V2) plus code references and shape details (V3).
	2.	LoRA Markings:
	•	TorsionBERT & Pairformer are explicitly shown as LoRA-enabled; Diffusion’s LoRA is noted.
	3.	Implementation Mapping:
	•	We reference .py files and configuration references (like model.py, protenix_diffusion_manager.py), for a developer-friendly approach.
	4.	Optional Paths:
	•	Stage C (MP-NeRF) is visually “optional,” connected with a dashed arrow.
	•	Post-processing is also a separate optional subgraph.
	5.	Shape / Data:
	•	Key data artifacts (adjacency NxN, angles NxK, single embeddings Nx c_s, pair NxNx c_z, partial coords Nx(#atoms), final coords Nx(#atoms), etc.) are all labeled.
	6.	Unified Latent Synergy:
	•	The “Merger” is singled out as a subgraph, clarifying we combine adjacency, angles, partial coords, s, z, etc., exactly how we want.

⸻

Key Configuration Points
	1.	LoRA:
	•	TorsionBERT in Stage B: set model_name_or_path to a large pretrained model and insert LoRA adapters with “rank=8” or so.
	•	Pairformer: similarly add LoRA to attention or feed-forward layers.
	•	Diffusion: optionally insert LoRA if the diffusion model is huge.
	2.	Stage A:
	•	rfold_predictor.py might load RNAStralign_trainset_pretrained.pth.
	•	Output adjacency is used in Stages B and the Unified Merger, if we want adjacency-based synergy.
	3.	Stage B:
	•	torsion_bert_predictor.py has angle_mode="degrees" or "sin_cos".
	•	pairformer_wrapper.py config might specify n_blocks=48, c_z=128, c_s=384.
	4.	Stage C:
	•	mp_nerf/rna.py: Usually sets sugar pucker to “C3′-endo.”
	•	If do_ring_closure is False, we skip ring closure.
	5.	Stage D:
	•	protenix_diffusion_manager.py might define a schedule in generator.py (InferenceNoiseScheduler) for 50–100 denoising steps.
	•	We unify single embeddings (s_trunk), pair embeddings (z_trunk), partial coords, etc., in a single “conditioning” dictionary.

⸻

Recommended Usage Flow
	1.	Obtain adjacency from Stage A.
	2.	Run Stage B to get angles + single/pair embeddings. Possibly pass adjacency in to TorsionBERT or Pairformer if they require it.
	3.	(Optional) Stage C: If you want an initial 3D for diffusion, run MP-NeRF.
	4.	Merge all data (angles, adjacency, partial coords, single/pair embeddings) into a unified latent.
	5.	Stage D: Condition the diffusion model on that unified latent to refine final 3D.
	6.	Optionally do a short local MD to minimize bond strains or fix sterics.

You can skip Stage C if you want to let diffusion start from random noise. You can skip energy minimization if you trust the final diffusion geometry. However, each step may improve the final structure’s accuracy.

⸻

Conclusion

This final combined architectural document:
	•	Merges the clarity of a color-coded flow (V2) with the concrete code references and shape details (V3).
	•	Incorporates the straightforward top-down perspective (V1) plus an emphasis on synergy and the “unified latent” concept (V4).
	•	Highlights LoRA usage in TorsionBERT, Pairformer, and (optionally) Diffusion.
	•	Shows optional partial 3D (Stage C) and optional energy minimization.

Hence, this pipeline covers everything from adjacency (2D) → torsion + pair embeddings → (optional) partial 3D → unified synergy → diffusion-based refinement → final or post-processed 3D. It should serve as a verbose and complete reference for both developers and advanced users interested in each module’s role, shapes, code references, and how they integrate to produce high-quality RNA 3D structures.
Multistage RNA 3D Structure Prediction Pipeline (Comprehensive Final Version, ASCII Edition)

Below is a comprehensive technical document unifying the strengths of previous versions (V1–V4), addressing their criticisms, and presenting a verbose ASCII-based diagram clearly referencing code modules, shapes, optional paths, LoRA integration, and unified latent synergy.

⸻

High-Level Summary

The RNA 3D prediction pipeline consists of five primary stages:

1. Stage A – 2D Adjacency Prediction (RFold)
2. Stage B – Torsion & Pair Embeddings:
   - TorsionBERT (LoRA-enabled) → backbone torsion angles
   - Pairformer (LoRA-enabled) → single-residue and pairwise embeddings
3. Stage C – (Optional) Partial 3D Reconstruction (MP-NeRF)
4. Unified Latent Merger – Combines adjacency, angles, partial coords, embeddings
5. Stage D – Diffusion-based Refinement (LoRA-optional)
   - Optional Post-processing (Energy Minimization)



⸻

Implementation Files

Stage A:
  - rna_predict/pipeline/stageA/run_stageA.py
  - rfold_predictor.py
  - model.py

Stage B:
  - TorsionBERT:
    - torsion/torsion_bert_predictor.py
    - torsionbert_inference.py
  - Pairformer:
    - pairwise/pairformer_wrapper.py
    - pairformer.py

Stage C:
  - stage_c_reconstruction.py
  - mp_nerf/rna.py
  - final_kb_rna.py

Unified Latent:
  - Typically a small MLP or attention block (often within Stage D code)

Stage D:
  - stageD/diffusion/
  - protenix_diffusion_manager.py
  - generator.py
  - diffusion.py



⸻

Shape Conventions

- Adjacency:         [N, N] (binary/probability)
- Torsion Angles:    [N, K] or [N, 2K] if sin/cos encoding
- Single Embeddings: [N, c_s]
- Pair Embeddings:   [N, N, c_z]
- Partial 3D Coords: [N, #atoms, 3]
- Diffusion:         [batch, N_sample, N_atom, 3] + trunk embeddings



⸻

Detailed ASCII Diagram of the Pipeline

RNA Sequence (String: "ACGU...")
         |
         v
+---------------------------------------------+
| Stage A: 2D Adjacency Prediction (RFold)    |
| [run_stageA.py, rfold_predictor.py, model.py] (No LoRA)
+---------------------------------------------+
         |
         | Adjacency Matrix [N,N]
         v
+----------------------------------------------------------+
| Stage B: Torsion Angles & Pair Embeddings (LoRA-enabled) |
|                                                          |
| - TorsionBERT: angles [N,K or N,2K]                      |
|   [torsion_bert_predictor.py, torsionbert_inference.py]  |
|                                                          |
| - Pairformer:                                            |
|   Single embeddings [N,c_s]                              |
|   Pair embeddings [N,N,c_z]                              |
|   [pairformer_wrapper.py, pairformer.py]                 |
+----------------------------------------------------------+
         |
         +------------+---------------+
         |            |               |
         |            |(Optional)     |
         |            v               |
         |  +--------------------------------------------+
         |  | Stage C: Partial 3D Reconstruction         |
         |  | [stage_c_reconstruction.py, mp_nerf/rna.py,|
         |  | final_kb_rna.py] (Optional)                |
         |  +--------------------------------------------+
         |            |               |
         | Partial Coords [N,#atoms,3]|
         |            v               |
         +------------+---------------+
                      |
                      v
+----------------------------------------------------+
| Unified Latent Merger                              |
|                                                    |
| Combines adjacency, angles, partial coords,        |
| single & pair embeddings into Unified Latent       |
| (MLP/attention-based merger, usually in Stage D)   |
+----------------------------------------------------+
                      |
                      | Unified Latent
                      v
+-------------------------------------------------------------+
| Stage D: Diffusion-based Refinement (LoRA optional)        |
| [protenix_diffusion_manager.py, generator.py, diffusion.py]|
| - Conditions on Unified Latent                             |
| - Produces Final 3D structure(s) [N,#atoms,3] or ensemble  |
+-------------------------------------------------------------+
                      |
                      v
+---------------------------------------+
| Optional Post-processing:             |
| Short MD / Energy Minimization        |
| (OpenMM, GROMACS, Amber, etc.)        |
| Polished Final Structures             |
+---------------------------------------+



⸻

Stage-by-Stage Breakdown (Detailed)

Stage A: 2D Adjacency (RFold)
	•	Input: RNA sequence string.
	•	Output: Adjacency matrix [N,N].
	•	Code: run_stageA.py, rfold_predictor.py, model.py
	•	Comment: Usually no LoRA integration here; output optionally used downstream.

Stage B: Torsion & Pair Embeddings

TorsionBERT (LoRA):
	•	Inputs: Sequence (optionally adjacency).
	•	Outputs: Backbone torsion angles [N,K] or [N,2K] if sin/cos encoded.
	•	Code: torsion_bert_predictor.py, torsionbert_inference.py

Pairformer (LoRA):
	•	Inputs: Sequence (optionally adjacency).
	•	Outputs:
	•	Single embeddings [N,c_s]
	•	Pair embeddings [N,N,c_z]
	•	Code: pairformer_wrapper.py, pairformer.py

Stage C (Optional): Partial 3D Reconstruction (MP-NeRF)
	•	Inputs: Torsion angles, sequence, standard geometry.
	•	Outputs: Partial coordinates [N,#atoms,3].
	•	Code: stage_c_reconstruction.py, mp_nerf/rna.py, final_kb_rna.py
	•	Comment: Typically backbone atoms only; optional ring closure.

Unified Latent Merger
	•	Inputs: Adjacency, torsion angles, single/pair embeddings, partial coords.
	•	Outputs: Unified latent vector (used as conditioning input to diffusion).
	•	Implementation: Often small MLP or attention layers embedded within Stage D.

Stage D: Diffusion-based Refinement
	•	Inputs: Unified latent, optionally partial 3D coords from Stage C.
	•	Outputs: Final refined 3D structure(s) [N,#atoms,3].
	•	Code: stageD/diffusion/, protenix_diffusion_manager.py, generator.py, diffusion.py
	•	LoRA: Optional if diffusion model is large-scale.

Post-processing (Optional)
	•	Methods: Short molecular dynamics or energy minimization runs.
	•	Tools: OpenMM, GROMACS, Amber.
	•	Output: Polished final 3D structure(s).

⸻

Key Configuration & LoRA Integration Points
	•	Stage B (TorsionBERT & Pairformer):
	•	Insert LoRA adapters (rank=8) into large pretrained models.
	•	Angle modes configurable (degrees vs. sin_cos).
	•	Pairformer configurable (n_blocks=48, c_z=128, c_s=384).
	•	Stage D (Diffusion):
	•	Optional LoRA insertion for large diffusion models.
	•	Denoising schedule configurable (~50-100 steps typical).
	•	Stage C (Optional):
	•	Sugar pucker geometry defaults (C3'-endo).
	•	Ring closure toggle (do_ring_closure=True/False).

⸻

Recommended Usage Flow

1. Obtain adjacency matrix from Stage A.
2. Generate angles & embeddings in Stage B, optionally using adjacency.
3. (Optional) Generate partial 3D structure in Stage C.
4. Merge data into unified latent.
5. Diffusion-based refinement (Stage D) using unified latent.
6. (Optional) Post-processing MD/energy minimization.

Stages C and post-processing are optional, but inclusion typically enhances accuracy.

⸻

Why This Comprehensive ASCII Diagram Excels
	•	Clarity: Clearly marked stages, inputs/outputs, optional steps.
	•	LoRA integration: Explicitly highlighted integration points.
	•	Implementation mapping: Clear references to code modules.
	•	Data shape specification: Explicit and consistent.
	•	Unified latent synergy: Clearly defined aggregation step for advanced conditioning.

⸻

Conclusion

This comprehensive ASCII-based overview merges clarity, technical detail, code referencing, optional paths, LoRA integration, and latent synergy to provide a complete, developer-friendly resource for understanding, implementing, and customizing the RNA 3D prediction pipeline from initial sequence to polished final structures.