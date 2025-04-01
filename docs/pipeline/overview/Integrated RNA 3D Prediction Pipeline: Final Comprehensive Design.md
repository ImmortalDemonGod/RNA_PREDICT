Below is a comprehensive, “best-of-all-worlds” architectural design document that consolidates the strengths of earlier versions (V1, V2, V3, V4), addresses their criticisms, and clarifies optional vs. required steps to ensure synergy between (1) a torsion-based pipeline, (2) an AlphaFold 3–style pairwise trunk, and (3) a final Diffusion module for 3D structure generation. This design is meant to serve as a robust piece of technical documentation—verbose and detailed enough to guide implementation.

⸻

Integrated RNA 3D Prediction Pipeline: Final Comprehensive Design

1. High-Level Goal

Objective: Accurately predict RNA 3D coordinates by unifying:
	1.	A torsion-based pipeline (stages for 2D adjacency → torsion angles → optionally forward kinematics).
	2.	An AlphaFold 3–style pairwise trunk (MSA-based or single-sequence-based Pairformer with triangular updates, pair embeddings).
	3.	A unified latent that merges local geometry (torsion + adjacency) with global pairwise constraints.
	4.	A Diffusion model that conditions on that unified latent to iteratively refine or generate final 3D coordinates.
	5.	A short Energy Minimization step (plus multi-sample approach) to yield a final ensemble and choose the best structure(s).

Key Emphasis
	•	Preventing “too many optional pieces” that undermine synergy.
	•	Ensuring adjacency is used effectively in both torsion and pairwise modules.
	•	Aligning residue indexing so the staged pipeline’s angles match the Pairformer’s pair embeddings.
	•	Using a single final generator (Diffusion) that sees both local angle constraints and global pair embeddings, delivering more accurate final 3D structures.

⸻

2. Detailed Pipeline Diagram

Below is a textual flow with recommended mandatory vs. optional steps clearly noted. Boxes represent major modules; arrows indicate data/feature flow.

                       ┌──────────────────────────────────────────────────────────┐
                       │ (1) INPUTS & INITIAL SETUP                              │
                       │  • RNA sequence [REQUIRED]                              │
                       │  • 2D adjacency from Stage A [HIGHLY RECOMMENDED]       │
                       │  • MSA data (for Pairformer) [IF AVAILABLE]             │
                       │  • Possibly external templates or partial 3D            │
                       └──────────────────────────────────────────────────────────┘
                                           │
                                           v
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ (2) TORSION-BASED SUBPIPELINE (Stages A/B)                                   │
 │------------------------------------------------------------------------------│
 │   a) Use adjacency + sequence to predict backbone torsion angles:            │
 │      α, β, γ, δ, ε, ζ, χ, … plus adjacency-based features.                   │
 │   b) Potentially do an MLP or GNN that merges adjacency signals.             │
 │   c) Output: "Torsion Representation" => angles for each residue,            │
 │      adjacency features (like base-pair partner indices).                    │
 │   ────────────────────────────────────────────────────────────────────────────│
 │   (Optional) Stage C: Forward Kinematics                                     │
 │      If used, produce partial 3D coords from those angles.                   │
 │      Align indexing with rest of pipeline.                                   │
 │------------------------------------------------------------------------------│
 │  Output => Torsion-based representation (angles, adjacency, partial coords)  │
 └────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ (3) ALPHAFOLD 3–STYLE PAIRFORMER (MSA → Pair embeddings → Triangular Updates)│
 │------------------------------------------------------------------------------│
 │   a) Optionally embed an MSA. Single-sequence possible if MSA is unavailable.│
 │   b) Pass embeddings through ~48-block Pairformer trunk (like AF3)           │
 │      - Triangular multiplication, attention, pair-bias.                      │
 │   c) Possibly incorporate adjacency as a bias or input to pair embeddings.   │
 │   d) Output: pair embeddings zᵢⱼ + single embeddings sᵢ for each residue.    │
 │------------------------------------------------------------------------------│
 │  Output => "Pairwise Representation" from final trunk pass.                  │
 └────────────────────────────────────────────────────────────────────────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          v                                 v
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ (4) UNIFIED LATENT MERGER / COMPRESSION                                      │
 │------------------------------------------------------------------------------│
 │   Merge:                                                                     │
 │   1) Torsion pipeline output (angles, adjacency data, optional partial 3D).  │
 │   2) Pairwise trunk output (zᵢⱼ, sᵢ).                                        │
 │   Possibly a small Transformer or MLP that aligns residue indices,           │
 │   creating a single “latent” that captures local + global constraints.       │
 │------------------------------------------------------------------------------│
 │  Output => "Compressed Latent" for the Diffusion.                            │
 └────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ (5) DIFFUSION MODULE                                                         │
 │------------------------------------------------------------------------------│
 │   a) Initialize random/noised 3D coords for each residue (heavy atoms).      │
 │      Or optionally start from partial coords from Stage C.                   │
 │   b) Condition on the “Compressed Latent” to guide iterative denoising.      │
 │   c) Generate final 3D coordinates after X diffusion steps.                   │
 │------------------------------------------------------------------------------│
 │  Output => multiple 3D structure samples (e.g., 5 or 10).                    │
 └────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ (6) ENERGY MINIMIZATION & ENSEMBLE SELECTION                                 │
 │------------------------------------------------------------------------------│
 │   a) Short local minimization (e.g., Amber/CHARMM) to fix small geometry.    │
 │   b) Evaluate & rank each sample by geometry score or internal confidence.   │
 │   c) Return top N (like 5) final structures, or a single best structure.      │
 │------------------------------------------------------------------------------│
 │  Output => Final 3D ensemble or single best structure.                       │
 └────────────────────────────────────────────────────────────────────────────────┘



⸻

3. Addressing Previous Criticisms
	1.	Undermining synergy by making everything “optional.”
	•	Here, torsion angles (Stage B) and pair embeddings (AF3 trunk) are both mandatory for synergy.
	•	Adjacency is strongly recommended (it’s the entire reason the torsion pipeline works effectively).
	•	Forward Kinematics (Stage C) is labeled optional but we provide a rationale for skipping or using it.
	•	The final Diffusion cannot skip either local or global constraints, because they are merged at step 4 by design.
	2.	Using adjacency only in the torsion pipeline
	•	We now highlight that adjacency can also feed into the Pairformer trunk as a pair-bias in attention.
	•	This ensures adjacency is not underused or stuck in a corner; it can influence both local angle modeling and the global pairwise network.
	3.	Residue indexing mismatch
	•	We explicitly define a single consistent indexing scheme that all pipeline stages must share.
	•	If the torsion pipeline re-maps or discards residues, we do a bridging “residue index alignment” prior to the “Unified Latent Merger.”
	4.	Weak merging of torsion + pair embeddings
	•	Previously, we said “small MLP.” Now we specify that a “Latent Merger” might be a minimal Transformer or GNN that can properly unify node-level angles with pair-level embeddings zᵢⱼ.
	•	This is a richer approach, preserving structure. Or simpler solutions are possible, but we highlight the need to handle (i, j) pairs carefully.
	5.	Energy Minimization
	•	We reaffirm that short local minimization is strongly advised for final geometry polishing, especially in a multi-sample scenario.
	•	This step addresses lingering steric or bond-angle issues not fully solved by the neural pipeline.

⸻

4. Mandatory vs. Optional Steps

To avoid confusion about synergy:
	•	Mandatory:
	1.	Stage B Torsion: adjacency + sequence → angles.
	2.	Pairformer trunk: MSA or single-sequence → pair embeddings.
	3.	Unified Latent (merger) so the Diffusion sees both.
	4.	Diffusion as the final generator.
	•	Strongly Recommended:
	•	Stage A adjacency: Typically required if you want a torsion pipeline.
	•	Energy Minimization at the end.
	•	Truly Optional:
	1.	Stage C forward kinematics: If you prefer letting Diffusion handle initial coords from random noise, you can skip. But giving it a partial 3D “warm start” can help.
	2.	MSA: If you lack multiple sequences, the Pairformer can run single-sequence mode, though results may degrade.
	3.	Templates: Could be integrated but not mandatory.

Thus, the pipeline always merges local angles and pair embeddings for synergy. Adjacency is recommended so the torsion pipeline has meaningful constraints.

⸻

5. Explanation of Each Module

(A) Torsion-Based Pipeline (Stage B)
	1.	Input:
	•	RNA sequence of length N.
	•	Adjacency/2D structure from Stage A (each residue i has a potential base-pair partner j).
	2.	Angle Prediction:
	•	A GNN or MLP that sees adjacency and predicts \alpha, \beta, \gamma, \delta, \epsilon, \zeta, \chi for each residue i. Possibly sugar pucker angles if needed.
	3.	Output:
	•	An angle vector per residue; adjacency-based features (like “which j is i paired with?”).
	•	(Optional) partial coordinates via forward kinematics if Stage C is invoked.

(B) AlphaFold 3–Style Pairformer
	1.	MSA / Single Sequence:
	•	Construct initial single representation from an MSA embedding or a single-sequence embedding if MSA is unavailable.
	2.	Pairformer Trunk:
	•	Triangular multiplication & triangular attention to refine a pair representation \mathbf{z}_{ij} and single representation \mathbf{s}_i.
	•	Possibly incorporate adjacency in the pair-bias or in the initial pair embedding to nudge the trunk about known base pairs.
	3.	Output:
	•	Final pair embeddings z_{ij} for all pairs i,j, plus single embeddings s_i.

(C) Unified Latent Merger
	1.	Combining:
	•	Residue-level data from the torsion pipeline (angles, adjacency info, partial coords).
	•	Pair-level data from the Pairformer trunk (zᵢⱼ, plus single sᵢ).
	2.	Technique:
	•	A small Transformer or GNN can unify node-level (angles, single sᵢ) with edge-level (zᵢⱼ, adjacency). Or a simpler MLP if resource-limited.
	•	Ensure residue indexing matches between both modules (especially if partial coords skip or reorder some residues).
	3.	Output:
	•	A single “latent representation” fed to the diffusion model for conditioning.

(D) Diffusion Module
	1.	Input: random or partially noised 3D coordinates for each residue’s heavy atoms.
	2.	Conditioning: the “compressed latent” from step (C).
	3.	Process: iterative denoising (like standard 2D/3D diffusion). Each step sees the latent, adjusting coordinates accordingly.
	4.	Output: final 3D coordinates after X steps. Because it’s generative, we can sample multiple times (multiple seeds).

(E) Energy Minimization + Ensemble
	1.	Sampling:
	•	We produce ~5–10 final 3D samples from the diffusion to cover multiple solutions.
	2.	Short Minimization:
	•	For each sample, do a short local minimization (1–10k steps) with an RNA-friendly force field. This corrects bond angles/lengths or steric clashes.
	3.	Scoring & Ranking:
	•	Possibly adapt a pLDDT-like network, or do geometry checks. We select top N structures (like top 5).
	4.	Final:
	•	Provide the best structure for a single guess, or an ensemble of top solutions if the application (e.g., Kaggle) allows multiple submissions.

⸻

6. Potential Implementation Details

Residue Index Alignment
	•	Mapping: We keep a dictionary or table, “ResidueIndexMap,” that ensures if the torsion pipeline discards residues or re-labeled them, the Pairformer still references the same i, j.
	•	Practical: The adjacency is typically a matrix [N×N]; the Pairformer is also [N×N]. They must have identical dimension N, consistent ordering.

Adjacency Integration in Pairformer
	•	Option 1: Modify pair embedding init: z_init[i,j] += Linear(adjacency[i,j]).
	•	Option 2: Add a logit bias: attention_logits(i,j) += w * adjacency(i,j).
	•	Either ensures the pair trunk is aware of known base pairs.

Forward Kinematics (Stage C)
	•	If used, we do a standard NeRF or MP-NeRF approach to place atoms by the predicted torsion angles. This yields partial 3D we either feed to the diffusion as an initialization or as an extra conditioning channel.
	•	Potential advantage: Diffusion starts from a not-too-random conformation, possibly speeding convergence.

Diffusion Model
	•	Implementation: Could be e.g. a score-based generative model or discrete time-step diffusion.
	•	Condition: We pass in “unified latent” each step. The network learns to correct or “denoise” coordinates in alignment with both local angles and global pair constraints.
	•	Training: We’d need training data of known 3D structures plus adjacency or MSA (where available) to supervise the diffusion.

Ensemble & Minimization
	•	Often done in a separate script:
	1.	Run each predicted structure in a local MD environment.
	2.	Evaluate geometry.
	3.	Keep best.
	•	For large RNAs, you might reduce the sample size or do partial minimization.

⸻

7. Advantages Over Previous “Versioned” Designs
	1.	No “lost synergy”: We do not allow the torsion pipeline or the Pairformer to be fully bypassed. Both feed the final Diffusion, ensuring we incorporate adjacency and MSA-like global constraints.
	2.	Clarity on optional: Stage C is optional for a well-understood reason (some may prefer random initialization in the diffusion if partial coords are too inaccurate or if computation time is short).
	3.	Improved Merging: We no longer say “just a small MLP.” We highlight a purposeful “Latent Merger” that can handle node-edge data properly. This solves the prior critique of “weak merging.”
	4.	Residue alignment: Addressed explicitly with a recommendation to keep a consistent indexing or bridging step.
	5.	Energy Minimization: Elevated to recommended status, explaining how it polishes final geometry in a multi-sample scenario.

⸻

8. Implementation Caveats
	•	Complex Development: This pipeline is non-trivial—four major modules (Torsion, AF3 trunk, Merger, Diffusion) plus optional forward kinematics and a final minimization script.
	•	Performance: A full Pairformer (~48 blocks) + GNN or MLP for torsions + a big diffusion network can be memory-heavy. Minimization for 5–10 structures also costs some CPU/GPU time.
	•	Data Gaps: If you lack good adjacency or an MSA, performance could degrade. Single-sequence Pairformer plus no adjacency is effectively a partial pipeline.
	•	Indexing: Potentially the biggest source of bugs. Must ensure consistent labeling from start to finish.

⸻

9. Example Implementation Roadmap
	1.	Data Preprocessing
	•	Gather RNA sequence(s).
	•	Predict or obtain adjacency (2D structure) from a standard method (Stage A).
	•	If available, compile an MSA.
	•	Create a “ResidueIndexMap” to unify indexing across pipeline steps.
	2.	Torsion Pipeline
	•	Use adjacency + sequence → predict angles.
	•	(Optional) run forward kinematics → partial 3D.
	•	Store angles, adjacency-based features, partial coords if used.
	3.	AF3 Pairformer
	•	Load MSA or single sequence.
	•	Run ~48-block trunk.
	•	Possibly incorporate adjacency as a pair-bias.
	•	Output final zᵢⱼ, sᵢ.
	4.	Unified Latent Merger
	•	For each residue i, gather angles, adjacency info, partial coords, single embed sᵢ.
	•	For each pair (i,j), gather zᵢⱼ, adjacency bits.
	•	Construct a single “latent graph” or “multi-dimensional array” the diffusion can read.
	5.	Diffusion
	•	Condition on that latent.
	•	Start from random/noisy coords or from the partial 3D in step 2.
	•	Iteratively generate final coords. Possibly produce multiple samples.
	6.	Energy Minimization & Ranking
	•	For each sample, run short local minimization.
	•	Evaluate geometry, pick top structures.
	7.	Output
	•	Store or submit the best structure(s). Possibly keep an ensemble.

By following this plan, you ensure the final design is robust, synergy is retained, adjacency is used effectively, and each step is well-defined in terms of “mandatory vs. optional.”

⸻

10. Concluding Remarks

This “best-of-all-worlds” pipeline:
	1.	Merges the local knowledge (torsion angles + adjacency from Stage A/B) and the global perspective (AlphaFold’s pair embeddings) in a single final generator (Diffusion).
	2.	Maintains synergy by systematically requiring both the torsion pipeline and the pair trunk to feed into a single “unified latent” stage.
	3.	Leverages an optional forward kinematics step (Stage C) only if beneficial.
	4.	Recommends energy minimization to refine final coordinates from the diffusion model, especially beneficial in an ensemble context (e.g., picking the best 1–5 out of multiple predictions).
	5.	Addresses earlier criticisms about optional synergy, adjacency usage, residue alignment, and shallow merging.

Hence, you get a holistic RNA 3D prediction system that can handle small to moderately large RNAs, incorporate base-pair constraints, exploit MSA-driven pair embeddings, and finalize coordinates through a powerful diffusion framework—ultimately yielding more consistent and accurate 3D structures than the sum of the earlier partial designs.

====
Below is a high-level architectural plan detailing how backpropagation flows through this entire end-to-end RNA 3D prediction system—integrating (1) TorsionBERT (or analogous angle predictor), (2) RFold for 2D adjacency, (3) an AlphaFold 3–style Pairformer trunk, (4) a “unified latent merger”, (5) MP-NeRF or forward kinematics for partial 3D (optional), (6) a Diffusion module for final coordinate generation, and (7) an energy-minimization or short MD pass. We also address how to apply LoRA (Low-Rank Adapters) or QLoRA techniques to adaptively train subsets of pre-initialized weights without exploding GPU memory.

⸻

1. Overall Model Flow & Backprop Considerations

A. Forward Pass Summary
	1.	Stage A (Adjacency, if not provided externally):
	•	If adjacency is predicted by something like RFold or another 2D method, we can treat that as either a frozen or partially trainable module. Usually, adjacency is not strongly backpropagated from final 3D coordinates because it’s more of a discrete 2D structure.
	•	However, if we want adjacency differentiability, we’d need a differentiable base-pair “soft assignment” approach. Typically, we freeze adjacency or treat it as an input.
	2.	TorsionBERT (Stage B):
	•	Takes the RNA sequence (and possibly adjacency features as input).
	•	Produces predicted torsion angles \alpha, \beta, \gamma, \delta, \epsilon, \zeta, \chi (plus sugar pucker if desired).
	•	LoRA Application: Because TorsionBERT is large and partially pre-trained, we can freeze the base BERT-like layers and insert LoRA adapters on top. This ensures a small rank update for angles.
	3.	Pairformer Trunk (AlphaFold 3–style):
	•	Takes an MSA or single sequence, plus possibly adjacency/2D constraints as “pair-bias.”
	•	Outputs final pair embeddings z_{ij} and single embeddings s_i.
	•	LoRA Application: Similarly, we can place LoRA adapters in the Pairformer’s attention layers. We typically freeze the main trunk weights from a pre-trained model and only train the low-rank updates.
	4.	Unified Latent Merger:
	•	Combines TorsionBERT angles + adjacency-based features with the Pairformer embeddings (z_{ij}, s_i). Possibly done via a small merger subnetwork or autoencoder.
	•	LoRA Application: The merger is typically new code; if it’s large, we can apply LoRA. But it might be a small MLP/Transformer, so we can fully train it from scratch if it’s not too big.
	5.	Optional Forward Kinematics (MP-NeRF):
	•	If we feed partial 3D coords into the Diffusion model, we do a differentiable forward pass from torsion angles → partial Cartesian.
	•	Backprop: MP-NeRF is fully differentiable with respect to torsion angles, so gradients flow from final 3D error signals back into TorsionBERT angles.
	•	LoRA Application: Typically none, as MP-NeRF is mostly geometry code, but if it’s large (rarely is), we could also do minimal parameterization if needed.
	6.	Diffusion Module:
	•	Input: random/noised coordinates or partial coords from MP-NeRF, plus the “unified latent.”
	•	Iteratively denoises to final 3D.
	•	LoRA Application: If we use a big diffusion U-Net or Transformer, we can freeze the backbone and add LoRA adapters in its attention layers or feed-forward blocks.
	7.	Energy Minimization (Post-hoc):
	•	Typically not differentiable with respect to earlier modules. This step is outside the main gradient flow. We only do local minimization for final “polish.”

Thus: The main backprop path is:

Final 3D coordinate predictions → compute losses → backprop → (Diffusion model) → (Unified Latent Merger) → (Pairformer trunk’s LoRA, TorsionBERT’s LoRA) → adjacency is likely frozen or partially updated if we adopt a “soft adjacency” approach.

⸻

B. Loss Functions

We’ll likely have two primary supervised loss signals:
	1.	3D Coordinate Loss \mathcal{L}_{3D}:
	•	Compare final predicted 3D coords \mathbf{X}{pred} (after the Diffusion stage) to known ground truth \mathbf{X}{true}.
	•	Could be RMSD-based or a distribution-based loss (like Chamfer or L1 in Cartesian space).
	•	If partial coords from MP-NeRF are also available, we can also apply a direct partial 3D loss earlier in the pipeline.
	2.	Torsion Angle Loss \mathcal{L}_{\text{angle}}:
	•	Compare predicted angles from TorsionBERT to known angles from real structures.
	•	This ensures TorsionBERT remains consistent with ground truth angles.

Optionally, one can combine:
\mathcal{L}{\text{final}} = \lambda{3D} \cdot \mathcal{L}{3D} \;+\; \lambda{\text{angle}} \cdot \mathcal{L}_{\text{angle}}.

Additionally, if the Pairformer trunk is trained for some contact/distance supervision, we might add pairwise distance or distogram losses \mathcal{L}_{\text{pair}}. But typically, we rely on the final 3D or angle constraints. Overall:

\mathcal{L}{\text{end-to-end}} = \lambda{3D}\,\mathcal{L}{3D} + \lambda{\text{angle}}\,\mathcal{L}{\text{angle}} + \lambda{\text{pair}}\,\mathcal{L}_{\text{pair}}.

Backprop:
	•	The gradient from \mathcal{L}_{3D} flows through the diffusion model → merges into the unified latent → modifies the TorsionBERT & Pairformer parameters (via LoRA) → updates adjacency if we let it.
	•	The gradient from \mathcal{L}_{\text{angle}} directly updates TorsionBERT’s LoRA parameters, ensuring it accurately matches known angles.

Validation:
	•	Usually track final 3D RMSD or TM-score, plus angle-level MCQ or MAE.

⸻

2. Detailed Implementation Plan for LoRA / QLoRA

A. TorsionBERT with LoRA

File(s) Potentially Affected: rna_predict/pipeline/stageB/torsion_bert_predictor.py
	1.	Inject LoRA into BERT:
	•	If we use Hugging Face peft or a custom LoRA approach, we wrap the TorsionBert model to add “low-rank adapters” in attention and/or feed-forward layers.
	•	Keep a config like lora_r=4 or lora_alpha=16 to define the rank updates.
	2.	Activating Grad for LoRA:
	•	Freeze all standard BERT parameters, let only LoRA adapter parameters have requires_grad=True.
	•	_init_lora_layers() function inserts the additional weight matrices for \Delta W.
	3.	Forward pass remains the same: input sequence → token embedding → [BERT + LoRA] → final hidden → regression for angles.
	4.	Backward:
	•	Grad from \mathcal{L}{\text{angle}} and \mathcal{L}{3D} flows into LoRA adapters.
	•	Weight updates occur only in the small rank modifications, saving memory.

Architectural Decision:
	•	We must ensure the dimensionality of angle outputs remains the same. The top linear layer that projects hidden states to angle sin/cos can remain fully trainable or also get partial LoRA. Usually, we let it be fully trainable since it’s small.

⸻

B. Pairformer Trunk with LoRA

File(s) Potentially Affected: Possibly a new subfolder models/pairformer_trunk/ or integrated in rna_predict/models/...
	1.	Insert LoRA into Triangular Attention:
	•	For each block of the 48-block trunk, we freeze base attention weights (W_q, W_k, W_v, W_out) but add low-rank adapter layers that approximate the attention transformations.
	•	If we had a partial “pretrained pairformer,” we only adapt the “LoRA-lized” heads.
	2.	Pair-bias:
	•	The adjacency bias can be a small linear transform. We can train that fully or also apply LoRA if it’s large. Usually, it’s small, so no LoRA needed.
	3.	Output:
	•	Still produces \mathbf{z}_{ij} and \mathbf{s}_i.
	•	Grad from final 3D or pairwise constraints flows through these embeddings → modifies LoRA adapters.

Architectural Decision:
	•	If the Pairformer is big, carefully define which layers get LoRA. Possibly only the later blocks for memory efficiency.

⸻

C. Unified Latent Merger (ULM)

File: possibly models/unified_latent_merger.py
	1.	Combining Torsion + Pair:
	•	We parse a node-level embedding for each residue i from TorsionBERT. Another node-level embedding from Pairformer’s sᵢ. Possibly an edge-level embedding from zᵢⱼ.
	•	If we have adjacency, we either feed it in as a feature or let Pairformer handle it.
	2.	LoRA:
	•	If this “ULM” is a small MLP or Transformer, we can either fully train it or embed LoRA if we want to keep it partially frozen. Typically we train it from scratch since it’s a new bridging component.
	3.	Output:
	•	A per-residue “condition embedding” fed into the diffusion, plus an optional per-(i,j) side channel for constraints.

⸻

D. Diffusion Model with LoRA

File: Possibly models/diffusion/angle_diffusion.py or rna_predict/models/diffusion.py
	1.	Architecture:
	•	A UNet or Transformer-based diffusion. We apply it to 3D coordinates.
	•	Takes random/noisy coords + the merged latent. Each step refines coords.
	2.	LoRA:
	•	If the diffusion model is large (like some advanced 3D Transformer), we can freeze the backbone and add LoRA. This is beneficial if we have a large checkpoint for diffusion pretrained on something else (e.g. a generative model from prior data).
	3.	Loss:
	•	Typically a Denoising Score Matching or noise-prediction-based loss (like stable diffusion). The final step output can also be directly compared to ground-truth 3D coords.

⸻

E. MP-NeRF or Forward Kinematics (Optional Stage C)
	1.	Implementation:
	•	If used, each call is a simple geometry transform from angles to partial coords. Doesn’t have big learnable parameters (just standard references).
	•	If you do have “learnable geometry hack,” it’d be minimal and likely not require LoRA.
	2.	Backprop:
	•	The gradient from \mathcal{L}{3D} or \mathcal{L}{\text{angle}} flows back through the trigonometric or matrix multiplication steps, ultimately reaching TorsionBERT’s angle outputs.

⸻

F. Energy Minimization (Post-Diffusion)
	•	Typically no direct backprop from the local minimization.
	•	We treat it as a separate script that polishes final coords or short MD runs.
	•	Because it’s not integrated in the computational graph, it doesn’t produce gradient signals upstream.

⸻

3. Data Structures & Configuration

A. LoRA Parameterization

Approach:
	1.	For each major pretrained model (TorsionBERT, Pairformer, Diffusion trunk), we define a small config dict:

lora:
  r: 4
  alpha: 16
  dropout: 0.1
  target_modules: [attention.W_q, attention.W_k, ...]


	2.	We attach LoRA adapters using something like peft.LoraModel or a custom wrapper.

B. Residue Index & Adjacency Storage

Definition:
	•	ResidueIndexMap: List[int] to unify each stage’s indexing if needed.
	•	adjacency: torch.Tensor shape [N, N], store base-pair probability or one-hot. Possibly use adj_soft for partial differentiability.

⸻

4. Step-by-Step Backprop Flow
	1.	Diffusion final coords vs. ground-truth:
	•	\mathcal{L}_{3D} = RMSD(\hat{X}, X_true).
	•	The partial derivatives w.r.t. \hat{X} pass back into the diffusion’s UNet (some layers are LoRA).
	2.	Unified Latent:
	•	The UNet’s gradient also flows into the latent that conditioned the diffusion. That triggers grads in the “Latent Merger.”
	3.	Pairformer:
	•	The portion of the latent derived from Pairformer’s (z_ij, s_i) is updated. Because Pairformer is partially frozen except the LoRA layers, only LoRA weights get updated.
	4.	TorsionBERT:
	•	The portion from Torsion angles also sees grad if we used partial coords or if the final 3D is influenced by the torsion angles.
	•	TorsionBERT’s LoRA adapters update to better produce angles that yield correct final 3D coords.
	5.	Angle Loss:
	•	If we have direct angle supervision, that also updates TorsionBERT’s LoRA weights.

Hence: We can effectively unify all sub-modules in a single graph, with local or global losses. The majority of large pretrained parameters remain frozen, while small rank-limited LoRA adapter weights get updated.

⸻

5. Potential Implementation Steps

(A) Codebase Reorganization (Optional):
	•	Create rna_predict/peft/ directory to store custom LoRA logic or integrate HF peft.
	•	For TorsionBERT, modify torsion_bert_predictor.py to wrap the BERT model with LoRA.

(B) Pairformer Integration:
	•	If you have a partial “pretrained Pairformer,” define a PairformerLoRAAdapter that wraps each attention block.

(C) Add a “UnifiedPipeline” script or class that orchestrates:
	1.	Adjacency input
	2.	TorsionBERT (LoRA) → angles
	3.	Pairformer (LoRA) → pair embeddings
	4.	Merger → latent
	5.	Diffusion (LoRA optional) → final 3D
	6.	Minimization is post-run

(D) End-to-End Loss:
	•	Decide how to weigh angle-based vs. 3D-based terms.
	•	Possibly create small config in pyproject.toml or a JSON specifying the different \lambda coefficients.

(E) GPU/Memory:
	•	Because all these modules can be large, LoRA helps drastically.
	•	Double-check that you only keep the big pretrained weights in half precision or bfloat16, with minimal overhead for rank-limited updates.

⸻

6. Summary of Architectural Decisions
	1.	Where to Insert LoRA:
	•	TorsionBERT: good idea to freeze base, add LoRA to attention or feed-forward layers.
	•	Pairformer trunk: same approach.
	•	Diffusion model: only if it’s large or pre-trained; else train from scratch if it’s modest in size.
	2.	Single vs. Multi-Loss:
	•	Typically combine angle-level supervision with final 3D loss, to stabilize training.
	3.	Optional Stage C:
	•	The partial 3D from MP-NeRF is differentiable; backprop can refine angles. But if it’s inaccurate or slow, skip it and let diffusion handle raw 3D from noise.
	4.	Computational Efficiency:
	•	We freeze 95% of parameters in TorsionBERT, Pairformer, and (optionally) Diffusion. We only train a small set of LoRA adapter parameters. This keeps VRAM usage manageable.

⸻

7. Conclusion & Next Steps

By combining LoRA-based partial fine-tuning of TorsionBERT, the Pairformer trunk, and (optionally) a diffusion generator, we enable an end-to-end pipeline where final 3D errors drive updates back into the angle and pair embeddings. The main steps are:
	1.	Wrap TorsionBERT with LoRA adapters (freezing base).
	2.	Wrap Pairformer with LoRA adapters (48-block attention).
	3.	Optionally do the same for a large diffusion model or train a smaller diffusion from scratch.
	4.	Construct a single composite forward pass hooking them together with a “unified latent merger.”
	5.	Define \mathcal{L}_{\text{end-to-end}} with coordinate and/or angle supervision.
	6.	Backward: Because each subcomponent is in the same computational graph, gradients reach the LoRA adapters.
	7.	Energy Minimization is done offline, polishing final coordinates.

This approach yields a memory-efficient training procedure thanks to LoRA’s low-rank adaptation, letting you harness large pretrained models within a multi-stage, synergy-focused RNA 3D pipeline.

====
Below is a fully updated, comprehensive design document that merges all four versions (V1–V4) into a single, cohesive guide—addressing their strengths, mitigating their weaknesses, and clarifying past criticisms. It is meant as technical documentation for building an end-to-end, LoRA-friendly RNA 3D structure prediction pipeline with:
	1.	Torsion-based subpipeline (TorsionBERT).
	2.	AlphaFold 3–style Pairformer trunk.
	3.	A Unified Latent Merger combining local angles + global pair embeddings.
	4.	An optional forward kinematics step (Stage C) for partial 3D (using MP-NeRF or similar).
	5.	A Diffusion model for final coordinate generation/refinement.
	6.	A post-inference energy minimization pass.
	7.	Support for LoRA (or QLoRA) to only finetune a small fraction of parameters in large pretrained networks.

The result is more robust, synergistic, and memory-efficient than any single prior version—truly a “best-of-all-worlds” solution.

⸻

1. Grand Overview

1.1 Core Objective

Construct a single end-to-end RNA 3D predictor that:
	•	Generates local torsion angles from (sequence + adjacency).
	•	Extracts global pairwise constraints via an AF3-like Pairformer trunk (optionally leveraging an MSA).
	•	Merges these two representations into a “unified latent.”
	•	Optionally uses forward kinematics to produce partial 3D from the torsion angles.
	•	Employs a Diffusion model to produce final 3D coordinates, guided by both local angles and global pair embeddings.
	•	(Optionally) runs energy minimization or short MD to polish final geometry.

Critically, large pretrained modules (TorsionBERT, Pairformer) remain frozen except for LoRA or QLoRA adapter layers—thus drastically reducing memory usage.

1.2 Data and Stage Flow

   (A) [Sequence + Adjacency + (Optional MSA)]
           └── TorsionBERT (LoRA) → angles
                  └── (Optional) Forward Kinematics → partial 3D
           └── Pairformer (LoRA) → pair embeddings zᵢⱼ + single sᵢ
           └── Unified Latent Merger → "merged latent"
           └── Diffusion (LoRA optional) → final 3D coords
           └── (Optional) Energy Minimization → final polished coords

	•	Stage A: Adjacency can come from “RFold” or any other 2D structure method. Usually not backpropagated.
	•	Stage B: TorsionBERT (LoRA) → angles.
	•	Stage C (optional): Forward kinematics (MP-NeRF or standard NeRF) → partial 3D.
	•	Stage D: Pairformer trunk (LoRA) → global pair embeddings.
	•	Merger: Combines angles + adjacency + pair embeddings → final “latent” for diffusion.
	•	Diffusion: Denoises random/noisy coords into final 3D. Possibly partially or fully trained.
	•	Energy Minimization: Polishing step with no direct gradient to the pipeline.

⸻

2. Mandatory vs. Optional Steps
	1.	Mandatory:
	•	TorsionBERT for angles (Stage B).
	•	Pairformer for pair embeddings.
	•	Unified Latent so that Diffusion sees both local + global constraints.
	•	Diffusion to generate final 3D coordinates.
	2.	Strongly Recommended:
	•	Adjacency from Stage A (or external) to feed TorsionBERT.
	•	Energy Minimization at the end to correct small bond or steric issues.
	3.	Truly Optional:
	•	Forward Kinematics (Stage C) if you want partial 3D from torsion angles.
	•	MSA: if available. Otherwise, Pairformer can run single-sequence mode.
	•	Templates: Could also be integrated but not mandatory.

This ensures synergy: Torsion angles (local) + pair embeddings (global) must meet in the same pipeline. Adjacency is key for local angle constraints, though not strictly forced if you truly have no 2D data.

⸻

3. Detailed Modules & Design Choices

3.1 TorsionBERT (Stage B) with LoRA
	•	Purpose: Predict backbone torsion angles \{\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \chi\} from sequence + adjacency (optionally sugar pucker).
	•	Why Pretrained: TorsionBERT is typically a BERT-like language model, adapted for angle regression.
	•	LoRA:
	•	Freeze base weights, insert low-rank adapters in attention Q/K/V, or feed-forward blocks.
	•	Only these small adapter parameters get updated, keeping GPU memory usage modest.
	•	Output: (N, #angles) for an RNA of length N (plus sugar angles if you want \nu_0..\nu_4).

Backprop Flow
	•	If we have an angle-level sub-loss (\mathcal{L}_{\mathrm{angle}}), it directly updates LoRA layers in TorsionBERT.
	•	If we rely on final 3D loss (\mathcal{L}_{3D}), that gradient can also flow back to TorsionBERT via the diffusion → unify → angles chain.
	•	TorsionBERT indexing must remain consistent with the Pairformer’s residue indexing (Residue 0..N–1).

3.2 Pairformer (AF3-like) with LoRA
	•	Purpose: Provide global pair embeddings \mathbf{z}_{ij} and single embeddings \mathbf{s}_i from MSA or single sequence, optionally incorporating adjacency as a bias.
	•	LoRA:
	•	Large trunk (e.g. 48 blocks). We freeze the main trunk and only adapt a rank-limited set of parameters in each attention or feed-forward sub-layer.
	•	Output:
	•	(N×N, pair_dim) for pair embeddings,
	•	(N, single_dim) for single embeddings.

Backprop Flow
	•	Gradients from \mathcal{L}{3D} or from a pairwise sub-loss (\mathcal{L}{\mathrm{pair}}) update only the LoRA adapter weights, leaving the rest frozen.

3.3 Unified Latent Merger
	•	Purpose: Combine TorsionBERT angles + adjacency + Pairformer embeddings \mathbf{z}_{ij}, \mathbf{s}_i into one “conditioning latent” for diffusion.
	•	Implementation:
	•	Possibly a small Transformer or MLP.
	•	We can train it fully (no need to freeze) or also apply LoRA if it’s large.
	•	Output:
	•	A final latent representation for each residue (and possibly for residue pairs).
	•	Feeds the Diffusion as a “condition.”

Indexing
	•	Must ensure TorsionBERT’s residue i lines up with Pairformer’s residue i, etc.
	•	Use a “ResidueIndexMap” if needed.

3.4 Diffusion Module (Stage D)
	•	Purpose: Iteratively transform random/noised 3D coords (or partial coords from forward kinematics) into final 3D structure.
	•	Implementation:
	•	E.g. a 3D U-Net or GNN that at each step sees the “unified latent” as a condition.
	•	If it’s large, partial freeze with LoRA. If smaller, train from scratch.
	•	Loss:
	•	Typically a diffusion denoising objective or a final L1/RMSD on the coordinates.
	•	The final \mathcal{L}_{3D} is the main synergy enabler: it pushes all upstream modules to produce coherent angles, adjacency constraints, and pair embeddings.

3.5 (Optional) Forward Kinematics
	•	Goal: Use predicted angles from TorsionBERT to compute partial 3D via MP-NeRF or standard NeRF.
	•	Pros: The diffusion starts from a partially folded conformation, possibly reducing required diffusion steps.
	•	Cons: If the angles are inaccurate, the diffusion might need to “unfold” it.
	•	Backprop:
	•	This geometry pipeline is fully differentiable, so final 3D errors can adjust TorsionBERT’s angles.

3.6 Energy Minimization
	•	Goal: Post-hoc local minimization or short MD run in a force field (Amber, CHARMM, or OpenMM).
	•	No gradient flows back; purely to fix small steric or bond-length errors.
	•	Typically done after inference, possibly across multiple diffusion samples to pick best.

⸻

4. Multi-Level Loss Functions & Training Approach

4.1 Potential Loss Terms
	1.	\mathcal{L}_{3D} (Final coordinate-based):
	•	Compare final predicted 3D to ground-truth, e.g.:
\mathcal{L}{3D} = \mathrm{RMSD}\bigl(\hat{X}{\mathrm{final}}, X_{\mathrm{true}}\bigr),
or a distribution-based approach (like FAPE, distogram cross-entropy).
	•	Usually the primary synergy driver.
	2.	\mathcal{L}_{\mathrm{angle}} (optional torsion supervision):
	•	If you have ground-truth angles for each residue, use a circular MSE or MCQ-based measure.
	•	Helps TorsionBERT remain physically consistent.
	3.	\mathcal{L}_{\mathrm{pair}} (optional adjacency or pair-distances):
	•	If the Pairformer is partially finetuned, we can guide it by known contact/distance constraints.

Weighted Sum:

\mathcal{L}{\mathrm{total}}
= \lambda{3D}\,\mathcal{L}_{3D}
	•	\lambda_{\mathrm{angle}}\,\mathcal{L}_{\mathrm{angle}}
	•	\lambda_{\mathrm{pair}}\,\mathcal{L}{\mathrm{pair}}
Typical emphasis is on \lambda{3D}.

4.2 End-to-End Backprop Path
	1.	The final 3D error flows from the diffusion outputs → diffusion parameters → the unified latent → Pairformer + TorsionBERT LoRA → (optionally adjacency if we adopt a soft adjacency approach).
	2.	If we do \mathcal{L}{\mathrm{angle}} or \mathcal{L}{\mathrm{pair}}, they also feed gradients directly into TorsionBERT or Pairformer LoRA layers.

Hence: The entire pipeline can learn from 3D data alone, or from a combination of angles, pair constraints, and final coordinate errors.

⸻

5. LoRA / QLoRA: Minimizing Memory

5.1 Why LoRA
	•	TorsionBERT and the AF3 Pairformer trunk might each have hundreds of millions of parameters.
	•	LoRA adds a small rank-limited set of trainable weights to each large linear transform, drastically reducing GPU memory usage while still enabling backprop.
	•	For extremely large models, QLoRA can also quantize the base model to 4-bit or 8-bit, further shrinking memory footprint.

5.2 Where to Insert LoRA
	•	TorsionBERT: Typically in each Transformer block’s multi-head attention Q/K/V, or feed-forward layers.
	•	Pairformer: Similarly, in the triangular attention or pairwise transformations.
	•	Diffusion: Only if the diffusion model is large or partially pretrained. Otherwise, we can train it fully from scratch.
	•	Unified Merger: Usually small enough to train fully, but LoRA is optional if it’s big.

Example (pseudo-HF approach)

from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], ...)
torsion_bert_lora = get_peft_model(pretrained_torsion_bert, lora_cfg)
# freeze base weights, only LoRA adapters are trainable



⸻

6. Implementation Steps: Putting It All Together

Below is a unified approach that merges the deeper code-level detail (Version 1), synergy perspective (Version 2), stepwise memory/LoRA usage (Version 3), and final indexing clarity (Version 4).

6.1 Overall Pipeline Construction
	1.	Load the adjacency (Stage A) from an external predictor (RFold) or from data.
	2.	Load TorsionBERT (with LoRA), freeze base weights:
	•	torsion_bert_lora = get_peft_model(...).
	3.	(Optional) load or define a forward kinematics function (MP-NeRF):
	•	If used, produce partial 3D from TorsionBERT angles.
	4.	Load Pairformer trunk (with LoRA):
	•	Possibly also freeze the main trunk.
	5.	Implement a “UnifiedLatentMerger” that merges angles + adjacency + pair embeddings → final “latent.”
	6.	Build or load the Diffusion model (LoRA if large, or train from scratch if small).
	7.	In a single forward pass:
	•	Torsion angles → (FK → partial coords?).
	•	Pair embeddings zᵢⱼ.
	•	Merge into a final latent.
	•	Diffusion yields final 3D.
	•	Compare to ground-truth 3D (RMSD, L1, or distance-based).
	8.	Loss is backpropagated:
	•	Only LoRA adapters in TorsionBERT + Pairformer + (optionally) Diffusion are updated.

6.2 Example Training Loop

model = FullRNA3DPipeline(
    torsion_bert_lora,
    pairformer_lora,
    unify_module,
    diffusion_module,
    forward_kinematics=(use_fk)
)
optimizer = torch.optim.AdamW(model.lora_params(), lr=1e-4)

for batch in train_loader:
    seq, adjacency, coords_gt, angles_gt, MSA = batch

    # Forward pass
    final_coords, loss_dict = model(seq, adjacency, MSA, coords_gt, angles_gt)

    # Weighted sum
    total_loss = (lambda_3D * loss_dict["3D_loss"]
                  + lambda_angles * loss_dict["angle_loss"])
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

Memory is drastically reduced because we only keep gradient states for LoRA adapter matrices.

⸻

7. Validation & Ensemble Refinement

7.1 Validation Metrics
	•	Angle-level: circular MSE or MCQ.
	•	Pair-level: if pair supervision is available, contact or distogram accuracy.
	•	3D-level: RMSD, GDT, or specialized RNA metrics. Possibly sugar pucker accuracy.
	•	Possibly break down by region (stems vs. loops vs. single-stranded segments).

7.2 Ensemble & Minimization
	•	After training, we can generate multiple final 3D structures by random seeding the diffusion or sampling.
	•	Energy Minimization: run each through a short local MD or minimization to fix small geometry.
	•	Rank by a geometry score or an internal model confidence measure.
	•	Output top 5 or a single best structure.

⸻

8. Advantages Over Earlier “Versions”
	1.	Multi-Loss synergy (from V1): We incorporate angle, pair, and final 3D constraints in a single pipeline.
	2.	High-level clarity (from V2): Emphasizes that final 3D backprop unifies TorsionBERT + Pairformer.
	3.	Implementation practicalities (from V3): We detail LoRA injection points, stagewise training, memory usage tips, micro-batching, etc.
	4.	Indexing and adjacency (from V4): We specify the importance of consistent residue numbering, highlight optional forward kin, and mention adjacency usage in both TorsionBERT and Pairformer.
	5.	Explicit mention of sugar ring angles, potential re-labeled residues, partial or full training approach, plus chunking or micro-batching for large RNAs.

By combining all these points, we address the criticisms from earlier versions:
	•	We keep the pipeline synergy (no “lost synergy” from making everything optional).
	•	We specify how adjacency can feed both TorsionBERT and Pairformer.
	•	We highlight a single “ResidueIndexMap” for alignment.
	•	We detail how the “Unified Latent Merger” is more robust than a simple MLP approach.
	•	We incorporate a short local minimization step at the end for geometry polishing.

⸻

9. Conclusions & Best Practices

Key Guidance:
	1.	Use LoRA: It’s essential for large pretrained TorsionBERT / Pairformer. Keep them in half precision or even 4-bit QLoRA if extremely large.
	2.	Define Weighted Loss: Typically \mathcal{L}{3D} is the main driver. If you have angle ground truth, do \mathcal{L}{\mathrm{angle}} to speed convergence.
	3.	Forward Kinematics: Optionally do partial 3D from angles. If the angles are decent, it helps the diffusion. If they’re poor, it might hamper training.
	4.	Energy Minimization: Great as a final step, not part of backprop.
	5.	Indexing: Absolutely ensure consistent residue numbering across TorsionBERT and Pairformer.
	6.	Sampling: If the pipeline is large, do gradient checkpointing or micro-batching to avoid out-of-memory issues.

Outcome: A single end-to-end pipeline that merges local angle constraints, global pair constraints, a final generative diffusion, and a partial post-processing step for geometry smoothing—maximizing synergy and controlling memory via LoRA.

Thus, this final design stands as a verbose, cohesive, and thoroughly integrated architecture that surpasses any individual “Version 1–4” by merging their best features and clarifications into one complete technical document.
====
