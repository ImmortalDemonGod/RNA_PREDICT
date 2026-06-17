# 02 — Static Audit

_Every defect findable by reading. Promoted only after surviving adversarial falsification (fixpoint reached in 5 round(s))._

## Summary
- Findings (survivors): **538**
- Severity: critical=4, high=75, medium=217, low=234, info=8
- Source files in denominator: **304**; examined: **305**
- Survived adversarial falsification; **0** kept as _unverified_ (falsifier could not adjudicate).
- Judged against Stage-1 provisional intent.

## Findings
### [CRITICAL] s2c3l0-001 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:22`
- **Evidence:** FeatureProcessor (line 22), AttentionComponents (attention_components.py:16) and CoordinateProcessor (coordinate_processing.py:13) are plain Python classes (no nn.Module base) but construct nn.Module submodules (LinearNoBias, nn.Sequential small_mlp, LayerNorm). They are assigned as attributes of the nn.Module AtomAttentionEncoder/AtomAttentionDecoder (atom_attention/encoder.py:56,73 ; atom_attention/decoder.py:48,56,63). Because nn.Module.__setattr__ only registers child Modules when the assigned value is itself an nn.Module, none of the parameters inside these wrapper objects are registered. Consequently they are absent from .parameters()/.state_dict(), are NOT moved by .to(device)/.cuda(), are NOT seen by the optimizer, and are NOT saved/loaded in checkpoints.
- **Recommendation:** Make FeatureProcessor/AttentionComponents/CoordinateProcessor subclass nn.Module (call super().__init__()), or register their layers directly on the parent encoder/decoder via setup_* functions as the refactored atom_attention_encoder.py does. Add a unit test asserting len(list(encoder.parameters())) covers these layers.

### [CRITICAL] s2c3l0-020 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py:4`
- **Evidence:** InputFeatureEmbedder imports `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder` at module top level (line 4) and again inside __init__ (line 36, AtomEncoderConfig). The package rna_predict/models does not exist (verified: `ls rna_predict/models` -> No such file or directory; the sibling atom_encoder.py:6 even comments 'Corrected import path from models.attention to legacy.attention'). Therefore importing this legacy module raises ModuleNotFoundError immediately — the file is unimportable.
- **Recommendation:** Repoint the imports to rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder (the real location), or delete this dead legacy module if superseded by the current/ tree.

### [CRITICAL] s2c6l2-006 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:251-293,418-427`
- **Evidence:** apply_tensor_fixes() (invoked in production at rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py:137) calls fix_tensor_add(), which globally replaces torch.Tensor.__add__ (:293) with a wrapper that, on any size-mismatch RuntimeError, silently unsqueezes/interpolates/expands operands to force the addition (lines 258-290). This mutates the semantics of '+' for every tensor in the process, masking genuine shape bugs and yielding silently-incorrect numerical results in a scientific structure-prediction pipeline.
- **Recommendation:** Remove global operator monkey-patching; fix the underlying shape mismatches at their source. If adaptation is truly needed, do it explicitly at the specific call sites with asserts, not by overriding torch.Tensor.__add__.

### [CRITICAL] s2c6l0-tenops-add-silent — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:19-38`
- **Evidence:** fix_tensor_add() monkeypatches torch.Tensor.__add__ globally; on a shape-mismatch RuntimeError ('must match the size'/'at non-singleton dimension') it does NOT add the tensors at all but returns whichever operand has more dims unchanged (`return self`/`return other`, lines 28-32). The addition is silently dropped, producing numerically wrong results everywhere `+` is used after apply, with no error. This corrupts diffusion/embedding arithmetic across the whole process once installed.
- **Recommendation:** Remove this patch. Never globally override Tensor.__add__; fix the real shape mismatch at the call site. If a compatibility shim is unavoidable, raise rather than return a silently-incorrect operand.

### [HIGH] s2c0l0-004 — bug
- **Location:** `.github/workflows/main.yml:39`
- **Evidence:** The 'Check for dependency vulnerabilities' step runs `pip freeze > requirements.txt`, overwriting the tracked source-of-truth requirements.txt with the full frozen CI environment. The later 'ruff auto-fix' steps (main.yml:50-70) then run `git add -A`, `git commit`, and `git push` on push-to-main events, so the clobbered requirements.txt can be committed back to the repository, corrupting the curated dependency list.
- **Recommendation:** Write the freeze to a throwaway file (e.g. `pip freeze > /tmp/frozen.txt`) and run pip-audit against that, never overwriting the repo's requirements.txt; or scope `git add` to specific paths instead of `-A`.

### [HIGH] s2c0l2-0006 — intent_mismatch
- **Location:** `.github/workflows/release.yml:38`
- **Evidence:** release.yml builds via `python setup.py sdist bdist_wheel`, but setup.py declares `version="1.0.0"` and `python_requires=">=3.8"` (setup.py:6,31), whereas pyproject.toml is the real metadata with `version = "2.0.8"` and `requires-python >= 3.10` (pyproject.toml:7,11) and rna_predict/VERSION says 2.0.8. The release pipeline would publish wheels labelled 1.0.0 with the wrong dependency set, diverging from the actual 2.0.8 package.
- **Recommendation:** Build with the PEP517 frontend (`python -m build`) so pyproject.toml metadata is used, and delete or reconcile the stale setup.py.

### [HIGH] s2c0l0-002 — bug
- **Location:** `Containerfile:1`
- **Evidence:** `FROM python:3.7-slim` but the project requires Python >=3.10 (pyproject.toml:11 `requires-python = ">=3.10"`, mypy.ini:2 `python_version = 3.10`, CI uses 3.11 in .github/workflows/main.yml:21). `RUN pip install .` (Containerfile:4) will fail dependency resolution / syntax on 3.7 (e.g. lightning>=2.2, torch>=2.0.1 wheels are unavailable for cp37 and the codebase uses 3.10+ syntax).
- **Recommendation:** Bump the base image to python:3.11-slim (or 3.10) to match requires-python and the CI matrix.

### [HIGH] s2c0l0-003 — bug
- **Location:** `Containerfile:5`
- **Evidence:** `CMD ["rna_predict"]` invokes the console script defined at pyproject.toml:61, whose target module `rna_predict.__main__:main` does not exist anywhere in the tree (verified via find). Even if the image built, the container's default command would crash at startup with an import error.
- **Recommendation:** Fix the console-script target (see s2c0l0-001) or change CMD to a working entry point, e.g. ["python", "-m", "rna_predict.predict"].

### [HIGH] s2c0l0-001 — bug
- **Location:** `pyproject.toml:61`
- **Evidence:** [project.scripts] declares `rna_predict = "rna_predict.__main__:main"`. A repo-wide `find . -name __main__.py` returns no results (verified across the full tree), so the package's only console-script entry point references a module that does not exist. Installing the wheel creates an `rna_predict` command that fails with ModuleNotFoundError on invocation. The Stage-1 map (audit/01-understanding.md:15) records the same missing target.
- **Recommendation:** Either add rna_predict/__main__.py defining main(), or repoint the console script to an existing Hydra main such as `rna_predict.predict:main` or `rna_predict.interface:main`.

### [HIGH] s2c1l2-dimsconfig-reduced-defaults — intent_mismatch
- **Location:** `rna_predict/conf/config_schema.py:46-107`
- **Evidence:** DimensionsConfig (and every per-stage dataclass) ships defaults that have been slashed from the documented AlphaFold3-inspired sizes to tiny test values, with inline comments admitting it: c_s default=8 '# CHANGED: was 384', c_z=4 '# was 128', c_s_inputs/c_token=8 '# was 449', c_atom=4 '# was 128', c_noise_embedding=4 '# was 32'. StageAConfig.num_hidden default=8 '# CHANGED: was 128 (to reduce memory usage)'. The Stage-1 provisional intent (audit/01-understanding.md:6,9) is a functional sequence-to-structure inference pipeline; these schema defaults instantiate a non-functional toy model unless a YAML overrides them. The structured schema, which exists to be the validated source of truth, instead encodes throwaway debug dimensions.
- **Recommendation:** Restore production dimensions as the dataclass defaults (c_s=384, c_z=128, c_s_inputs/c_token=449, c_atom=128, etc.) and move the reduced sizes into a dedicated 'test'/'minimal' Hydra override group, so the schema default reflects the real intended architecture.

### [HIGH] s2c1l2-schema-yaml-dim-drift-protenix — doc_drift
- **Location:** `rna_predict/conf/config_schema.py:604-613 vs rna_predict/conf/model/protenix_integration.yaml:8-12`
- **Evidence:** Two sources of truth for the same dimensions disagree. ProtenixIntegrationConfig defaults c_token=8, restype_dim=8, profile_dim=8, c_atom=4, c_pair=4 (all marked 'CHANGED: was 449/32/32/128/32'). The YAML actually composed into default.yaml (default.yaml:14 'model/protenix_integration@model.protenix_integration') sets c_token=449, restype_dim=32, profile_dim=32, c_atom=128, c_pair=32. Whichever wins depends on Hydra merge order, and the schema's documented default is the opposite of what the runtime YAML supplies, so the schema cannot be trusted as documentation.
- **Recommendation:** Pick one authoritative set of production dimensions and make the dataclass default and the YAML agree; or have the YAML interpolate the dimension from the structured schema/shared block rather than hardcoding a conflicting literal.

### [HIGH] s2c1l2-predict-yaml-hardcoded-checkpoint — intent_mismatch
- **Location:** `rna_predict/conf/predict.yaml:13`
- **Evidence:** checkpoint_path: /Users/tomriddle1/RNA_PREDICT/outputs/checkpoints/last.ckpt is a hardcoded absolute developer path in predict.yaml, which is the config_name for predict.py — the primary, README-recommended inference entry (audit/01-understanding.md:28). On any other machine this path does not exist, so the documented 'Functional' inference flow loads a default/wrong checkpoint or fails. Comment '# Updated for correct test location' confirms it was tuned to one developer's box.
- **Recommendation:** Make checkpoint_path relative to the project (e.g. outputs/checkpoints/last.ckpt) or an env/CLI override (${oc.env:RNA_CKPT,...}); never ship an absolute /Users/... path in the default inference config.

### [HIGH] s2c1l2-loader-dummy-shape-mismatch — bug
- **Location:** `rna_predict/dataset/loader.py:291-298 vs 300-304`
- **Evidence:** _load_atom_features returns inconsistent tensor ranks/dtypes between its two branches. The empty-pdb dummy branch returns coords of shape (L, max_atoms, 3), atom_mask (L, max_atoms) float32, atom_to_tok (L, max_atoms) int32, elem_emb/name_emb (L, max_atoms, C) (lines 291-295). The real branch returns coords (max_atoms, 3), atom_mask (max_atoms,) bool, atom_to_tok (max_atoms,) long, elem_emb/name_emb (max_atoms, C) (lines 300-304). The function docstring (lines 255-261) documents only the 2D shapes. Downstream collate/model code cannot consume both a 2D and a 3D coords tensor, so the missing-file path produces shapes that disagree with every real sample.
- **Recommendation:** Make the dummy branch emit the same rank/dtype as the real branch ((max_atoms,3) coords, bool atom_mask, etc.), and add a shape assertion so the two paths cannot diverge.

### [HIGH] s2c1l0-loader-atomfeat-shape-mismatch — bug
- **Location:** `rna_predict/dataset/loader.py:291-298 vs 300-326`
- **Evidence:** _load_atom_features returns tensors of DIFFERENT rank/dtype depending on whether the structure file is present. Missing-file path (lines 291-298) returns coords shape (L, max_atoms, 3), atom_mask (L, max_atoms) float32, atom_to_tok (L, max_atoms) int32, elem_emb (L, max_atoms, ref_element_size). The real-data path (lines 300-326) returns coords (max_atoms, 3), atom_mask (max_atoms,) bool, atom_to_tok (max_atoms,) long, elem_emb (max_atoms, ref_element_size). __getitem__ (line 129) stores these directly into the sample, so a batch mixing present/absent structure files produces tensors of incompatible shapes/dtypes; rna_collate_fn (collate.py:99 torch.stack) will raise or silently produce wrong batch shapes, and downstream code receives a per-residue-blocked tensor in one case and a flat-atom tensor in the other.
- **Recommendation:** Make the dummy/missing-file branch produce identical rank and dtypes as the real branch: coords (max_atoms,3) coord_dtype, atom_mask (max_atoms,) bool, atom_to_tok (max_atoms,) long, elem_emb/name_emb (max_atoms, size) float32.

### [HIGH] s2c1l2-latent-merger-rebuilds-weights-and-ignores-config — design_defect
- **Location:** `rna_predict/pipeline/merger/simple_latent_merger.py:63-73`
- **Evidence:** SimpleLatentMerger.forward() reconstructs self.mlp with brand-new randomly-initialized nn.Linear layers whenever the runtime input dim differs from the constructed in_features (lines 63-73). At inference this silently discards any loaded/trained weights and emits output from an untrained MLP, with only a '[Debug] Creating MLP' print. Separately, the merger ignores the LatentMergerConfig contract: config_schema LatentMergerConfig defines merge_method ('concat'/'add'/'attention'), attention_heads, use_residual, output_dim=384 (config_schema.py:1148-1176), but this implementation hardcodes concat, has no residual/attention path, and takes only positional dim_* args — none of those config fields are referenced here.
- **Recommendation:** Validate/raise on dimension mismatch instead of rebuilding the network at forward time (or use a lazy module initialized once), and either implement the LatentMergerConfig options (merge_method/use_residual/attention_heads) or remove them from the schema so config and code agree.

### [HIGH] s2c2l2-rfold-istest-returns-zeros — intent_mismatch
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:520-528`
- **Evidence:** RFoldModel.forward contains `is_test = seqs.shape[0] <= 2 and seqs.shape[1] <= 16` and, when true, returns `torch.zeros((B, L, L))` instead of running the U-Net/Seq2Map. This is gated only on tensor sizes, not on any test flag. rfold_predictor.predict_adjacency pads every sequence to a multiple of 16 via _get_cut_len (rfold_predictor.py:316-329, :401), so any RNA sequence of length <=16 produces padded_len==16 with batch 1, hitting is_test and silently returning an all-zero adjacency matrix. Stage-1 provisional intent (01-understanding.md:6) is that inference (Stage A adjacency) is the functional deliverable; returning zeros for short sequences contradicts that.
- **Recommendation:** Remove the size-based is_test shortcut from production forward. If a fast path is needed for unit tests, gate it behind an explicit test-only constructor flag or move it into the test harness; never infer 'test mode' from real input dimensions.

### [HIGH] s2c2l0-001 — bug
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:520-528`
- **Evidence:** RFoldModel.forward sets is_test = seqs.shape[0] <= 2 and seqs.shape[1] <= 16, and when true returns torch.zeros((B, L, L)) instead of running the U-Net/Seq2Map. This 'test mode' heuristic fires on REAL inference: rfold_predictor.StageARFoldPredictor.predict_adjacency always calls the model with batch=1 (rfold_predictor.py:410 asserts shape[0]==1) and pads the sequence to a multiple of 16 via _get_cut_len, so any RNA sequence of length <=16 yields padded_len==16, triggering is_test and producing an all-zero adjacency matrix for genuine input. The stated intent (predict 2D adjacency/secondary structure) is silently violated for short sequences.
- **Recommendation:** Remove the is_test/zeros shortcut from production forward(); gate any test-only behavior behind an explicit constructor/config flag (e.g. self.test_mode) rather than inferring it from input dimensions, so real short-sequence inputs are processed by the real network.

### [HIGH] s2c2l1-rfold-torchload-pickle — security
- **Location:** `rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:279`
- **Evidence:** StageARFoldPredictor._load_checkpoint calls `ckp = torch.load(checkpoint_path, map_location=self.device)` with no `weights_only=True` and no integrity check. torch.load uses Python pickle, which executes arbitrary code embedded in the file during unpickling. checkpoint_path comes from Hydra config (stage_cfg.checkpoint_path, set at :167) and the code explicitly contemplates the file being absent and fetched from a remote `checkpoint_url` (:191, :259-262) — the companion Stage A runner downloads and unzips it from a URL via urllib (rna_predict/pipeline/stageA/run_stageA.py:70 download_file + extract, referenced at :190). A malicious or MITM-tampered RFold checkpoint .pth therefore yields arbitrary code execution at load time. The provisional intent (Stage 1 map: 'RFold checkpoint download/extraction', README marks Inference as Functional) treats checkpoint loading as a normal trusted step, so the unrestricted pickle load is a defect relative to safe model-weight loading.
- **Recommendation:** Load with `torch.load(checkpoint_path, map_location=self.device, weights_only=True)` (or use a safetensors format) so only tensors/state-dicts are deserialized, never arbitrary objects. Additionally verify the downloaded artifact against a pinned checksum/signature before loading, and serve the checkpoint over HTTPS from a trusted host.

### [HIGH] s2c2l2-adaln-runtime-layer-recreation — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:180-192`
- **Evidence:** AdaptiveLayerNorm.forward, when s.shape[-1] != layernorm_s.normalized_shape, recreates self.layernorm_s, self.linear_s and self.linear_nobias_s as brand-new modules inside the forward pass and mutates self.c_s/self.c_s_layernorm. This discards any trained weights for those layers and replaces them with freshly-initialized parameters mid-run; in training these new params are not in the optimizer, and in inference loaded checkpoint weights are silently thrown away. AF3 Algorithm 26 (which this claims to implement, :24) has fixed dimensions.
- **Recommendation:** Validate/adjust input feature dimensions upstream (or raise) instead of silently rebuilding learnable layers during forward; never re-instantiate parameters inside forward().

### [HIGH] s2c2l2-attninternal-test-hardcoded-reshapes — intent_mismatch
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils_internal.py:170-195,256-259,290-293,341-342`
- **Evidence:** The attention output path (wrap_up -> _infer_and_reshape / apply_gating, all reachable from Attention.forward) is littered with test-specific hardcoded reshapes and magic numbers: apply_gating has 'Special case for the test_n_sample_handling test' keyed on `o.numel()==8192 and o.shape[1]==128` with literals 1024/8/128/64 (lines 170-195); _infer_and_reshape repeats 'Special case for the test_n_sample_handling test' returning `o.reshape(64,128)` when numel==8192 (lines 256-259, 290-293); wrap_up has 'Special case for the specific error we're seeing' for `o.shape[-1]==1024 and in_features==128` (lines 341-342). Production tensor reshaping is shaped around specific test fixtures rather than a principled shape contract.
- **Recommendation:** Define and enforce an explicit shape contract for attention outputs and remove all test-named special cases and magic-number reshapes; encode the expected shapes in the tests, not in production reshape logic.

### [HIGH] s2c3l0-002 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:164`
- **Evidence:** create_pair_embedding treats ref_pos as 2D [N,3]: d=linear_no_bias_d(ref_pos) then p_i=p.unsqueeze(1) (line 182), p_j=p.unsqueeze(0) (line 183), p_ij=p_i+p_j (line 184). For a batched input [B,N,c] this yields p.unsqueeze(1)=[B,1,N,c] and p.unsqueeze(0)=[1,B,N,c] which broadcast to [B,B,N,c] (an incorrect batch-x-batch outer product), not the intended [B,N,N,c]. extract_atom_features however handles batched [B,N,*] via torch.cat, so the two halves of the encoder disagree on rank.
- **Recommendation:** Use dim-relative unsqueeze (e.g. p.unsqueeze(-2) and p.unsqueeze(-3)) so the pair outer product is computed over the atom axis regardless of batch dims, and add a shape assertion for [..., N, N, c_atompair].

### [HIGH] s2c3l2-003 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:56-81`
- **Evidence:** AtomAttentionEncoder (nn.Module, line 23) and AtomAttentionDecoder (decoder.py:20) assign self.feature_processor/self.coordinate_processor/self.attention_components to PLAIN classes (FeatureProcessor at atom_attention_feature_processing.py:22, AttentionComponents at attention_components.py:16, CoordinateProcessor at coordinate_processing.py:13 — none subclass nn.Module). The LinearNoBias/LayerNorm/AtomTransformer layers they hold are therefore never registered as submodules: they are absent from .parameters(), .state_dict() and are not moved by .to(device). For an encoder claiming to implement AF3 Algorithm 5 (encoder.py:26) this means its core weights are untrainable/unsaveable through the standard nn.Module API.
- **Recommendation:** Make FeatureProcessor/AttentionComponents/CoordinateProcessor subclass nn.Module (or register their layers via nn.ModuleDict/add_module on the parent), so parameters are tracked.

### [HIGH] s2c3l2-013 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:622`
- **Evidence:** AttentionPairBias.forward begins (line 622) with an unconditional `print(f"[DEBUG][APB] ENTRY: ...")` and emits five more unconditional `print("[DEBUG][APB] ...")` statements (lines 643,656,659,664,665). Because a statement (the print) precedes the triple-quoted block at lines 623-637, that block is NOT the function docstring (forward.__doc__ is None) and the documented args are lost. This is the core attention block of the transformer; every forward floods stdout.
- **Recommendation:** Remove the debug prints (or use logger.debug guarded by isEnabledFor) and move the docstring to the first statement of forward.

### [HIGH] s2c3l2-020 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:251-257`
- **Evidence:** In _process_style_embedding, the conditional `if getattr(c_l,'shape',[None])[1] != getattr(x,'shape',[None])[1] and atom_to_token_idx is not None:` (line 251) has a body consisting ONLY of comments ('# ... (no change to the detailed broadcasting logic...)', '# (The rest of the function remains unchanged...)', lines 253-256) — the actual 'Broadcasting c_l from residues to atoms using atom_to_token_idx' operation it claims to perform was removed, leaving a no-op. The same gutting appears at lines 284-288 ('# (rest of fallback unchanged)'). The intended residue->atom broadcast is silently absent.
- **Recommendation:** Restore the removed broadcasting logic or delete the dead conditional and document the real behaviour; do not leave placeholder comments standing in for code.

### [HIGH] s2c3l2-019 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:392-393`
- **Evidence:** process_inputs_with_coords — the main coordinate forward path for the refactored AtomAttentionEncoder (called from atom_attention_encoder.py:167) — starts with two UNCONDITIONAL `print("[DEBUG][process_inputs_with_coords] ...")` statements (lines 392-393), bypassing the file's own config-driven `debug` flag used everywhere else. Every coords forward prints to stdout.
- **Recommendation:** Replace the prints with logger.debug guarded by the existing `debug` flag.

### [HIGH] s2c3l0-010 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/pair_embedding.py:39`
- **Evidence:** create_pair_embedding builds the atom-pair embedding with nested Python loops over all atom pairs: _process_distances iterates `for query_idx in range(N) for key_idx in range(N)` (lines 39-40) calling encoder.linear_no_bias_d per pair (line 54) and indexed-assigning into pair_embed (line 63); _process_charges does the same O(N^2) Python double loop (lines 110-125). For realistic RNA atom counts (hundreds-to-thousands) this is O(N^2) Python-level iterations with a per-pair nn.Linear call, making the encoder effectively unusable beyond toy inputs and breaking the 'functional inference' intent.
- **Recommendation:** Vectorize: compute all pairwise distance vectors via ref_pos.unsqueeze(-2)-ref_pos.unsqueeze(-3) and apply linear_no_bias_d once over [...,N,N,3]; compute charge products with an outer product, eliminating the Python loops.

### [HIGH] s2c3l2-025 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/pair_embedding.py:39-68`
- **Evidence:** create_pair_embedding builds the [N_atom,N_atom,c_atompair] pair tensor via _process_distances (lines 39-68) and _process_charges (lines 110-125), each using O(N_atom^2) nested Python for-loops that call encoder.linear_no_bias_d per pair and perform in-place autograd writes `pair_embed[...,q,k,:] += ...`. For realistic atom counts this is orders of magnitude slower (and far more memory/graph-heavy) than a vectorized outer-difference + linear, undermining the 'Functional' inference path.
- **Recommendation:** Vectorize: compute all pairwise distance vectors with broadcasting (ref_pos[...,None,:,:]-ref_pos[...,:,None,:]) and apply the linear once; same for charge products.

### [HIGH] s2c3l2-017 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:108-134`
- **Evidence:** ConditionedTransitionBlock.forward applies AdaptiveLayerNorm TWICE due to an instrumentation block. Line 110 sets `a_norm = self.adaln(a, s)`; line 119 sets `a = a_norm`; then line 131 recomputes `a = self.adaln(a, s)` on the already-normalized a_norm. The intermediates linear_a1/linear_a2/b computed at lines 112-117 are discarded and recomputed at line 134 on the doubly-normalized tensor. The real output path therefore uses adaLN applied twice — a correctness regression introduced by the instrumentation, affecting every DiffusionTransformerBlock feed-forward (diffusion.py:119).
- **Recommendation:** Delete the instrumentation block (lines 109-119) so adaln is applied once; keep only the lines 130-153 computation.

### [HIGH] s2c3l0-005 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:119`
- **Evidence:** ConditionedTransitionBlock.forward computes a_norm=self.adaln(a,s) (line 110) and then sets a=a_norm (line 119) before the 'real' forward body. The real body re-applies adaptive layernorm: a=self.adaln(a,s) (line 131), so adaln is applied TWICE to the input (adaln(adaln(input))) and the subsequent gated SiLU (line 134) operates on the doubly-normalized tensor. This is a numerical correctness defect introduced by leftover instrumentation code (lines 108-119) sitting above the docstring/real logic.
- **Recommendation:** Delete the instrumentation block (lines 108-119) including the `a = a_norm` reassignment so adaln is applied exactly once; keep only the documented forward path.

### [HIGH] s2c3l2-029 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:173`
- **Evidence:** aggregate_atom_to_token branches on test identity in production: it reads `os.environ.get('PYTEST_CURRENT_TEST')` (line 173) and special-cases named tests 'test_run_stageD_basic' (line 183) and 'test_run_stageD_diffusion_inference_original' (lines 234,300), taking different reshape/fallback paths only when those tests are running (including an O(N^2) python double-loop scatter fallback at lines 318-322). A core aggregation utility behaving differently under specific pytest names means tests exercise code paths the real pipeline never takes, and vice-versa.
- **Recommendation:** Remove all PYTEST_CURRENT_TEST/test-name branches; implement one shape-handling path and fix the underlying shape contracts so tests and production share behaviour.

### [HIGH] s2c3l1-001 — security
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:112-113`
- **Evidence:** unzip_file() extracts a downloaded checkpoint archive with `with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extract_dir)` (run_stageA.py:112-113) with NO validation of member names. A crafted zip whose entries contain '../' path components (or absolute paths) writes files outside extract_dir (classic Zip Slip, CWE-22 arbitrary file write). The archive originates from a network download (download_file at run_stageA.py:70-71) whose URL is the Hydra-configurable stage_cfg.checkpoint_url (run_stageA.py:188, default conf/model/stageA.yaml:12). extract_dir is os.path.dirname(checkpoint_dir) (run_stageA.py:191), i.e. an attacker-influenced member like '../../RFold/../../../home/user/.bashrc' could overwrite files. The intended behavior per 01-understanding.md is to fetch the RFold pretrained checkpoint, not to write arbitrary host paths.
- **Recommendation:** Before extracting, validate every member: reject names that are absolute or whose os.path.realpath(os.path.join(extract_dir, name)) does not stay within os.path.realpath(extract_dir); or extract members individually with a sanitized basename. Prefer Python 3.12's ZipFile.extractall(filter='data') / shutil.unpack_archive equivalents.

### [HIGH] s2c3l2-035 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageB/main.py:171-175`
- **Evidence:** run_stageB_combined branches on `pairformer_model.return_value` (lines 171-175) — `return_value` is a unittest.mock.Mock attribute, and the surrounding comments (lines 164-167) explicitly say 'The test is mocking the pairformer_model ... use the mock's return values directly'. Production library code thus inspects test-mock internals to decide its data flow, coupling the shipped Stage B orchestrator to the test harness.
- **Recommendation:** Remove the return_value branch; rely solely on the genuine pairformer_output and assert its type/shape.

### [HIGH] s2c3l2-036 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageB/main.py:225-245`
- **Evidence:** The ProtenixIntegration 's_inputs' branch fabricates synthetic inputs 'for testing purposes ... to speed up tensor generation': it shrinks dims (test_c_atom=min(32,c_atom), test_restype_dim=min(8,...), test_profile_dim=min(8,...) at lines 227-229) and builds constant/hard-coded input_features (ref_pos from a fixed 2-point tensor repeated, ref_charge/ref_element/restype/profile all torch.ones*0.1, lines 232-245). So the returned s_inputs are derived from fabricated dummy features rather than real embeddings, even in the non-test production call path of run_stageB_combined.
- **Recommendation:** Build input_features from the real sequence/embedding data at the configured dimensions; remove the 'for testing' downsizing from the production path.

### [HIGH] s2c4l2-003 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:378-415`
- **Evidence:** PairformerWrapper.predict() docstring (:381-392) states 'Predict RNA structure using the Pairformer model' and returns single/pair embeddings, but the body returns random tensors: `s_emb = torch.randn(L, self.c_s ...)`, `z_emb = torch.randn(L, L, self.c_z ...)` (:399-400) with the comment 'For now, return dummy tensors' (:397). self.stack (the real PairformerStack) is never invoked. Two unconditional debug prints remain at :413-414. This contradicts the Stage-1 intent that the pairwise branch produces meaningful embeddings.
- **Recommendation:** Implement predict() to run the embeddings through self.stack, or remove the method and document that only forward() is functional; remove the stray print() calls.

### [HIGH] s2c4l2-009 — design_defect
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:164-205,885-897`
- **Evidence:** Production predictor code branches on test identity via os.environ['PYTEST_CURRENT_TEST'] and hardcoded test names: raises for 'test_legacy_config_path_raises' (:164,:173), forces dummy_mode for named tests (:189), raises for 'test_stageb_torsionbert_config_structure_property' (:204), and __call__ contains a 'Special case for tests' block reshaping output to [N,16] when num_angles==16 and angle_mode=='degrees' (:885-897). Test-specific behavior is baked into the runtime model, coupling production output to test names.
- **Recommendation:** Move all test-only behavior into test fixtures/mocks; remove PYTEST_CURRENT_TEST and test-name conditionals and the num_angles==16 reshape from production code paths.

### [HIGH] s2c4l1-001 — security
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:294-317`
- **Evidence:** StageBTorsionBertPredictor loads the TorsionBERT tokenizer and model with trust_remote_code=True in all four branches: AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True, local_files_only=True) (line 294), AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True) (line 301), AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True) (line 307), AutoModel.from_pretrained(..., trust_remote_code=True) (line 314). trust_remote_code=True causes HuggingFace transformers to download and execute arbitrary Python (modeling_*.py / configuration_*.py) from the model repository at load time. self.model_name_or_path defaults to the third-party hub id 'sayby/rna_torsionbert' (DEFAULT_MODEL_PATH at line 20) and is otherwise taken directly from Hydra config (lines 240/266: getattr(torsion_cfg, 'model_name_or_path', ...)). For non-local ids (else branches at 299/312) the repo is fetched from the network with no pinned revision/commit hash, so a compromised or hijacked hub repo, or a config pointing at an attacker-controlled repo, results in remote code execution in the inference/training process. This is the README-documented default Stage B model, so the dangerous path is the normal execution path, not an edge case.
- **Recommendation:** Default trust_remote_code to False and only enable it behind an explicit, documented opt-in flag for a vetted model. Pin the model with a fixed revision/commit hash (revision=...) when calling from_pretrained for non-local ids. Prefer vendoring the model weights/code locally and loading with local_files_only=True from a path under the repo's control, and validate model_name_or_path against an allowlist rather than passing arbitrary config strings straight to from_pretrained.

### [HIGH] s2re3-002 — security
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:301-303,314-316`
- **Evidence:** AutoTokenizer.from_pretrained (:301-304) and AutoModel.from_pretrained (:314-317) are called with trust_remote_code=True on a remote Hugging Face hub id (self.model_name_or_path resolves to 'sayby/rna_torsionbert' per README.md:53,193) whenever is_local_path() is false (:299,:312). trust_remote_code=True executes arbitrary Python shipped in that third-party model repo at load time — a remote-code-execution exposure tied to an external repo the project does not control. Not in the listed findings.

### [HIGH] s2c4l0-torsionbert-getpeftmodel-args — bug
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:381`
- **Evidence:** get_peft_model is called as get_peft_model(self.model, lora_config, self.model_name_or_path, self.lora_cfg.r, self.lora_cfg.lora_alpha, self.lora_cfg.target_modules, self.lora_cfg.bias). The PEFT signature is get_peft_model(model, peft_config, adapter_name='default', mixed=False, ...); the extra positional args map model_name_or_path->adapter_name, r(int)->mixed(bool), lora_alpha->autocast/revision, target_modules(list)/bias onto further keyword-only params. This will raise a TypeError or silently mis-bind parameters whenever LoRA is actually applied. The hyperparameters (r, alpha, target_modules, bias) are already carried by lora_config, so they must not be passed again.
- **Recommendation:** Call get_peft_model(self.model, lora_config) (optionally with a string adapter_name) and remove the trailing positional arguments.

### [HIGH] s2c4l1-002 — security
- **Location:** `rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:202-230`
- **Evidence:** TorsionBertModel.__init__ loads tokenizer and model with trust_remote_code=True: AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True) (line 202), AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) (line 209), AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True) (line 220), AutoModel.from_pretrained(model_path, trust_remote_code=True) (line 227). model_path is a free-form constructor argument (line 157) and the non-local branches (lines 207/225) pull from the HuggingFace Hub with no pinned revision, so loading executes arbitrary repository-supplied Python. Same remote-code-execution / supply-chain exposure as the StageBTorsionBertPredictor path; both code paths load the RNA TorsionBERT model and share the risk.
- **Recommendation:** Same mitigation as s2c4l1-001: disable trust_remote_code by default, gate it behind explicit opt-in, pin a revision hash for remote loads, validate/allowlist model_path, and prefer locally vendored weights loaded with local_files_only=True.

### [HIGH] s2re2-002 — security
- **Location:** `rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:204,211,222,229; rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:296-316`
- **Evidence:** AutoTokenizer.from_pretrained / AutoModel.from_pretrained are called with trust_remote_code=True for the TorsionBERT model. When model_path is not a local dir it resolves the HuggingFace hub id 'sayby/rna_torsionbert' (README.md:53,193) and trust_remote_code=True instructs transformers to download and execute arbitrary Python (modeling_*.py) from that remote repo at import time. A compromised or hijacked hub repo yields remote code execution on any inference/training host. No trust_remote_code finding exists in the current set.

### [HIGH] s2re3-003 — security
- **Location:** `rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:209-211,227-229`
- **Evidence:** Second, independent code path: AutoTokenizer.from_pretrained (:209-212) and AutoModel.from_pretrained (:227-230) also pass trust_remote_code=True for the non-local hub-id branch (else of is_local_path at :207,:225). Same arbitrary-code-execution risk on the downloaded TorsionBERT model repo as s2re3-002, in a distinct file. Not in the listed findings.

### [HIGH] s2c4l2-024 — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/protein_utils/structure_utils.py:148-162,447`
- **Evidence:** Unit mismatch: BB_BUILD_INFO['BONDANGS'] stores angles in RADIANS (sidechain_data.py:17-22: ca-c-n=2.124, c-n-ca=2.035, n-ca-c=1.939, ca-c-o=2.094). structure_utils._get_atom_placement_params reads these as 'bond_angle_deg' (:149-162, e.g. BB_BUILD_INFO[...].get('ca-c-n',116.2)) and AtomPlacementParams.to_mp_nerf_params then applies theta=np.radians(self.bond_angle_deg) (:126-130). So when the dict key is present, np.radians(2.124)=0.037 rad is used instead of 2.124 rad. _place_first_residue does the same (n_ca_c_angle_deg=BB_BUILD_INFO...get('n-ca-c',111.0); np.radians(...) at :446-447). The fallback defaults (116.2/111.0) are degrees, but the actual stored values are radians, so present-key cases yield grossly wrong bond angles.
- **Recommendation:** Treat BB_BUILD_INFO BONDANGS as radians (do not re-apply np.radians) or convert the stored constants to degrees consistently; add a unit assertion/test.

### [HIGH] s2c4l0-rna-baseangle-units — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:262-266,307-311,274,331`
- **Evidence:** Base-atom bond angles come from final_kb_rna BASE_GEOMETRY['bond_angles_deg'] (DEGREES, e.g. 105.8/120.3) and the fallback default is literally 120.0; they are converted only with torch.tensor(float(bond_angle)) (no deg->rad) and passed as the `theta` argument to calculate_atom_position(). calculate_atom_position treats theta as RADIANS (uses torch.cos(theta)/torch.sin(theta), line 89-91) and even emits '[WARN-RNAPREDICT-ANGLE-RANGE-001] Bond angle theta is outside [-pi, pi]' (rna_atom_positioning.py:47-48) for any value >pi. A 120-degree angle is therefore consumed as ~120 radians, producing geometrically wrong base coordinates.
- **Recommendation:** Convert bond angles to radians (deg_to_rad / math.radians) before passing them as theta to calculate_atom_position, in both the OP1/OP2 branch (line 262-266) and the default-placement branch (line 307-311), and for the 120.0 default.

### [HIGH] s2c5l0-001 — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:43-48`
- **Evidence:** build_rna_chain_from_internal_coords loops over every residue i and unconditionally places the 'P' atom at the origin [0.0,0.0,0.0] (line 48), then builds the remaining backbone atoms only relative to atoms WITHIN the same residue (NeRF refs at lines 62,86,121 all index residue_coords[i, ...]). No inter-residue translation/rotation is ever applied, and the residue-linking point_ref_mask produced by rna_scaffolding (which references the previous residue's C4'/C3'/O3', rna_scaffolding.py:101-108) is never consumed here. Result: all residues are superimposed at the same local frame, so the returned (num_residues, num_atoms, 3) tensor is a physically collapsed structure rather than a connected RNA chain. This contradicts Stage C's stated intent (forward-kinematics reconstruction to atomic coordinates, audit/01-understanding.md:9).
- **Recommendation:** Carry the chain frame forward: place residue i's P relative to residue i-1's O3' (or otherwise chain residues via the previous residue's terminal atoms) instead of resetting P to the origin for every residue, or document/route per-residue local frames to a downstream assembler if that is the true contract.

### [HIGH] s2c5l0-007 — bug
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:250-262,288-293`
- **Evidence:** validate_stageC_config explicitly accepts device 'auto' (line 168), and run_stageC builds a default config with device='auto' when device is None (line 459). run_stageC_rna_mpnerf then sets device = stage_cfg.device (='auto'), only WARNS for unsupported devices (line 260-261 'Proceeding anyway'), and passes device='auto' straight into build_scaffolds_rna_from_torsions and rna_fold, which call torch.zeros(..., device='auto') / torch.device('auto'). PyTorch rejects 'auto' as a device string, raising RuntimeError. So the documented/default 'auto' device crashes instead of resolving to cpu/cuda/mps.
- **Recommendation:** Resolve 'auto' to a concrete device (cuda if available else cpu/mps) before constructing any tensors; do this in validate/normalization rather than warning-and-proceeding.

### [HIGH] s2c5l2-009 — design_defect
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:96-112`
- **Evidence:** StageCReconstruction.__call__ (the 'legacy' method) returns all-zero tensors: coords=torch.zeros((N*3,3)), coords_3d=torch.zeros((N,3,3)), with empty atom_metadata. validate_stageC_config explicitly permits method=='legacy' (line 165), and run_stageC dispatches to this zero-producing path when method != 'mp_nerf' (lines 477-488). So selecting the configured 'legacy' reconstruction silently yields physically meaningless all-zero coordinates rather than reconstructed atoms, contradicting Stage C's intent of producing atomic coordinates.
- **Recommendation:** Either remove 'legacy' from the allowed methods, raise NotImplementedError when selected, or implement a real legacy reconstruction; do not return zero placeholders behind a configurable option.

### [HIGH] s2c5l0-016 — bug
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py:163-185,369-376`
- **Evidence:** Inside forward (_process_pair_features / _process_single_features), when the runtime feature dimension differs from the configured one, the module REPLACES sub-layers with freshly constructed ones: `self.layernorm_z = LayerNorm(actual_z_dim).to(...)` (line 171), `self.linear_no_bias_z = LinearNoBias(in_features=actual_z_dim, ...)` (lines 180-183), and `self.linear_no_bias_s = LinearNoBias(in_features=single_s.shape[-1], ...)` (lines 373-376). This is done in the forward pass, so each mismatching forward creates new randomly-initialized parameters that are not registered with any optimizer and discard previously trained weights, breaking training and yielding nondeterministic inference output.
- **Recommendation:** Fix tensor dimensions to the configured sizes (pad/project deterministically) instead of recreating nn layers inside forward; size layers once in __init__ and treat a runtime mismatch as a hard error.

### [HIGH] s2c5l2-027 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:99-132,950-956,1021-1028`
- **Evidence:** DiffusionModule couples production behavior to the test harness. __init__ checks os.environ['PYTEST_CURRENT_TEST'] and, for 'test_init_with_basic_config', sets a few attrs from kwargs and returns early skipping all real module construction (lines 99-132). forward() and _compute_loss() inspect the caller's frame name via get_caller_frame() and, if it contains 'test_n_sample_handling', return only coordinates / a dummy zero loss (lines 950-956, 1023-1028). Real model output depends on whether a caller function is named like a test.
- **Recommendation:** Eliminate PYTEST_CURRENT_TEST and caller-frame-name branching from the module; express these special cases via test doubles/parameters.

### [HIGH] s2c5l2-031 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:317-323,447,509-510`
- **Evidence:** multi_step_inference builds its noise schedule via _get_noise_schedule which returns torch.linspace(1.0, 0.0, num_steps+1) for BOTH the 'linear' branch and the default branch (lines 317-323) — schedule_type is effectively ignored, and the NoiseScheduleConfig parameters (s_max=160, s_min=4e-4, p=7, sigma_data) read into noise_schedule_cfg (line 447) are never used. Meanwhile generator.py provides a proper EDM InferenceNoiseScheduler (generator.py:78-141) that is never invoked for inference. The diffusion is run with a trivial 1->0 linear schedule, contradicting the AF3/EDM-inspired schedule the config describes.
- **Recommendation:** Use InferenceNoiseScheduler driven by the noise_schedule config (s_max/s_min/p/sigma_data) instead of the placeholder linspace, and honor schedule_type or remove the dead config.

### [HIGH] s2c5l0-020 — bug
- **Location:** `rna_predict/pipeline/stageD/memory_optimization/run_stageD_memory_efficient.py:10`
- **Evidence:** Imports `from rna_predict.pipeline.stageD.memory_fix import run_stageD_with_memory_fixes`, but no such module exists at that path; the file is at rna_predict/pipeline/stageD/memory_optimization/memory_fix.py (verified: `find rna_predict/pipeline/stageD -name memory_fix.py` returns only the memory_optimization/ copy, and stageD/ top level has no memory_fix.py). The argparse main entry (audit/01-understanding.md:26 lists it as a Stage D entry point) therefore fails immediately with ModuleNotFoundError and can never run.
- **Recommendation:** Fix the import to `from rna_predict.pipeline.stageD.memory_optimization.memory_fix import run_stageD_with_memory_fixes` (or relative `from .memory_fix import ...`).

### [HIGH] s2c6l0-init-add-broadcast — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:251-293`
- **Evidence:** A second, different global override of torch.Tensor.__add__ (fix_tensor_add) silently reshapes/interpolates/avg-pools operands to make additions 'succeed' (_expand_tensor_dimension uses adaptive_avg_pool1d / repeat_interleave). This masks genuine shape bugs by inventing data and changing numerics globally. Note this is a distinct implementation from tensor_operations.py's fix_tensor_add, so behavior depends on which apply path runs.
- **Recommendation:** Remove global arithmetic monkeypatching; resolve shape mismatches at their source. At minimum gate behind an explicit opt-in and never silently resample tensor values.

### [HIGH] s2c6l0-init-gather-global — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:296-316`
- **Evidence:** fix_gather_pair_embedding() replaces torch.gather globally with patched_gather(x, dim_or_idx_q, index_or_idx_k=None). The real torch.gather signature is gather(input, dim, index, *, sparse_grad=False, out=None); the wrapper drops sparse_grad/out and, in the non-int branch, ignores idx_k semantics and hard-codes dim=1 with extra unsqueezes. Any code in the process calling torch.gather with keyword args or non-trivial dims gets wrong results or TypeErrors.
- **Recommendation:** Never replace torch.gather process-wide. Provide a named helper for the pair-embedding case and call it explicitly.

### [HIGH] s2c6l0-init-module-forward — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:370-386`
- **Evidence:** fix_atom_transformer() replaces torch.nn.Module.forward (the base class method for EVERY nn.Module) with patched_forward(self, q, c, p, inplace_safe=False, chunk_size=None). apply_tensor_fixes() (line 424) calls this. The signature is specific to an atom-transformer yet is installed on the universal base class; any module relying on Module.forward (or introspection of it) is affected, and the positional contract is meaningless for general modules.
- **Recommendation:** Do not patch torch.nn.Module.forward. Patch the concrete AtomTransformer class only (as transformer_fixes.py attempts), or pass corrected tensors explicitly.

### [HIGH] s2c6l0-attn-sdpa-args — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/attention_fixes.py:19-41`
- **Evidence:** patched_attention(q,k,v,attn_bias,dropout_p,scale,dtype) calls original_scaled_dot_product_attention(q,k,v,attn_bias,dropout_p,scale,dtype) positionally. torch.nn.functional.scaled_dot_product_attention's positional order is (query,key,value,attn_mask,dropout_p,is_causal,scale,...); there is no dtype parameter. So `scale` is passed into the is_causal slot and `dtype` into the scale slot, silently producing causal masking / wrong scaling, and the replacement is installed globally (line 64).
- **Recommendation:** Match the real SDPA signature (use keyword args: attn_mask=, dropout_p=, is_causal=, scale=) and drop the non-existent dtype param; avoid global replacement.

### [HIGH] s2c6l2-011 — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/attention_fixes.py:19-64`
- **Evidence:** fix_attention_bias_shape() replaces torch.nn.functional.scaled_dot_product_attention with patched_attention(q,k,v,attn_bias=None,dropout_p=0.0,scale=None,dtype=None) (:19-21,:64). This signature does not match the real F.scaled_dot_product_attention(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,enable_gqa=False): a positional caller passing is_causal would bind it to 'scale', and extra positional args raise TypeError. The bogus 'dtype' param is not accepted by the real function either.
- **Recommendation:** If patching is unavoidable, mirror the exact upstream signature with *args/**kwargs passthrough; better, remove the patch and feed correctly-shaped masks.

### [HIGH] s2c6l2-012 — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:10-38`
- **Evidence:** tensor_operations.fix_tensor_add defines a SECOND torch.Tensor.__add__ override (:38) whose behavior contradicts the one in tensor_fixes/__init__.py: on a 'must match the size...non-singleton dimension' error it returns one operand unchanged (return self / return other, :29-32) WITHOUT performing the addition, silently producing a wrong result.
- **Recommendation:** Delete this contradictory global override; never silently drop an addition. Resolve shape mismatches at the source.

### [HIGH] s2c6l2-013 — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:41-108`
- **Evidence:** fix_matrix_multiplication patches torch.matmul/torch.bmm/torch.nn.functional.linear to truncate inner dimensions to the min and retry (lines 68-72,81-85,98-102), producing silently-wrong numeric results. The retry path calls the now-patched globals (torch.matmul/torch.bmm/F.linear) risking recursion, and the re-patch guard checks attributes (_patch_applied_safe_linear/_safe_matmul/_safe_bmm, :47-53) that are never set on the wrappers, so the guard never actually prevents re-patching.
- **Recommendation:** Remove dimension-truncating math overrides; fix dimension mismatches at the model level. If a guard is needed, actually set the sentinel attribute on the wrapper.

### [HIGH] s2c6l2-024 — design_defect
- **Location:** `rna_predict/runners/full_pipeline.py:154-171,198-244,533-552`
- **Evidence:** The orchestrator swallows broad exceptions and returns silently-fabricated outputs: Stage A failures return torch.eye identity adjacency (:151,:158,:171); Stage B init/run failures return all-zero torsion_angles/embeddings (:200-207,:217-223,:238-244); a top-level RuntimeError/AssertionError handler returns empty/dummy tensors for every key (:536-552). A consumer cannot distinguish a real prediction from a zero/identity fallback, contradicting the pipeline's purpose of producing structure predictions.
- **Recommendation:** Fail fast (or return an explicit error/status flag) instead of returning identity/zero placeholders that masquerade as valid predictions.

### [HIGH] s2c6l2-022 — intent_mismatch
- **Location:** `rna_predict/runners/full_pipeline.py:227-234,377-379`
- **Evidence:** run_full_pipeline runs Stage A (run_stage_a, :377) to produce an adjacency matrix, but run_stage_b (:379) ignores it: run_stage_b internally calls run_stageB_combined with adjacency_matrix=torch.eye(len(sequence)) (:229), a hardcoded identity. Stage A's output is only later fed to the optional latent merger (:467), not to Stage B torsion/pairformer. This contradicts the Stage-1 provisional intent of a composed A->B->C->D pipeline where Stage A's secondary structure informs Stage B.
- **Recommendation:** Thread the real adjacency from run_stage_a into run_stage_b/run_stageB_combined, or document that Stage A is not consumed by Stage B and remove the misleading composition.

### [HIGH] s2c6l1-lightning-zip-slip — security
- **Location:** `rna_predict/training/rna_lightning_module.py:173`
- **Evidence:** _unzip_file() does `with zipfile.ZipFile(zip_path,'r') as zip_ref: zip_ref.extractall(extract_dir)` with no validation of archive member names. A zip whose entries contain '../' or absolute paths (Zip Slip) will be written outside extract_dir, enabling arbitrary file overwrite. The zip is the Stage A checkpoint archive fetched from a config-supplied remote URL via _download_file (rna_lightning_module.py:204-208), so a malicious or MITM'd checkpoint_url leads to arbitrary file write on the host.
- **Recommendation:** Before extraction, validate each member: reject names that are absolute or whose normalized path escapes extract_dir (os.path.realpath join check), or use a vetted safe-extract helper. Verify archive integrity (hash/signature) before unzipping.

### [HIGH] s2c6l2-035 — intent_mismatch
- **Location:** `rna_predict/training/rna_lightning_module.py:211,365,332-336`
- **Evidence:** RNALightningModule instantiates self.stageA = StageARFoldPredictor (and downloads/extracts its checkpoint, :191-215) and registers it in the pipeline ModuleDict (:238), but forward() never calls it: adjacency comes from batch['adjacency'] (:363) and a full-file grep for self.stageA(/self.stageA.predict/predict_adjacency in this module returns no matches. Stage A is dead in the training/inference forward pass despite the pipeline claiming an A->B->C->D composition.
- **Recommendation:** Either invoke self.stageA to produce adjacency within forward, or remove Stage A from the module and document that adjacency is supplied externally.

### [HIGH] s2c6l2-036 — design_defect
- **Location:** `rna_predict/training/rna_lightning_module.py:56-64,519,687,810-817`
- **Evidence:** Production training logic branches on test identity: __init__ inspects the caller's filename for 'test_partial_checkpoint_full_pipeline.py' to set self._integration_test_mode (:56-62), and training_step repeatedly reads os.environ['PYTEST_CURRENT_TEST'] to alter control flow ('test_noise_and_bridging_runs' :519,:687; 'test_run_stageD_basic' :813). This couples shipped behavior to specific test names and makes runtime behavior depend on the test harness.
- **Recommendation:** Drive test-only behavior via explicit constructor flags/config, not by sniffing caller filenames or PYTEST_CURRENT_TEST in production code paths.

### [HIGH] s2c6l2-033 — bug
- **Location:** `rna_predict/training/train.py:19-20,217`
- **Evidence:** Both @hydra.main decorators hardcode config_path='/Users/tomriddle1/RNA_PREDICT/rna_predict/conf' (:20 and :217), an absolute developer-specific path that will not resolve on any other machine, breaking training/CI. The immediately preceding comment '# Use a relative config path instead of absolute' (:19) directly contradicts the code.
- **Recommendation:** Use a path relative to the file (e.g. config_path='../conf') or Hydra search-path/ConfigStore; honor the comment that already states the intent.

### [HIGH] s2c6l0-train-abs-config — bug
- **Location:** `rna_predict/training/train.py:20,217`
- **Evidence:** Both @hydra.main decorators hardcode config_path='/Users/tomriddle1/RNA_PREDICT/rna_predict/conf' (a developer-specific absolute path), despite the comment 'Use a relative config path instead of absolute' (line 19). On any other machine Hydra cannot find the config dir and training fails to launch.
- **Recommendation:** Use config_path relative to the module (e.g. '../conf') or derive from PROJECT_ROOT/importlib.resources.

### [HIGH] s2c6l2-034 — bug
- **Location:** `rna_predict/training/train.py:20,217-219`
- **Evidence:** execute_training_run is itself decorated with @hydra.main (:20), and main() (also @hydra.main, :217) calls execute_training_run(cfg) (:219). Invoking a @hydra.main-wrapped function re-enters Hydra initialization and ignores the cfg passed in. The Stage-1 inventory also notes the Kaggle harness calls execute_training_run programmatically, which would trigger the same nested-Hydra problem.
- **Recommendation:** Have a single @hydra.main entry point delegate to an undecorated implementation function (e.g. _execute_training_run(cfg)); expose that plain function for programmatic callers.

### [HIGH] s2c6l0-train-nested-hydra — bug
- **Location:** `rna_predict/training/train.py:20-22,217-219`
- **Evidence:** execute_training_run is itself decorated with @hydra.main (line 20). main (also @hydra.main, line 217) calls execute_training_run(cfg) directly (line 219). Invoking a @hydra.main-wrapped callable re-enters Hydra initialization (re-parses sys.argv / re-instantiates Hydra), which errors ('Hydra is already initialized' / argv reparsing). The Kaggle harness also imports and calls execute_training_run, hitting the same double-initialization.
- **Recommendation:** Split the logic: a plain function body (no decorator) containing the work, wrapped by a single @hydra.main entry. Have both main and external callers invoke the undecorated function.

### [HIGH] s2c6l2-046 — design_defect
- **Location:** `rna_predict/utils/shape_utils.py:186-209,231-236`
- **Evidence:** ensure_consistent_sample_dimensions (imported into production by stageD/diffusion/run_stageD_unified.py) reads os.environ['PYTEST_CURRENT_TEST'] and special-cases tensor expansion for named tests 'test_single_sample_shape_expansion'/'test_multi_sample_shape_fix' (:189-190,:200-209,:232-236). Shipped tensor-shape behavior thus differs depending on whether a specific pytest is running.
- **Recommendation:** Remove PYTEST_CURRENT_TEST branches; make sample-dimension handling deterministic and identical in and out of tests.

### [HIGH] s2c6l0-analyze-token — security
- **Location:** `scripts/analysis/analyze_code.sh:130-135`
- **Evidence:** A CodeScene access token is hardcoded and exported in the script (CS_ACCESS_TOKEN default value at line 132). A live credential is committed to the repository in plaintext; anyone with repo access obtains it, and it cannot be rotated without a code change. (Secret referenced by location/category only, not reproduced here.)
- **Recommendation:** Remove the embedded token, require it via environment/secret manager, and rotate the leaked credential.

### [HIGH] s2c6l1-analyze-hardcoded-token — security
- **Location:** `scripts/analysis/analyze_code.sh:132`
- **Evidence:** The script hardcodes and exports a default CodeScene access token (category: third-party API access credential) directly in source when CS_ACCESS_TOKEN is unset (analyze_code.sh:130-135). A committed credential is exposed to anyone with repo read access and remains valid until revoked; it grants CodeScene CLI 'refactor.access'/'cli.access' per the embedded token claims.
- **Recommendation:** Remove the literal token from source, require CS_ACCESS_TOKEN to be supplied via the environment/secret store, and rotate/revoke the leaked token since it is already committed to history.

### [HIGH] s2c7l0-003 — bug
- **Location:** `scripts/dev.js:16`
- **Evidence:** `import { runCLI } from './modules/commands.js';` targets scripts/modules/commands.js, but the scripts/modules/ directory does not exist (corroborated by audit/01-understanding.md:14, entry-point table, which notes the missing target). Every npm script that maps to `node scripts/dev.js` (dev/list/generate/parse-prd per package.json) crashes at module resolution time.
- **Recommendation:** Restore/ship the scripts/modules/ package or repoint the import to the actual CLI implementation; otherwise remove the dead npm scripts and dev.js to avoid a broken declared entrypoint.

### [HIGH] s2c7l0-004 — bug
- **Location:** `scripts/partial_checkpoint_full_pipeline_script.py:67`
- **Evidence:** `with hydra.initialize(config_path="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", ...)`. hydra.initialize requires a path RELATIVE to the calling module (absolute paths must use initialize_config_dir), and this absolute path is a developer-specific machine path that does not exist on any other host. The script earlier computes a correct relative `config_path_selected` (lines 42-62) but then ignores it and hardcodes the absolute path, so initialization fails on every non-author machine (caught at lines 71-73 -> sys.exit(1)).
- **Recommendation:** Use the already-computed relative `config_path_selected` with hydra.initialize, or switch to hydra.initialize_config_dir(config_dir=str(config_path.resolve())) for an absolute path.

### [HIGH] s2c7l2-009 — bug
- **Location:** `scripts/run_all_pipeline.py:6-7`
- **Evidence:** PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__)))) (:7) with comment 'assuming this script is in <root>/rna_predict/scripts' (:6). But reorganize_scripts.sh moved this file to the top-level scripts/ (line 33), only TWO levels below root. From /home/user/RNA_PREDICT/scripts/run_all_pipeline.py the triple-dirname yields /home/user (one level too high). Consequently every entry in python_files (e.g. os.path.join(PROJECT_ROOT,'rna_predict/pipeline/stageA/run_stageA.py'), :61-66) resolves under /home/user/rna_predict/... which does not exist, so main() skips all files as 'File not found' (:83-84) and cwd=PROJECT_ROOT (:25) is also wrong.
- **Recommendation:** Update the path computation to two levels (PROJECT_ROOT = dirname(dirname(abspath(__file__)))) and fix the stale comment to reflect the scripts/ location.

### [HIGH] s2c7l0-005 — bug
- **Location:** `scripts/run_all_pipeline.py:7`
- **Evidence:** `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` assumes the script lives at <root>/rna_predict/scripts/ (comment at line 6), but the file now resides at top-level scripts/. Verified resolution: for scripts/run_all_pipeline.py PROJECT_ROOT becomes /home/user (one directory ABOVE the repo root /home/user/RNA_PREDICT). All targets built as os.path.join(PROJECT_ROOT, 'rna_predict/pipeline/...') (lines 61-66) then point at /home/user/rna_predict/... which does not exist, so every stage is reported 'File not found' and skipped (lines 83-91).
- **Recommendation:** Compute PROJECT_ROOT with two dirname() calls (scripts/ -> repo root) or anchor on a marker, e.g. `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.

### [HIGH] s2c7l2-001 — bug
- **Location:** `scripts/test_utils/batch_test_generator.py:4`
- **Evidence:** Imports `from rna_predict.scripts.hypot_test_gen import run_test_generation`, but rna_predict/scripts/hypot_test_gen.py only defines `remove_logger_lines` and `fix_leading_zeros` (verified via grep -n 'def '); it has no `run_test_generation`. The real `run_test_generation` lives in the sibling file scripts/test_utils/hypot_test_gen.py:836. The import therefore raises ImportError at module load, so the script (which __main__-guards main() at :60) cannot run at all. Root cause is reorganize_scripts.sh moving batch_test_generator.py to scripts/test_utils/ (line 10) without updating its import target.
- **Recommendation:** Change the import to `from scripts.test_utils.hypot_test_gen import run_test_generation` (or a relative `from .hypot_test_gen import ...`) pointing at the file that actually defines the function.

### [HIGH] s2c7l0-007 — bug
- **Location:** `scripts/test_utils/batch_test_generator.py:4-6`
- **Evidence:** Module-level `from rna_predict.scripts.hypot_test_gen import run_test_generation`. Verified that rna_predict/scripts/hypot_test_gen.py defines only remove_logger_lines (line 8) and fix_leading_zeros (line 29) and NOT run_test_generation (grep returned no match). The full run_test_generation/TestGenerator implementation lives instead in scripts/test_utils/hypot_test_gen.py. Therefore this import raises ImportError at load time and the script cannot run at all.
- **Recommendation:** Import from the sibling module that actually defines it, e.g. `from scripts.test_utils.hypot_test_gen import run_test_generation` (or a relative `from .hypot_test_gen import run_test_generation`), and fix the stale rna_predict.scripts package reference.

### [HIGH] s2c7l2-005 — bug
- **Location:** `scripts/test_utils/mark_slow_tests.py:69-72`
- **Evidence:** add_slow_marker() guards `if modified:` (:69) BEFORE running the transformer that sets `modified` — `SlowTestMarker().visit(tree)` (:70) and the `nonlocal modified; modified = True` (:65-66) are inside that very block. Since `modified` is initialized False (:33) and nothing flips it beforehand, the block never executes, the visitor never runs, and no file is ever written. The script's sole purpose (inserting @pytest.mark.slow) is dead.
- **Recommendation:** Run the transformer unconditionally first (e.g. `tree = SlowTestMarker().visit(tree)`), then write the file only if `modified` became True.

### [HIGH] s2c7l0-006 — bug
- **Location:** `scripts/test_utils/mark_slow_tests.py:69-73`
- **Evidence:** add_slow_marker sets `modified = False` (line 33), defines SlowTestMarker whose visit_FunctionDef sets `nonlocal modified = True` and inserts the marker, but then guards the actual transform with `if modified:` (line 69) BEFORE invoking the visitor (`tree = SlowTestMarker().visit(tree)` is on line 70, inside that block). Since modified is False at line 69, the block is skipped, the visitor never runs, modified is never set, and the file is never written. The script is a permanent no-op and never adds @pytest.mark.slow to any test.
- **Recommendation:** Run the transformer first, then check the flag: `marker = SlowTestMarker(); tree = marker.visit(tree); if modified: ast.fix_missing_locations(tree); write`. Capture `modified` from the visitor instance rather than a closure guarded before the visit.

### [HIGH] s2c7l2-007 — bug
- **Location:** `scripts/test_utils/run_failing_tests.sh:319`
- **Evidence:** COVERAGE_GOAL is captured from get_coverage_goal() (which echoes many lines plus a final bc value like '80.50') then re-extracted via `grep -o '[0-9]\+' | tail -1`. Because grep -o splits '80.50' into separate matches '80' and '50', tail -1 returns the fractional part. Verified: printf '80.50' | grep -o '[0-9]\+' | tail -1 => 50; for a scale=2 value like '80.00' it returns '00' => 0. The wrong value is then passed to `--cov-fail-under=$COVERAGE_GOAL` (:409), so the coverage gate that is this script's whole purpose is set to the decimal digits (often 0), effectively disabling enforcement.
- **Recommendation:** Have get_coverage_goal emit ONLY the numeric goal on stdout (route diagnostics to stderr), and parse with a decimal-aware pattern, e.g. `grep -oE '[0-9]+(\.[0-9]+)?' | tail -1`.

### [HIGH] s2c7l2-012 — design_defect
- **Location:** `setup.py:8-21`
- **Evidence:** setup.py install_requires omits the project's core runtime deps — no hydra-core, no lightning/pytorch-lightning, no omegaconf — even though Hydra @main and PyTorch Lightning are central (pyproject.toml lists hydra-core==1.3.2 at :32 and lightning>=2.2 at :35). Installing via setup.py would leave `import hydra`/Lightning failing. Conversely it pins GUI/vendored-tool deps as CORE requirements: opencv-python, Pillow, mss, pyautogui, and PySimpleGUI (:16-20), the latter not even used (gui_launcher.py imports dearpygui, which setup.py omits while pyproject.toml lists dearpygui>=1.10 at :31).
- **Recommendation:** Make setup.py match pyproject.toml: add hydra-core/lightning/omegaconf, drop unused PySimpleGUI, and move screen_finder GUI deps to an optional extra rather than core install_requires (or remove setup.py entirely).

### [MEDIUM] s2c0l2-0021 — intent_mismatch
- **Location:** `.augement_code_rules:1`
- **Evidence:** The file is entirely a Task Master AI dev-workflow rulebook centered on `scripts/dev.js` and its supposed modular split into `scripts/modules/` (e.g. lines 312-321), but scripts/modules/ does not exist (verified by ls) so the documented `node scripts/dev.js`/`task-master` workflow is non-functional. The content is unrelated to the RNA prediction pipeline that is the project's stated intent (audit/01-understanding.md:6).
- **Recommendation:** Remove or relocate the Task Master rules; if retained, fix references to the missing scripts/modules/ structure.

### [MEDIUM] s2c0l2-0016 — design_defect
- **Location:** `.coveragerc:11`
- **Evidence:** .coveragerc sets `[report] fail_under = 0`, meaning coverage enforcement is effectively disabled, yet .coverage_config.json declares base_coverage 80, max_coverage 95, current_coverage 89.99 and a phased target schedule (.coverage_config.json:2-39). The governance described in .coverage_config.json is never enforced by the actual coverage tool config.
- **Recommendation:** Either wire .coverage_config.json thresholds into CI/.coveragerc fail_under, or remove the unenforced coverage-config JSON to avoid implying a gate that does not exist.

### [MEDIUM] s2c0l0-009 — intent_mismatch
- **Location:** `.coveragerc:11`
- **Evidence:** `[report] fail_under = 0` disables any coverage gate, while .coverage_config.json:3-5 declares `base_coverage: 80`, `current_coverage: 89.99`, `max_coverage: 95` and phase targets up to 95/98 (.coverage_config.json:31-38), and .windsurfrules:24 states 'Aim for near 100% test coverage'. The actual enforced threshold (0) contradicts the documented coverage policy, so coverage can drop arbitrarily without CI failing.
- **Recommendation:** Set fail_under to the intended floor (e.g. 80) or wire the phase-based thresholds from .coverage_config.json into the coverage gate; otherwise the coverage policy is unenforced.

### [MEDIUM] s2c0l2-0020 — intent_mismatch
- **Location:** `.env.example:1-14`
- **Evidence:** .env.example documents only Task Master / LLM-CLI variables (ANTHROPIC_API_KEY, PERPLEXITY_API_KEY, MODEL=claude-3-7-sonnet-20250219, MAX_TOKENS, DEFAULT_SUBTASKS, etc.) — the vendored Node 'Task Master' tool's config — and contains zero variables relevant to the RNA structure-prediction pipeline. A new user copying this file gets no guidance for the actual product.
- **Recommendation:** Replace with env vars the RNA pipeline actually reads (e.g. HF token / cache, device, data paths), or move this file under the Task Master tooling and label it as such.

### [MEDIUM] s2re4-006 — design_defect
- **Location:** `.gitattributes:1 (DNA_bert_3.zip)`
- **Evidence:** .gitattributes declares '*.zip filter=lfs diff=lfs merge=lfs -text', and DNA_bert_3.zip is committed as a Git-LFS pointer (on-disk content is the 134-byte text 'version https://git-lfs.github.com/spec/v1 ... size 321782167', a 321MB object). Cloning without git-lfs installed yields the pointer stub instead of the model archive, and all *.zip assets (root DNA-BERT model plus the rna_predict/dataset/preprocessing DSSR binary zips) silently fail to materialize. No README/CONTRIBUTING/setup instruction documents the git-lfs requirement, so preprocessing/model loading breaks with confusing errors for fresh clones/CI.

### [MEDIUM] s2c0l1-001 — security
- **Location:** `.github/init.sh:28-63`
- **Evidence:** download_template() runs `git clone "${template_url}"` where template_url="https://github.com/rochacbruno/${template}-project-template" (:35) with `template` taken from the -t flag or an interactive read (:7,:14). Line 63 then executes the just-downloaded code unconditionally: `./.github/templates/${template}/apply.sh -a ... -d ...`. This is download-and-execute of remote, third-party code over the network with no integrity/pin (no commit SHA, no checksum). The `template` value is also interpolated directly into both the clone URL and the executed path, so a crafted value (e.g. containing path traversal or `;`) flows into the executed path. Invoked via `make init` (Makefile:120-122).
- **Recommendation:** Pin the template source to a specific commit/tag and verify it, restrict `template` to an allow-list (currently only 'flask' is advertised at :13), validate/sanitize the value before using it in URLs or paths, and require explicit confirmation before executing downloaded apply.sh.

### [MEDIUM] s2c0l0-005 — design_defect
- **Location:** `.github/workflows/main.yml:50-70`
- **Evidence:** Two near-identical ruff auto-fix steps ('Run ruff check (auto-fix)' and 'Fix auto-fixable lint issues') run the same `ruff check --fix --unsafe-fixes` then commit and push to the branch. Both run on `pull_request` events (main.yml:10-11); auto-committing/force-style pushing during CI mutates the branch under test, and pushes from fork PRs will fail (masked by `continue-on-error: true`), so failures are silently swallowed and the duplication is wasted work.
- **Recommendation:** Collapse to a single fix step, gate the commit/push on push-to-main (not pull_request), and drop continue-on-error so genuine failures surface.

### [MEDIUM] s2c0l2-0007 — design_defect
- **Location:** `.github/workflows/rename_project.yml:3`
- **Evidence:** The 'Rename the project from template' workflow triggers `on: [push]` with `permissions: write-all` and force-pushes a commit on every push (it is gated only by the presence of .github/template.yml). This is leftover python-project-template scaffolding unrelated to the RNA pipeline; if .github/template.yml ever reappears it would sed-rewrite and force-push every tracked file (.github/rename_project.sh:24-33).
- **Recommendation:** Delete the rename_project workflow and rename_project.sh now that the project is no longer a template, or restrict the trigger to workflow_dispatch only.

### [MEDIUM] s2c0l0-007 — bug
- **Location:** `.github/workflows/rename_project.yml:30`
- **Evidence:** The 'Is this still a template' step uses `echo "::set-output name=is_template::..."`. The `::set-output` workflow command was deprecated and disabled by GitHub Actions (mid-2023); it no longer sets step outputs. Consequently `steps.is_template.outputs.is_template` (referenced at rename_project.yml:33) is always empty, so the rename step never fires. The workflow also triggers `on: [push]` (line 3) for every push to this already-renamed repo, and the git-auto-commit-action with `push_options: --force` (lines 38-42) runs on every push regardless.
- **Recommendation:** Replace `::set-output` with `echo "is_template=..." >> "$GITHUB_OUTPUT"`. Since the project is no longer a template, consider deleting rename_project.yml/.sh entirely to remove the force-push-on-every-push hazard.

### [MEDIUM] s2c0l1-003 — security
- **Location:** `.github/workflows/rename_project.yml:5,3,38-43`
- **Evidence:** `permissions: write-all` (:5) grants the GITHUB_TOKEN full write scope to all repository resources, and the workflow triggers on every push (`on: [push]`, :3). The final step always runs stefanzweifel/git-auto-commit-action@v5 with `push_options: --force` (:42), force-pushing a commit on every push. The rename body is gated on `.github/template.yml` existing (:33), and that file is absent (verified: `.github/template.yml` does not exist), so the broad-permission + force-push step now fires on every push with no useful work — an over-privileged, surprising mutation surface.
- **Recommendation:** Remove this template-bootstrap workflow now that the project is no longer a template (template.yml is gone), or scope `permissions` to the minimum (`contents: write` only) and drop the unconditional `--force` auto-commit.

### [MEDIUM] s2c0l2-0035 — design_defect
- **Location:** `.gitignore:220-221`
- **Evidence:** .gitignore lists `package-lock.json` and `package.json` as ignored, but package.json is a tracked, inventoried config file (audit/01-understanding.md:175) defining the Task Master npm scripts. Ignoring a file that is intentionally version-controlled is contradictory and will cause newly-introduced copies to be silently skipped.
- **Recommendation:** Remove package.json (and package-lock.json if it is meant to be tracked) from .gitignore, or untrack them deliberately.

### [MEDIUM] s2c0l2-0023 — doc_drift
- **Location:** `.roomodes:6`
- **Evidence:** The Test mode roleDefinition instructs running `./rna_predict/scripts/run_failing_tests.sh` 'its way faster', but that path does not exist; the actual script is at scripts/test_utils/run_failing_tests.sh (verified: find run_failing_tests.sh → ./scripts/test_utils/run_failing_tests.sh, and rna_predict/scripts/ contains only __init__.py and hypot_test_gen.py). Following the instruction yields 'No such file or directory'.
- **Recommendation:** Correct the path to scripts/test_utils/run_failing_tests.sh.

### [MEDIUM] s2c0l2-0025 — design_defect
- **Location:** `.windsurfrules:6-7`
- **Evidence:** The project rules record a known unresolved pipeline defect: 'inconsistent atom counts between stages, with Stage C producing 21 atoms total while Stage D expects 44 atoms per residue.' This documents a real intent/contract mismatch between Stage C output and Stage D input that, per the project's own notes, breaks the C->D handoff in the full pipeline.
- **Recommendation:** Reconcile the atom-count contract between Stage C reconstruction output and Stage D diffusion input (verify against stageC/stageD source) and update or close out this rule once fixed.

### [MEDIUM] s2c0l1-005 — security
- **Location:** `Containerfile:1`
- **Evidence:** `FROM python:3.7-slim`. Python 3.7 reached end-of-life in June 2023 and no longer receives security patches; the base image carries unpatched OS/interpreter CVEs. It also contradicts intended runtime: pyproject.toml:11 sets `requires-python = ">=3.10"`, so `pip install .` (Containerfile:4) would fail on this base anyway.
- **Recommendation:** Use a supported, patched base image consistent with requires-python (e.g. python:3.11-slim) and rebuild regularly to pick up security updates.

### [MEDIUM] s2c0l0-014 — design_defect
- **Location:** `Makefile:47-48`
- **Evidence:** `test: lint` makes the test target depend on the `lint` target, which runs `ruff check --fix --unsafe-fixes rna_predict/ tests/` and `mypy --ignore-missing-imports rna_predict/` (Makefile:33-35). Both return non-zero on unfixable lint issues / type errors, so `make test` aborts before any test runs. CI invokes `make test` (.github/workflows/main.yml:101), so a mypy/ruff finding fails the test job for reasons unrelated to test outcomes, and `--unsafe-fixes` mutates source as a side effect of running tests.
- **Recommendation:** Decouple linting from testing: have `test` run pytest only, and run lint as a separate CI step/target (the linter job already exists in main.yml).

### [MEDIUM] s2c0l2-0013 — design_defect
- **Location:** `mypy.ini:1`
- **Evidence:** mypy configuration is split across two files: mypy.ini (full config: python_version, per-module overrides for deepspeed/protenix/scipy/torch/numpy) and pyproject.toml [tool.mypy] (overrides for cv2 and dearpygui at pyproject.toml:94-103). mypy reads mypy.ini in preference to pyproject.toml when both exist, so the cv2/dearpygui ignore_missing_imports overrides in pyproject are silently ignored, allowing missing-import errors for those modules.
- **Recommendation:** Consolidate all mypy config into one file (move the cv2/dearpygui overrides into mypy.ini, or delete mypy.ini and keep everything in pyproject [tool.mypy]).

### [MEDIUM] s2c0l0-018 — bug
- **Location:** `package.json:2-7`
- **Evidence:** All npm scripts (dev/list/generate/parse-prd) invoke `node scripts/dev.js`, but scripts/dev.js imports from './modules/commands.js' while `scripts/modules/` does not exist (verified: `ls scripts/modules` -> 'modules missing'; corroborated by audit/01-understanding.md:14). Every declared npm script therefore fails at runtime with a module-resolution error.
- **Recommendation:** Restore the missing scripts/modules/ directory or repoint dev.js, or remove the dead npm scripts/Task Master tooling if it is no longer used.

### [MEDIUM] s2c0l2-0037 — bug
- **Location:** `package.json:3-6`
- **Evidence:** npm scripts (dev/list/generate/parse-prd) all shell out to `node scripts/dev.js`, and scripts/dev.js imports from './modules/commands.js' (audit/01-understanding.md:14) but scripts/modules/ does not exist (verified by ls). Every npm script therefore fails at runtime with a module-not-found error. The whole package.json belongs to the vendored Task Master tool, unrelated to the RNA pipeline.
- **Recommendation:** Restore the missing scripts/modules/ tree or remove the Task Master Node tooling (package.json + dev.js) if it is not part of the shipped product.

### [MEDIUM] s2c0l0-010 — design_defect
- **Location:** `pyproject.toml:14-38`
- **Evidence:** Core runtime `dependencies` include unrelated GUI/automation packages for the vendored screen_finder_app (pyautogui>=0.9.54, opencv-python>=4.8.0, mss>=9.0.1, dearpygui>=1.10, Pillow>=10.0.0) and developer tooling (black>=25.1.0, isort>=6.0.1, ruff>=0.11.2, pytest>=8.3.5) as hard install requirements. Per the Stage-1 intent (audit/01-understanding.md:9), screen_finder is 'vendored tooling unrelated to RNA structure prediction'. Every consumer of the rna-predict package is forced to install heavyweight GUI stacks and lint/test tools at runtime.
- **Recommendation:** Move dev tools to the dev optional-dependencies group and the GUI/automation packages to a dedicated optional extra (e.g. [project.optional-dependencies].screenfinder); keep core dependencies to true runtime needs.

### [MEDIUM] s2c0l2-0014 — design_defect
- **Location:** `pytest.ini:1`
- **Evidence:** pytest configuration is split: pytest.ini provides testpaths/markers/addopts, while pyproject.toml [tool.pytest.ini_options] (pyproject.toml:88-90) sets asyncio_mode="strict" and asyncio_default_fixture_loop_scope. pytest uses pytest.ini in preference to pyproject when both exist, so the asyncio settings are ignored — pytest-asyncio (a declared dependency) will not run in strict mode as intended.
- **Recommendation:** Move the asyncio_mode/loop-scope settings into pytest.ini, or drop pytest.ini and keep all pytest config in pyproject.toml.

### [MEDIUM] s2c0l0-011 — doc_drift
- **Location:** `requirements.txt:1-21`
- **Evidence:** requirements.txt and pyproject.toml [project.dependencies] are divergent, conflicting sources of truth: requirements.txt pins `torch>=2.0.0` (no upper bound) vs pyproject.toml:25 `torch>=2.0.1, <2.6.0`; `transformers>=4.30.0` vs pyproject.toml:26 `transformers>=4.49.0`; `biopython>=1.81` vs pyproject.toml:15 `biopython>=1.83`. requirements.txt also lists packages absent from pyproject (PySimpleGUI, py-cpuinfo, psutil, pandas, tqdm, datasets-less) while pyproject lists datasets/einops/lxml/mdanalysis/tensorboard absent from requirements.txt. A `pip install -r requirements.txt` yields a materially different environment than `pip install .`.
- **Recommendation:** Designate one canonical dependency source (pyproject.toml) and either delete requirements.txt or generate it from the project metadata; reconcile the version bounds.

### [MEDIUM] s2c0l2-0027 — intent_mismatch
- **Location:** `requirements.txt:14-18`
- **Evidence:** requirements.txt lists GUI/automation packages pyautogui, opencv-python, Pillow, mss, PySimpleGUI as runtime dependencies of the RNA pipeline. These belong to the unrelated vendored scripts/screen_finder_app GUI (audit/01-understanding.md:9,36), not to sequence-to-structure prediction, bloating and risking the install (PySimpleGUI now requires a license server).
- **Recommendation:** Move screen-finder GUI deps into an optional extra (e.g. [screen_finder]) and keep the core requirements limited to pipeline needs.

### [MEDIUM] s2c0l0-012 — bug
- **Location:** `requirements.txt:18`
- **Evidence:** `PySimpleGUI` is listed as an unpinned dependency. PySimpleGUI was relicensed and the open-source releases were removed from PyPI (the package now requires a private server / paid key), so `pip install -r requirements.txt` is liable to fail or pull an incompatible release on a clean machine. It is also a GUI dependency unrelated to the RNA pipeline.
- **Recommendation:** Remove PySimpleGUI from runtime requirements (or pin a known-installable version and move it to an optional extra) since it is only relevant to the unrelated GUI tooling.

### [MEDIUM] s2c0l2-0028 — design_defect
- **Location:** `requirements.txt:2`
- **Evidence:** Dependency declarations are fragmented and divergent across four+ sources of truth: requirements.txt (torch>=2.0.0 unbounded, PySimpleGUI, protenix unpinned), setup.py install_requires (torch>=2.0.0,<2.6.0, PySimpleGUI, version 1.0.0), pyproject.toml dependencies (torch>=2.0.1,<2.6.0, dearpygui instead of PySimpleGUI, protenix>=0.4.4), and pyproject [project.optional-dependencies].dev plus a separate [dependency-groups].dev with different contents. The GUI lib even differs (PySimpleGUI vs dearpygui).
- **Recommendation:** Pick pyproject.toml as the single source of dependency truth; delete setup.py/requirements*.txt or generate them from pyproject, and reconcile the GUI library choice.

### [MEDIUM] s2c1l0-validate-metadata-noop — design_defect
- **Location:** `rna_predict/conf/config_schema.py:222-227,435-441,356-362`
- **Evidence:** Numerous dataclass fields declare metadata={'validate': lambda x: ...} (e.g. StageAConfig.dropout 222-227, PairformerBlockConfig.dropout 435-441, LoRAConfig.dropout 356-362, many others). OmegaConf/dataclasses never invoke metadata['validate'], so these range checks are decorative and never enforced. Only StageAConfig, DeviceConfig and StageCConfig implement real __post_init__ validation; all other 'validate' metadata gives a false impression of input validation (e.g. out-of-range dropout/heads pass silently).
- **Recommendation:** Either remove the misleading 'validate' metadata or implement __post_init__ checks (or a shared validator) that actually call these predicates.

### [MEDIUM] s2c1l1-003 — security
- **Location:** `rna_predict/conf/config_schema.py:270-272`
- **Evidence:** StageAConfig.checkpoint_url defaults to a hardcoded third-party Dropbox URL ('https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1'), duplicated in conf/model/stageA.yaml:12, with checkpoint_zip_path defaulting to 'RFold/checkpoints.zip' and checkpoint_path to a '.pth' file (lines 266-277). Per the audit map (01-understanding.md:22) Stage A downloads/extracts this checkpoint and loads it; a PyTorch '.pth' is an unpickled artifact, so an unauthenticated download from an externally-controlled file-host (no checksum/signature pinning) is an unsafe-download / supply-chain vector that can lead to arbitrary code execution on torch.load of the downloaded weights.
- **Recommendation:** Pin the checkpoint by content hash and verify it after download; prefer a versioned, integrity-checked source; load weights with weights_only=True (torch>=2.0) and document the trust boundary. Do not ship a mutable third-party share link as the default.

### [MEDIUM] s2c1l0-devmgmt-unregistered-resolver — bug
- **Location:** `rna_predict/conf/device_management/default.yaml:2`
- **Evidence:** primary: ${device:cpu} uses OmegaConf custom-resolver syntax (resolver name 'device', argument 'cpu'). No OmegaConf.register_new_resolver call exists anywhere in the repository (grep for register_new_resolver returned no matches), so resolving this node raises UnsupportedInterpolationType. The other YAMLs use plain interpolation ${device} which references the top-level device key and is fine; this colon form is a latent failure for any composition that pulls in device_management.
- **Recommendation:** Change to plain interpolation ${device} (with a default elsewhere) or register a 'device' resolver; otherwise selecting the device_management group crashes config resolution.

### [MEDIUM] s2re1-004 — security
- **Location:** `rna_predict/conf/model/stageA.yaml:12`
- **Evidence:** checkpoint_url is a personal Dropbox share link: "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1". run_stageA.py:188-190 fetches this via download_file() (urllib.request.urlopen, run_stageA.py:70) and unzips it; the resulting checkpoint is then torch.load-ed (unsafe, see s2re1-002). A mutable, unauthenticated, non-content-addressed third-party URL as the canonical model-weights source is a supply-chain/single-point-of-failure risk: the owner can delete/replace it and there is no hash pinning to detect tampering.

### [MEDIUM] s2c1l2-stageb-pairformer-internal-dim-inconsistency — design_defect
- **Location:** `rna_predict/conf/model/stageB_pairformer.yaml:8-32`
- **Evidence:** stageB_pairformer.yaml mixes production and toy dimensions within one file in mutually inconsistent ways: top-level c_token=384, c_atom=128, c_pair=32 (production) but c_z=2 (toy) and c_s=0; the nested protenix_integration block then overrides c_token=2, restype_dim=2, profile_dim=2, c_atom=2, c_pair=2 — i.e. the same logical embedding (c_token) is 384 at the Pairformer level and 2 in its ProtenixIntegration sub-config. config_schema.py PairformerConfig defaults c_token=8/c_atom=4 again differ. A leftover marker '[UNIQUE-ERR-STAGEB-DEBUGLOGGING-001]' (line 93) further signals ad-hoc debug edits left in the config.
- **Recommendation:** Reconcile the Pairformer dims so token/atom/pair sizes are consistent between the block and its protenix_integration sub-config; remove the debug marker comment; document why c_s=0 is intentional or drop it.

### [MEDIUM] s2c1l2-stagec-angle-representation-semantics — doc_drift
- **Location:** `rna_predict/conf/model/stageC.yaml:18-19 vs rna_predict/conf/config_schema.py:767-770`
- **Evidence:** The field 'angle_representation' is documented with two incompatible meanings. config_schema.py StageCConfig.angle_representation defaults to 'cartesian' with help 'Angle representation: cartesian or internal'. stageC.yaml sets angle_representation: 'degrees' with comment 'Expected input format (\'degrees\' or \'radians\')'. The YAML value 'degrees' is not even a member of the schema's documented value set {cartesian, internal}, so the structured-schema documentation and the actual config describe different concepts (coordinate representation vs angle units).
- **Recommendation:** Decide what angle_representation actually controls, unify the help text, and constrain the value set (e.g. an Enum or __post_init__ validator) so 'degrees' vs 'cartesian' cannot silently coexist.

### [MEDIUM] s2c1l2-stageD-arch-layer-count-inconsistency — design_defect
- **Location:** `rna_predict/conf/model/stageD.yaml:34-35,87-88 vs rna_predict/conf/model/stageD_diffusion.yaml:15-23`
- **Evidence:** stageD.yaml hardcodes model_architecture.num_layers: 6 and num_heads: 8 and test_residues_per_batch: 25 (production scale) while every embedding dimension it pulls from stageD_diffusion is the toy size (c_token=8, c_s=8, c_z=4, c_atom=4; stageD_diffusion.yaml:16-22) and stageD_diffusion's own transformer is n_blocks: 2 / n_heads: 2 (lines 62-63). The result is an internally contradictory Stage D config: an 8-dim token model asked to run 6 layers / 8 heads in one place but 2 blocks / 2 heads in another, with num_heads=8 not dividing several of the reduced dims cleanly.
- **Recommendation:** Source num_layers/num_heads/test_residues_per_batch from the same stageD_diffusion block via interpolation (as the dims are) rather than hardcoding production-scale literals, so the architecture stays self-consistent.

### [MEDIUM] s2c1l0-predict-yaml-hardcoded-ckpt — bug
- **Location:** `rna_predict/conf/predict.yaml:13`
- **Evidence:** predict.yaml sets checkpoint_path: /Users/tomriddle1/RNA_PREDICT/outputs/checkpoints/last.ckpt — an absolute developer-specific path that does not exist on any other machine. The README-recommended inference entry (predict.py) composes this config, so a default run loads a nonexistent checkpoint path.
- **Recommendation:** Use a relative path (e.g. outputs/checkpoints/last.ckpt) or null with an explicit override requirement.

### [MEDIUM] s2c1l2-testdata-embedding-dims-stale — intent_mismatch
- **Location:** `rna_predict/conf/test_data.yaml:25-28`
- **Evidence:** test_data.yaml embedding_dims sets s_trunk: 384, z_trunk: 128, s_inputs: 449 with comment 's_inputs ... must match c_s_inputs', but the reduced schema/YAMLs put c_s_inputs at 8 (config_schema.py:820-823; stageD_diffusion.yaml:18). So the asserted invariant 'must match c_s_inputs' is violated (449 vs 8), and these production-size test dims contradict the toy model dims used everywhere else, making the test fixture inconsistent with the model it feeds.
- **Recommendation:** Align test_data.embedding_dims with the actual configured c_s/c_z/c_s_inputs (interpolate from the shared/model config rather than hardcoding 384/128/449), or remove the misleading 'must match' comment.

### [MEDIUM] s2c1l2-atom-lists-incomplete-source-of-truth — intent_mismatch
- **Location:** `rna_predict/dataset/atom_lists.py:1-9`
- **Evidence:** The module header claims to be the 'Single source of truth for atom ordering and max atoms per residue', but STANDARD_ATOMS lists only 22 atoms (A/G backbone+base, with the comment '# Extend as needed for all bases'), so MAX_ATOMS_PER_RES = 22. The rest of the system assumes ~44 atoms/residue: config_schema RNAConfig.atoms_per_residue=44, TestDataConfig.atoms_per_residue=44 (config_schema.py:1271-1274,1362-1365), default.yaml:27 atoms_per_residue: 44. loader.py:309 iterates STANDARD_ATOMS to fill atom features, so the canonical-atom enumeration is truncated/inconsistent with the declared per-residue atom count.
- **Recommendation:** Complete STANDARD_ATOMS to the full canonical RNA atom set (or document why 22 is intended) and reconcile MAX_ATOMS_PER_RES with the atoms_per_residue=44 used throughout config and the loader.

### [MEDIUM] s2c1l0-collate-atomnames-inconsistent — bug
- **Location:** `rna_predict/dataset/collate.py:55-57,107-108`
- **Evidence:** rna_collate_fn batches 'atom_names'/'residue_indices' inconsistently across batch sizes. For a single-item batch each is wrapped as [v] (a list containing one per-sample list) at lines 55-57, preserving per-sample nesting. For a multi-item batch (lines 107-108) any value whose first element is a list is FLATTENED across all samples ([item for sublist in vs for item in sublist]), collapsing per-sample boundaries. A consumer indexing batch['atom_names'][sample_i] gets a per-sample list when batch_size==1 but a single merged flat list when batch_size>1.
- **Recommendation:** Treat 'atom_names'/'residue_indices' explicitly as a list-of-lists in both branches (out[k] = vs) rather than flattening sublists, matching the single-item behavior.

### [MEDIUM] s2c1l0-loader-except-undef-var — bug
- **Location:** `rna_predict/dataset/loader.py:469-472`
- **Evidence:** The broad except handler in _load_angles references structure_file and selected_chain_id in the warning string (line 471). Both are local variables assigned only inside the try block (selected_chain_id at lines 402/405/408, structure_file at line 411). If the exception is raised earlier — e.g. during backend resolution at line 361 (getattr on cfg) — these names are unbound, so the except block raises NameError, masking the original error and discarding the intended zeros fallback at line 472.
- **Recommendation:** Initialize structure_file=None and selected_chain_id=None before the try block (or use locals().get) so the fallback path and warning never raise on early failures.

### [MEDIUM] s2c1l0-loader-cuda-numworkers — bug
- **Location:** `rna_predict/dataset/loader.py:84,131-166 + rna_predict/conf/data/default.yaml:10`
- **Evidence:** RNADataset.__getitem__ allocates every tensor directly on self.device (torch.device(cfg.device), which may be 'cuda' or 'mps') — e.g. residue_mask (line 131), coords/embeddings via _load_atom_features, angles (line 166). data/default.yaml sets num_workers: 8 (config_schema DataConfig defaults to 0 and explicitly comments 'set to 0 for debugging device mismatch'). Creating CUDA tensors inside forked DataLoader worker processes triggers 'Cannot re-initialize CUDA in forked subprocess' errors / undefined behavior; this couples a config default with __getitem__ device placement in a way that breaks multi-worker GPU loading.
- **Recommendation:** Build tensors on CPU in __getitem__ and move to device in the training loop / collate, or force num_workers=0 whenever cfg.device is non-CPU.

### [MEDIUM] s2c1l0-angles-snoop-decorator — perf
- **Location:** `rna_predict/dataset/preprocessing/angles.py:11,14`
- **Evidence:** extract_rna_torsions is decorated with @snoop (line 14, `import snoop` line 11), a line-by-line execution tracer. Left enabled in the production extraction entry point it emits per-line trace output for every residue/structure processed (called from RNADataset._load_angles and compute_ground_truth_angles), drastically slowing dataset loading and flooding logs, and makes `snoop` a hard runtime dependency.
- **Recommendation:** Remove the @snoop decorator (and the import) or gate it behind a debug flag.

### [MEDIUM] s2c1l2-angles-import-side-effect — design_defect
- **Location:** `rna_predict/dataset/preprocessing/angles.py:420-449`
- **Evidence:** Module import has a heavy, failure-prone side effect: at import time angles.py unzips a bundled DSSR distribution (_DSSR_ZIP) into a 'dssr' directory, selects a platform-specific nested zip, renames/chmods the binary, and raises RuntimeError on any unsupported platform (lines 425-449). Importing this module (e.g. via loader.py:356 'from ...angles import extract_rna_torsions') performs disk extraction and can hard-fail if the zip is missing or the OS is unrecognized, even when the DSSR backend is never used.
- **Recommendation:** Move the DSSR extraction into a lazily-invoked function called only when backend=='dssr' is actually selected, and surface failures as a handled error rather than an import-time exception.

### [MEDIUM] s2c1l1-002 — security
- **Location:** `rna_predict/dataset/preprocessing/angles.py:424-449`
- **Evidence:** Importing this module triggers, on first use, extraction and `os.chmod(_DSSR_BIN, 0o755)` of a third-party executable (x3dna-dssr) bundled as `dssr-basic-linuxMacWindows-v2.5.3.zip`, which is then executed via `subprocess.run([_DSSR_BIN, ...])` at line 470. There is no integrity/authenticity check on the bundled binary before it is made executable and run. Because this is top-level code (runs at `import rna_predict.dataset.preprocessing.angles`), any code path importing the module — including RNADataset._load_angles (loader.py:356) — silently materializes and prepares an external binary for execution. A compromised or substituted vendored zip yields arbitrary code execution under the user's account.
- **Recommendation:** Gate binary extraction/execution behind an explicit opt-in (config flag), verify a pinned checksum of the binary before chmod/exec, and move the side-effecting setup out of import scope into an explicitly-called function.

### [MEDIUM] s2c1l0-angles-import-side-effect-chmod — bug
- **Location:** `rna_predict/dataset/preprocessing/angles.py:424-449`
- **Evidence:** At module import time the code extracts a DSSR zip and then unconditionally calls os.chmod(_DSSR_BIN, 0o755) at line 449. _DSSR_BIN is only created if the rename loop (lines 445-448) finds a top-level file starting with 'x3dna-dssr' that is X_OK. If the nested archive extracts the binary into a subdirectory (the inventory shows the macOS bundle as dssr-basic-macOS-v2.5.3/x3dna-dssr) or the X_OK check fails (e.g. Windows .exe), the loop never renames a file, _DSSR_BIN does not exist, and os.chmod raises FileNotFoundError — crashing on mere import of the module.
- **Recommendation:** Only chmod when a binary was actually located; search recursively (os.walk) for the binary, and wrap the extraction in a function invoked lazily rather than at import.

### [MEDIUM] s2c1l1-001 — security
- **Location:** `rna_predict/dataset/preprocessing/angles.py:425-449`
- **Evidence:** Module-level (import-time) code unpacks bundled archives with `zipfile.ZipFile(_DSSR_ZIP).extractall(_DSSR_DIR)` (line 430) and `zipfile.ZipFile(nested_path).extractall(_DSSR_DIR)` (line 443) with no member-name sanitization. `extractall` honors absolute paths and `../` traversal entries in the zip ('Zip Slip'), so any maliciously crafted or tampered DSSR archive could write files outside `_DSSR_DIR` (e.g. overwrite arbitrary files in the package tree or user home). The extraction then `os.rename`s the discovered binary and `os.chmod(_DSSR_BIN, 0o755)` (line 449), and the binary is later run via subprocess (line 470). Intent (01-understanding.md:17, audit map) is benign torsion-angle preprocessing, so unsanitized archive extraction is a defect relative to that intent.
- **Recommendation:** Validate each ZipInfo member before extraction (reject absolute paths and any normalized path that escapes the destination dir), or extract members individually with a sanitized join. Verify archive integrity (checksum/signature) before trusting it, and avoid doing extraction/chmod as an import-time side effect.

### [MEDIUM] s2c1l0-ggt-hardcoded-conf-path — bug
- **Location:** `rna_predict/dataset/preprocessing/compute_ground_truth_angles.py:53`
- **Evidence:** main() hardcodes the Hydra config path to '/Users/tomriddle1/RNA_PREDICT/rna_predict/conf' (line 53). On any other machine hydra.initialize(config_path=...) fails; the surrounding try/except (line 57) swallows it, silently dropping all config-derived backend/chain selection so the CLI always falls back to argparse defaults rather than honoring the project config.
- **Recommendation:** Derive the conf path relative to the package (e.g. importlib.resources / Path(__file__) parents) instead of an absolute developer path.

### [MEDIUM] s2c1l0-interface-deadcode-after-raise — bug
- **Location:** `rna_predict/interface.py:46-54`
- **Evidence:** In main()'s predictor-init except block, line 46 executes `raise ValueError(...) from e` unconditionally, so the diagnostic code at lines 48-54 (printing stageB_torsion/stageB_pairformer config and a second `raise`) is unreachable dead code. The intended debug output on initialization failure never runs.
- **Recommendation:** Remove the redundant early raise (line 46) or move the diagnostic prints before the raise so they actually execute.

### [MEDIUM] s2c1l0-datautils-concat-empty — bug
- **Location:** `rna_predict/kaggle/data_utils.py:140-154`
- **Evidence:** process_test_sequences collects per-sequence frames in `frames`, appending only on success and logging.error on failure (lines 146-151). If every sequence raises, `frames` stays empty and `pd.concat(frames, ignore_index=True)` at line 153 raises 'ValueError: No objects to concatenate', crashing the whole submission run instead of producing an empty/partial submission or a clear error.
- **Recommendation:** Guard for empty frames (e.g. if not frames: raise a descriptive error or write an empty submission with the required columns) before pd.concat.

### [MEDIUM] s2c1l2-hardcoded-external-drive-data-path — design_defect
- **Location:** `rna_predict/kaggle/data_utils.py:36 and rna_predict/kaggle/rna_predict.py:77`
- **Evidence:** The non-Kaggle 'local environment' data root is hardcoded to one developer's removable volume: BASE_INPUT_ROOT_EXTERNAL_DRIVE = pathlib.Path('/Volumes/Totallynotaharddrive/RNA_structure_PREDICT/kaggle/') in both data_utils.py:36 and rna_predict.py:77. For any other contributor running the Kaggle harness locally, load_kaggle_data() raises FileNotFoundError. There is no config/env override path for the local data root.
- **Recommendation:** Source the local data root from config (cfg.data.root_dir / a DATA_ROOT env var) with a repo-relative default such as ./data/kaggle, instead of a hardcoded /Volumes mount.

### [MEDIUM] s2c1l1-004 — security
- **Location:** `rna_predict/kaggle/kaggle_env.py:40-63`
- **Evidence:** install_wheels() builds `--find-links` from `/kaggle/input` and every subdirectory under it (lines 46-47), then runs `pip install --no-index ... <pkg>` via subprocess (lines 58-61) for a hardcoded package list. On Kaggle, `/kaggle/input` holds attached datasets which can be arbitrary user-supplied content (e.g. when a notebook is forked or a malicious dataset is attached); allowing pip to resolve those package names from attacker-controllable local wheels enables installation of trojaned wheels (arbitrary code execution at install time). setup_kaggle_environment() similarly pip-installs wheels discovered by path/glob from `/kaggle/input` (lines 217-263). Gated by is_kaggle() but still executes whenever running in that environment.
- **Recommendation:** Restrict find-links to a single trusted, integrity-verified directory; pin exact wheel filenames + hashes (pip --require-hashes); avoid globbing untrusted dataset dirs as a package source.

### [MEDIUM] s2c1l2-legacy-modeling-nonrunnable — design_defect
- **Location:** `rna_predict/kaggle/legacy_feature_engineering_and_modeling.py:17-18,189-204`
- **Evidence:** This is a flattened notebook (`# %%` cells) executed at import/module top level that references undefined globals: train_sequences, validation_sequences, train_labels, validation_labels (lines 17-18), test_sequences (line 218), X_full (line 257) — none are defined or imported, so the module raises NameError immediately if run/imported. Moreover the modeling half is gone: cells 8-9 are stubs with TODOs 'param_dist removed in cleanup pass 1', 'get_best_xgb removed', 'all related code has now been removed' (lines 189-204), so despite the filename '..._and_modeling', no model is ever trained or used.
- **Recommendation:** Either convert this back into a guarded function/script that takes its inputs as parameters (no top-level free variables) and restore or delete the modeling section, or remove the file and keep it as a notebook artifact outside the importable package.

### [MEDIUM] s2c1l2-kaggle-runner-filename-and-paths — doc_drift
- **Location:** `rna_predict/kaggle/rna_predict.py:1-23,314`
- **Evidence:** The file is named rna_predict.py (underscore) but its header comment is '# rna-predict.py' (line 1) and every HOW-TO-RUN example invokes a non-existent 'rna_predict/kaggle/rna-predict.py' (hyphen) (lines 12,18,20) plus a trailing example at line 314. The examples also hardcode developer-specific absolute paths '/Users/tomriddle1/.local/bin/uv' and '/Users/tomriddle1/RNA_PREDICT/rna_predict/conf'. A user copy-pasting the documented command runs nothing.
- **Recommendation:** Fix the filename references to rna_predict.py and replace the /Users/tomriddle1 absolute paths with portable relative invocations (e.g. 'uv run rna_predict/kaggle/rna_predict.py --config-path ../conf').

### [MEDIUM] s2c1l2-kaggle-train-cfg-hydra-access — bug
- **Location:** `rna_predict/kaggle/rna_predict.py:270`
- **Evidence:** run_training_pipeline reads cfg.hydra.run.dir inside the is_kaggle() branch, but the 'hydra' node is not part of the composed application cfg by default (the runtime cfg keys in combined_pipeline_output.txt:1575 contain no 'hydra'). Accessing cfg.hydra.run.dir will raise ConfigAttributeError/AttributeError, breaking Kaggle 'train' mode at exactly the point it tries to verify the output directory.
- **Recommendation:** Use hydra.core.hydra_config.HydraConfig.get().run.dir (as conf/utils.get_run_dir already does) instead of cfg.hydra, or guard the access with a presence check.

### [MEDIUM] s2c1l0-kaggle-cfg-hydra-rundir — bug
- **Location:** `rna_predict/kaggle/rna_predict.py:270-271`
- **Evidence:** Inside @hydra.main main() -> run_training_pipeline, line 270 reads cfg.hydra.run.dir. Under hydra.main the job config does not contain the 'hydra' node (it is stripped from the composed cfg and exposed only via HydraConfig.get()), so cfg.hydra.run.dir raises ConfigAttributeError. This path is reached on Kaggle training (is_kaggle() and mode=train), aborting training during the run-dir check.
- **Recommendation:** Use hydra.core.hydra_config.HydraConfig.get().run.dir instead of cfg.hydra.run.dir.

### [MEDIUM] s2c1l0-submission-validator-sysexit — design_defect
- **Location:** `rna_predict/kaggle/submission_validator.py:32-35`
- **Evidence:** run_sanity_checks is imported and called as a library function by the Kaggle harness (rna_predict.py:190), but on a missing input file it calls sys.exit(...) at line 35 (the code comment itself notes 'Or raise an error'). sys.exit terminates the entire process rather than letting the caller handle the condition, so a missing test/submission CSV kills the whole pipeline run instead of being logged/handled.
- **Recommendation:** Raise FileNotFoundError (or return a status) instead of sys.exit so callers can decide how to react.

### [MEDIUM] s2c1l0-merger-ignores-inputs — design_defect
- **Location:** `rna_predict/pipeline/merger/simple_latent_merger.py:14-17,43-76`
- **Evidence:** SimpleLatentMerger's class docstring states it merges 'adjacency, angles, single embeddings, pair embeddings, plus partial coords' but forward() only uses inputs.angles, inputs.s_emb and inputs.z_emb (lines 55-58,74); inputs.adjacency and inputs.partial_coords (LatentInputs fields) are never consumed. Additionally, the stored self.expected_dim_angles/_s/_z (lines 30-32) are never used.
- **Recommendation:** Either incorporate adjacency/partial_coords into the merge or remove them from LatentInputs and the docstring to reflect actual behavior; drop the unused expected_dim_* attributes.

### [MEDIUM] s2c1l0-merger-dynamic-mlp-reinit — bug
- **Location:** `rna_predict/pipeline/merger/simple_latent_merger.py:63-73`
- **Evidence:** forward() rebuilds self.mlp with freshly-initialized random Linear layers whenever the runtime concatenated input dim differs from self.mlp[0].in_features (lines 63-71). This silently discards any learned/loaded weights on a dimension change, the new submodule's parameters are not registered with any existing optimizer (created mid-forward), and repeated calls with fluctuating dims would re-randomize every step, breaking training/inference reproducibility.
- **Recommendation:** Fix the input dimension at __init__ (from the declared dim_angles/dim_s/dim_z) and assert/validate incoming shapes, rather than reconstructing the MLP inside forward().

### [MEDIUM] s2c2l0-003 — bug
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:104-110`
- **Evidence:** process_seqs calls F.one_hot(nseq).float() (line 109) without num_classes. F.one_hot infers the class count from the max index present, so a sequence that lacks the highest-valued base (G=3) yields a one-hot tensor with fewer than 4 channels, breaking the fixed 4-channel assumption used by constraint_matrix (x[:,:,0..3]) and downstream conv input. Shape becomes data-dependent. (Direct caller of process_seqs within this file is unverified — RFoldModel.forward uses Seq2Map instead.)
- **Recommendation:** Pass an explicit num_classes=4: F.one_hot(nseq, num_classes=4).float() so the channel dimension is deterministic regardless of which bases appear.

### [MEDIUM] s2c2l2-rfold-seq2dot-hardcoded-test — design_defect
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:164-167`
- **Evidence:** seq2dot() begins with `if len(seq) == 4 and seq[0]==2 and seq[1]==0 and seq[2]==3 and seq[3]==0: return "(.))"` — a hardcoded answer for a specific test input baked into production secondary-structure dot-bracket generation. The general logic that follows is bypassed for this input.
- **Recommendation:** Delete the hardcoded special case; if the test expects this output, fix the general algorithm to produce it or assert it in the test fixture rather than in source.

### [MEDIUM] s2c2l2-rfold-encoder-decoder-test-identity — design_defect
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:377-404`
- **Evidence:** Encoder.__init__ uses `if len(C_lst) <= 3:  # This is likely a test case` to install nn.Identity, and Encoder.forward returns `x, [x]` when `len(self.enc) <= 1` (lines 397-399). Decoder.__init__ has the same `if len(C_lst) <= 3` test branch (:413-415) returning the input unchanged. Production model topology is silently replaced with a no-op based on a heuristic guess that small channel lists 'are likely a test'.
- **Recommendation:** Drop the test-detection branches from the model classes; construct small/identity variants explicitly in tests instead of inferring intent from C_lst length.

### [MEDIUM] s2c2l0-002 — bug
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:75-92`
- **Evidence:** constraint_matrix builds au_ua + cg_gc + ug_gu and returns it, but the no-sharp-loop mask produced by base_matrix(...) (defined at line 66) is never applied. The inline comments at lines 89-91 ('Apply the base_matrix constraint' / 'Apply base matrix constraints while preserving the correct pairs') claim a constraint is applied that is not. base_matrix is defined but has no caller within this file, so its intended masking effect is dropped. (Caller set outside this file is unverified.)
- **Recommendation:** Either multiply the combined pair matrix by base_matrix (constraint = (au_ua+cg_gc+ug_gu) * base_matrix(length, device)) as the comments intend, or delete the misleading comments and the unused base_matrix helper to remove the behavior/doc divergence.

### [MEDIUM] s2c2l2-rfoldpred-docstring-official-claim — doc_drift
- **Location:** `rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:2-86`
- **Evidence:** Module/class docstrings claim it 'uses the official RFold_Model code from "RFold/model.py" so that pretrained checkpoints load successfully without key mismatch' and reference importing 'from RFold.model import RFold_Model'. In reality the predictor imports the local RFoldModel from RFold_code (line 29-31, 172) and loads checkpoints with `load_state_dict(ckp, strict=False)` (line 282-284), which tolerates key mismatches rather than guaranteeing exact matching. No 'RFold/' package is imported.
- **Recommendation:** Update the docstrings to describe the actual local RFoldModel and strict=False loading behaviour, and remove references to a non-existent official RFold module.

### [MEDIUM] s2c2l2-rfoldpred-silent-random-weights — design_defect
- **Location:** `rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:244-289`
- **Evidence:** _load_checkpoint logs a warning and returns (continuing with randomly-initialized weights) when checkpoint_path is None, the file is missing, or torch.load/load_state_dict raises. Given Stage-1 intent that Stage A inference is 'Functional' (01-understanding.md:6), silently proceeding with random weights yields meaningless adjacency predictions while appearing to succeed.
- **Recommendation:** For inference mode, fail loudly (raise) when a required checkpoint cannot be loaded, or surface a clear unrecoverable error; reserve random-weight fallback for explicitly opted-in dummy/training-from-scratch paths.

### [MEDIUM] s2c2l0-005 — bug
- **Location:** `rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:281-289`
- **Evidence:** _load_checkpoint calls self.model.load_state_dict(ckp, strict=False) and then logs '[Load] Checkpoint loaded successfully.' unconditionally. With strict=False, a checkpoint whose keys do not match (e.g. official RFold key naming vs this refactored RFoldModel) loads ZERO parameters yet still reports success, leaving the model on random weights while predict_adjacency proceeds. The returned missing/unexpected keys from load_state_dict are discarded, so this silent failure is undetectable from logs.
- **Recommendation:** Capture the IncompatibleKeys result (missing/unexpected) returned by load_state_dict and warn/raise when the overlap is empty or below a threshold; report counts of matched vs missing keys instead of an unconditional success message.

### [MEDIUM] s2c2l2-embedders-lazy-linear — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/embedders.py:280-294`
- **Evidence:** InputFeatureEmbedder.extras_linear is left as None in __init__ (line 127) and constructed lazily inside forward(): `self.extras_linear = LinearNoBias(extras_dim, self.c_token).to(extras_cat.device)`. A submodule created during the first forward is absent from the module's initial state_dict and from any optimizer parameter group built before that forward, so its weights are not checkpointed/restored consistently and are not optimized if the optimizer was created at construction time.
- **Recommendation:** Determine extras_dim at construction (from restype_dim+profile_dim+1) and create extras_linear in __init__, or use nn.LazyLinear, so the parameter is registered before training/serialization.

### [MEDIUM] s2c2l0-019 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/embedders.py:280-294`
- **Evidence:** InputFeatureEmbedder.forward creates self.extras_linear lazily on the first forward pass (LinearNoBias(extras_dim, c_token) at lines 285-287) with random weights, after __init__ has run and after any checkpoint load. Because it is not constructed in __init__, it is absent from the module's state_dict at load time, so a loaded checkpoint cannot populate it; at inference these projection weights remain untrained/random, and the added projection extras_proj = self.extras_linear(extras_cat) (line 294) injects noise into s_inputs. For a 'Functional' inference path this silently degrades the embedding.
- **Recommendation:** Construct extras_linear in __init__ from the known restype_dim+profile_dim+1 input size so it is part of the checkpointed state, or assert/raise if it must be lazily created at inference time.

### [MEDIUM] s2c2l0-007 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives.py:18-33`
- **Evidence:** This module imports `_attention` from .primitives.attention_base (line 18-22), but attention_base.py only defines/exports `attention` (attention_base.py:10-26, __all__ has no '_attention'); and it imports broadcast_token_to_local_atom_pair, gather_pair_embedding_in_dense_trunk, rearrange_qk_to_dense_trunk, rearrange_to_dense_trunk from .primitives.data_transforms (lines 28-33), but data_transforms.py is an empty facade exporting nothing (data_transforms.py:1-16). Either import would raise ImportError. The module survives only because it is shadowed: a `primitives/` package (primitives/__init__.py) exists in the same directory and takes import precedence over `primitives.py`, so this file is never actually imported — it is dead code carrying broken imports.
- **Recommendation:** Delete primitives.py (it is shadowed by the primitives/ package), or fix its imports to source the symbols from the modules that actually define them (attention_core.attention, atom_pair_transforms.*, attention.dense_trunk.*).

### [MEDIUM] s2c2l0-010 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:102,121,126,139,143,152-156,172,176-177,194-195,199`
- **Evidence:** AdaptiveLayerNorm.forward and _apply_conditioning contain ~10 unconditional print(...) statements (not gated by any debug flag) that fire on every forward pass, formatting tensor shapes/devices/grad state each call. In a transformer with many AdaLN invocations per step this floods stdout and adds per-call Python/format overhead on the hot path. Additionally the method docstrings at lines 103-112 and 157-167 are placed AFTER executable print statements, so they are ordinary string expressions, not docstrings.
- **Recommendation:** Remove the print() calls or guard them behind `if self.debug_logging:` using logger.debug; move the docstrings to the first statement of each function.

### [MEDIUM] s2c2l2-adaln-unconditional-print — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:102-199`
- **Evidence:** AdaptiveLayerNorm._apply_conditioning and forward emit numerous unconditional `print(f"[DEBUG][AdaLN]...")` statements (lines 102, 121, 126, 139, 143, 152-156, 172, 176-177, 194-195, 199) on every invocation, not gated by any debug flag. Other modules in this tree use logger.debug guarded by debug_logging; here every forward pass writes many lines to stdout, degrading performance and flooding output during inference/training.
- **Recommendation:** Replace these prints with logger.debug guarded by a debug flag, or remove them.

### [MEDIUM] s2c2l0-009 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:182-192`
- **Evidence:** Inside AdaptiveLayerNorm.forward, on a feature-dim mismatch (s.shape[-1] != layernorm_s.normalized_shape) the module re-instantiates self.layernorm_s, self.linear_s, and self.linear_nobias_s with freshly random weights (lines 186-189) at runtime. During inference this silently discards the trained/checkpoint-loaded weights for those layers and replaces them with random ones, producing garbage conditioning instead of failing loudly. It also mutates module structure during forward, which is unsafe under checkpointing/DDP.
- **Recommendation:** Treat a conditioning-dimension mismatch as an error (raise with a clear message), or adapt the input tensor s to c_s instead of rebuilding trained submodules; never reinitialize learned layers inside forward().

### [MEDIUM] s2c2l2-adalnutils-print-and-silent-resize — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm_utils.py:117,342-345`
- **Evidence:** interpolate_sequence_dim() emits an unconditional `print(f"[DEBUG][AdaLN][interpolate_sequence_dim]...")` (line 117). More substantively, the helpers silently coerce mismatched token dimensions: _handle_more_tokens_in_scale truncates scale/shift to a.shape[-2] (lines 342-343) and _handle_fewer_tokens_in_scale / interpolate_sequence_dim use nearest-neighbour interpolation to resize the token axis. These mask shape bugs in conditioning by altering tensor semantics rather than failing.
- **Recommendation:** Remove the print; treat token-dimension mismatches between scale/shift and the conditioned tensor as errors rather than silently interpolating/truncating, since the result is not a valid AdaLN.

### [MEDIUM] s2c2l2-attn-duplicate-divergent-stack — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_bias.py:50-83`
- **Evidence:** There are two parallel, divergent implementations of the local-attention utilities. primitives/__init__.py wires in attention_utils.py (imports _local_attention, create_local_attn_bias, optimized_concat_split from .attention_utils, __init__.py:38-42). A second, non-wired stack (attention_bias.py, attention_local.py, attention_tensor.py, attention_types.py) redefines the same names with DIFFERENT behaviour: e.g. create_local_attn_bias here returns shape (1,1,n_queries,n_keys) with simple edge masking (attention_bias.py:47,50-83), whereas attention_utils.create_local_attn_bias returns (1,n_chunks,n_queries,n_keys) with a sliding window (attention_utils.py:108-147). The dataclasses (LocalAttentionInputs, AttentionChunkConfig, etc.) are likewise duplicated in attention_types.py vs attention_utils.py. The second stack appears dead but diverges silently, inviting wrong-import bugs.
- **Recommendation:** Consolidate to a single attention-utility implementation; delete the unused divergent modules (attention_bias/attention_local/attention_tensor/attention_types) or make them thin re-exports of the canonical one.

### [MEDIUM] s2c2l0-012 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_module.py:92-98`
- **Evidence:** Attention._initialize_parameters zero-initializes the query, key, AND value projection weights (nn.init.zeros_ on self.to_q/to_k/to_v at lines 94-96) in addition to the output projection and gating. With to_q/to_k/to_v all zero, at initialization q=k=v=0, softmax is uniform, the attended value is 0, and to_out(0)=0; gradients to all these layers are also 0 (the zero output projection blocks gradient flow), so the module is a dead unit that cannot learn when trained from scratch and outputs zeros until a checkpoint overwrites the weights. Standard AF3/openfold practice zero-inits only the final output (and gating) projection, not q/k/v.
- **Recommendation:** Initialize to_q/to_k/to_v with the default (LeCun/Glorot) initialization and zero-init only to_out (and gating_linear); keep qkv non-zero so the module is trainable.

### [MEDIUM] s2c2l0-014 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_processing.py:120,175,179,183,187,189,193,200,202`
- **Evidence:** _process_with_batch_matmul (line 120) and _reshape_attention_bias (lines 175-202) emit numerous unconditional print("DEBUG...") statements on every attention call/bias reshape, with no debug-flag gating. These run on the attention hot path for every block and head, spamming stdout and adding per-call overhead in production inference/training.
- **Recommendation:** Delete the print statements or convert them to logger.debug guarded by an explicit debug flag.

### [MEDIUM] s2c2l2-attnprocessing-print-spam-hotpath — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_processing.py:120,175-202`
- **Evidence:** _process_with_batch_matmul (used by Attention.forward via process_same_query_keyvalue) emits an unconditional `print(f"[DEBUG][BatchMatmul] ...")` per call (line 120), and _reshape_attention_bias emits ~8 unconditional `print("DEBUG: ...")` statements per call (lines 175,179,183,187,189,193,200,202). These are on the live attention path and run on every forward when bias is present, not gated by any debug flag.
- **Recommendation:** Replace all bare print() debug statements with logger.debug guarded by a debug flag, or remove them.

### [MEDIUM] s2c2l0-013 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_processing.py:49-74; attention_utils_internal.py:79-84; attention_core.py:332-334`
- **Evidence:** Double query scaling on the self-attention path. In Attention.forward, head_config.apply_scale = (q_x is kv_x and q_x.ndim==3) (attention_module.py:122), and prep_qkv multiplies q by head_dim**-0.5 when apply_scale is True (attention_utils_internal.py:80-84). The same inputs then flow into process_same_query_keyvalue, whose efficient path F.scaled_dot_product_attention re-divides by sqrt(head_dim) (attention_processing.py:49-55) and whose manual path calls attention() -> compute_attention_weights which divides q@k by math.sqrt(d_k) again (attention_core.py:332-334 / attention_weights.py:334). The result is scaling by 1/d instead of 1/sqrt(d), over-sharpening the softmax for 3D self-attention.
- **Recommendation:** Apply temperature scaling in exactly one place: either keep prep_qkv's pre-scaling and pass scale=1.0 to SDPA / skip the /sqrt(d_k) in the manual path, or drop prep_qkv's apply_scale and rely solely on the in-attention scaling.

### [MEDIUM] s2c2l0-016 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils_internal.py:171-195,255-293`
- **Evidence:** apply_gating (lines 171-195) and _infer_and_reshape/wrap_up (lines 255-293) contain reshape logic hardcoded to specific test tensor sizes: 'if o.numel()==8192 and target_hidden==128: return o.reshape(64,128)  # Special case for the test_n_sample_handling test' and `o.shape[-2]*o.shape[-1]==1024`, `g.numel()==1024`, `o.shape[1]==128`. These size-keyed branches alter how multi-head output is folded/gated only for those exact magic numbers, making behavior data-size-dependent and embedding test fixtures into the production reshape path.
- **Recommendation:** Derive the reshape purely from num_heads/head_dim/c_hidden and the tensor's own dims; remove the numel()==8192 / 1024 / shape==128 special cases and raise on truly incompatible shapes.

### [MEDIUM] s2c2l2-attnweights-silent-qk-resize — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_weights.py:61-71`
- **Evidence:** handle_dimension_mismatch silently zero-pads or truncates the query tensor's contraction dimension to match the key (lines 63-71) before matmul. This produces a numerically wrong attention score (padded zeros or dropped features) instead of surfacing a genuine dimension bug.
- **Recommendation:** Raise on q/k contraction-dimension mismatch rather than padding/truncating; such a mismatch indicates an upstream configuration error.

### [MEDIUM] s2c2l0-015 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_weights.py:86-103,180-191; attention_bias.py:238-260; attention_utils.py:410-432`
- **Evidence:** Multiple attention helpers silently mutate bias/weight shapes using test-fixture-specific magic constants rather than failing on genuine mismatches. _handle_bias_dimension_mismatch keys on attn_bias.dim()==5 & attn_weight.dim()==4 'for the test_reproduce_shape_mismatch.py test' (attention_weights.py:89-101); _handle_5d_bias_mismatch repeats bias to fill a larger dim (lines 180-191), changing attention semantics; and _fix_dimension_mismatch branches on q_dim_2==5/bias_dim_2==4 i.e. comparing a dimension SIZE to literal 5/4 (attention_bias.py:251-257 and attention_utils.py:423-429). These auto-fixes can paper over real shape bugs and produce silently wrong attention masks/outputs instead of raising.
- **Recommendation:** Replace the magic-number reshape branches with explicit validation that raises on incompatible bias shapes; reserve broadcasting to genuine size-1 dims and remove test-specific constants from library code.

### [MEDIUM] s2c2l2-attnweights-test-special-case — intent_mismatch
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_weights.py:89-101`
- **Evidence:** _handle_bias_dimension_mismatch contains a branch labelled 'Special case for the test_reproduce_shape_mismatch.py test' that expands the bias dim for the exact shapes that test produces ([1,1,4,25,25] vs [1,4,25,25]). Production attention-weight computation is shaped around a named test file.
- **Recommendation:** Generalize the bias-broadcast handling (rely on torch broadcasting) and remove the test-specific branch and comment.

### [MEDIUM] s2c2l2-shapeadapter-hardcoded-transformer-case — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/shape_adapter.py:95-114`
- **Evidence:** adapt_tensors_for_addition handles non-broadcastable mismatches only for the 'specific case in the transformer' with hardcoded assumptions about p_lm/z_transformed shapes ([1,1,1,10,10,10,16] vs [1,1,1,32,128,16]) and mean-pool+expand at dims 3/4/5 (lines 95-112, labelled 'temporary solution for the specific case'). For any other mismatch (or rank < 6) it returns the still-incompatible tensors unchanged, so a real addition downstream would error or silently mis-broadcast.
- **Recommendation:** Replace the hardcoded-shape hack with a principled broadcasting/validation routine, or raise on unsupported mismatches instead of returning incompatible tensors.

### [MEDIUM] s2c2l2-tensorshapepatch-crossstage-and-dup — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/tensor_shape_patch.py:13-134`
- **Evidence:** This Stage A module's apply_patches()/patch_* functions monkey-patch pipeline functions by importing from rna_predict.pipeline.stageD.diffusion.run_stageD_unified (lines 21, 117, 128-131) — a Stage A 'shape patch' reaching into Stage D, a band-aid coupling across stages. It also re-defines adapt_indices_for_gather and adapt_tensors_for_addition (lines 30-108) that duplicate shape_adapter.py but with DIVERGENT bodies (here adapt_tensors_for_addition only adjusts dim 4 / unsqueezes dim 5, vs shape_adapter.py which mean-pools dims 3/4/5). apply_patches() also prints via bare print() (line 26).
- **Recommendation:** Remove the monkey-patching approach in favour of fixing the underlying tensor shapes; eliminate the duplicate divergent adapt_* helpers (keep one source of truth) and avoid Stage A depending on Stage D internals.

### [MEDIUM] s2c2l2-transformer-module-shadowed — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer.py:1-48`
- **Evidence:** current/ contains both transformer.py (module) and transformer/ (package with __init__.py). Python resolves the package, so transformer.py is unreachable dead code. It also diverges from the package: transformer.py imports AtomAttentionEncoder/Decoder from .transformer.atom_attention and does not export AtomAttentionConfig, whereas the package __init__.py imports them from atom_attention_encoder.py/atom_attention_decoder.py and does export AtomAttentionConfig (the symbol embedders.py:24-28 actually relies on).
- **Recommendation:** Delete the shadowed transformer.py (the transformer/ package is the live module) to remove the dead, divergent duplicate.

### [MEDIUM] s2c2l2-atomattention-module-shadowed — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:1-3`
- **Evidence:** transformer/ contains both atom_attention.py (module) and atom_attention/ (package with __init__.py). Python resolves the package, so atom_attention.py (1001 lines) is unreachable dead code; the live encoder/decoder are atom_attention/encoder.py and atom_attention/decoder.py per atom_attention/__init__.py:5-15. Stage-1 also marks the package's encoder/decoder as canonical (01-understanding.md:309-310).
- **Recommendation:** Delete the shadowed atom_attention.py module (or merge any still-wanted logic into the package) to eliminate the dead duplicate.

### [MEDIUM] s2c2l2-atomattention-forward-legacy-mismatch — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:687-778`
- **Evidence:** In atom_attention.py, AtomAttentionEncoder.forward has signature forward(self, input_feature_dict, chunk_size=None) (line 687), but forward_legacy builds an EncoderForwardParams dataclass and calls self.forward(params) (lines 770-778), passing the params object where a feature dict is expected. forward then does `"ref_space_uid" in input_feature_dict` / dict access on a dataclass and would fail. (This file is also shadowed, so the bug is latent.) Additionally _process_input_features is defined twice with identical bodies (lines 300-316 and 483-499); the second silently overrides the first.
- **Recommendation:** If this module is kept, fix forward_legacy to unpack params into forward's expected arguments and remove the duplicate _process_input_features definition; otherwise delete the shadowed file.

### [MEDIUM] s2c2l0-020 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:743-778`
- **Evidence:** AtomAttentionEncoder.forward_legacy builds an EncoderForwardParams dataclass and calls self.forward(params) (lines 770-778), but the actual forward signature is forward(self, input_feature_dict, chunk_size=None) (line 687-691). The params object is therefore bound to input_feature_dict and immediately passed to _process_input_features which does `'ref_space_uid' in input_feature_dict` and dict indexing (lines 483-499), so the call raises/misbehaves because an EncoderForwardParams is not the expected feature dict. Note this whole module (transformer/atom_attention.py) is shadowed at import time by the same-named package transformer/atom_attention/ (transformer/atom_attention/__init__.py), and transformer/__init__.py imports the encoder from atom_attention_encoder.py instead, so this file is likely dead — but the legacy API it advertises is broken.
- **Recommendation:** If the module is dead, remove it to avoid the package/module name collision; otherwise fix forward_legacy to unpack params (self.forward(params.input_feature_dict, params.chunk_size)) or make forward accept an EncoderForwardParams.

### [MEDIUM] s2c3l2-001 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:52,112`
- **Evidence:** FeatureProcessor.__init__ (line 52) and extract_atom_features (line 112) execute unconditional `print(f"[DEBUG][FeatureProcessor] ...")` on every construction and every forward. The constructor docstring (line 35) explicitly states debug_logging is 'ignored in this implementation', yet these prints fire regardless of any flag. The encoder built on this (Stage A input-embedding, README 'Functional' inference path) will spam stdout on each run.
- **Recommendation:** Remove the stray DEBUG prints or gate them behind a logger.debug() call; honor the documented debug_logging contract.

### [MEDIUM] s2c3l2-006 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/config.py:16`
- **Evidence:** Two divergent dataclasses both named AtomAttentionConfig exist: atom_attention/config.py:16 (no debug_logging field) and encoder_components/config.py:30 (adds debug_logging). Consumers defensively use `getattr(config, 'debug_logging', None)` (e.g. atom_attention/encoder.py:47,52) to paper over the divergence. Same-named config with different fields invites silent attribute-missing behaviour depending on which is imported.
- **Recommendation:** Consolidate to a single AtomAttentionConfig (or clearly namespace the two) and remove the defensive getattr fallbacks.

### [MEDIUM] s2c3l2-021 — doc_drift
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:127 (docstring); return paths forward_logic.py:166 and :525`
- **Evidence:** AtomAttentionEncoder.forward docstring (atom_attention_encoder.py:127) states the 4-tuple return is (token embeddings, pair embeddings, style embeddings, coordinate embeddings). Actual returns are inconsistent: _process_simple_embedding returns `(a, q_l, c_l, torch.zeros_like(a))` (line 166) while process_inputs_with_coords returns `(a, q_l, c_l, p_for_transformer)` (line 525). Position 2 is q_l (atom features), not 'pair embeddings', and position 4 is zeros vs p across the two paths — the documented tuple semantics do not match either implementation.
- **Recommendation:** Align the docstring with the actual tuple, and make the simple and coords paths return the same element semantics.

### [MEDIUM] s2c3l0-026 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:144`
- **Evidence:** AtomAttentionEncoder.forward (components version) expands the attention mask along the pair-embedding channel count: when mask.shape[-1]==1 it does mask = mask.expand(-1,-1,self.c_atompair) (lines 144-145), producing [B,N,c_atompair]. An atom mask should index atoms (length N), not be tiled to c_atompair feature channels; the resulting tensor is then passed as `mask` to apply_transformer (line 148), conflating a per-atom mask with a per-channel tensor.
- **Recommendation:** Keep the mask as a per-atom boolean/float of shape [..., N] (or [..., N, 1]); do not expand it to c_atompair. Align with how AttentionPairBias consumes masks.

### [MEDIUM] s2c3l2-008 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:131,135,144`
- **Evidence:** AtomAttentionDecoder.forward mutates its input dataclass in place: it reassigns params.extra_feats (lines 131,135) and params.atom_mask (line 144). Callers reusing the same DecoderForwardParams object across calls would see corrupted state; forward should be free of input side-effects.
- **Recommendation:** Operate on local copies (e.g. local extra_feats/atom_mask variables) rather than writing back into params.

### [MEDIUM] s2c3l2-009 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:141-188`
- **Evidence:** Two ~45-line near-identical nested try/except 'mask adaptation' blocks (lines 141-188 and 234-278) attempt several reshape heuristics and, on failure, log an error and silently 'Skipping mask application' (lines 186-188, 276-278). Masking that silently no-ops produces wrong (unmasked) outputs instead of a clear error, and the duplicated best-effort python loops (lines 176-183, 266-273) are a maintenance hazard.
- **Recommendation:** Define mask shapes up front and broadcast deterministically; fail loudly on incompatible shapes instead of skipping masking; de-duplicate the pre/post blocks.

### [MEDIUM] s2c3l2-007 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:43`
- **Evidence:** Two unrelated classes named AtomAttentionDecoder coexist with contradictory contracts: atom_attention/decoder.py:20 returns atom-level embeddings, while atom_attention_decoder.py:43 ('Implements Algorithm 6') returns 3D coordinates [...,N_atom,3]. Similarly two AtomAttentionEncoder classes (atom_attention/encoder.py:23 vs atom_attention_encoder.py:40, the latter delegating to encoder_components/). The duplicate-name/duplicate-purpose trees make it ambiguous which implementation the pipeline actually uses.
- **Recommendation:** Designate one canonical encoder/decoder pair, delete or clearly mark the other as deprecated, and update imports.

### [MEDIUM] s2c3l0-018 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:167`
- **Evidence:** transformer_patch.patch_transformer monkeypatches AtomAttentionEncoder.forward (transformer_patch.py:108) with patched_forward(self, input_feature_dict, r_l, s, z, ...) that calls original_forward(self, input_feature_dict, r_l, s, z, inplace_safe, chunk_size) positionally (lines 103-104). But the real forward signature is forward(self, *args, **kwargs) (atom_attention_encoder.py:106) which extracts r_l/s/z via kwargs.get('r_l') (lines 130-132); positional r_l/s/z are swallowed into *args and never read, so they are lost. The patch also calls self.layernorm_a/self.layernorm_s/self.linear_no_bias_z (transformer_patch.py:36,40,64) which are not attributes set on the refactored encoder.
- **Recommendation:** If the patch is still needed, forward r_l/s/z as keyword args matching the real signature and verify the referenced layernorm/linear attributes exist; otherwise delete this dead/broken patch module.

### [MEDIUM] s2c3l2-010 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:233-240`
- **Evidence:** forward_debug issues unconditional `print(f"[DEBUG][forward_debug] ...")` for atom_to_token_idx, c_l, s, z, r_l, t_hat_noise_level, restype (lines 233-240) regardless of self.debug_logging. This method is a public method on a production nn.Module and will dump to stdout whenever invoked.
- **Recommendation:** Gate all prints behind logger.debug()/self.debug_logging, or remove forward_debug from shipped code.

### [MEDIUM] s2c3l0-009 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_transformer.py:117`
- **Evidence:** _reshape_4d_tensor guards the reshape with `if p.shape[1] * p.shape[2] == p.shape[1] * p.shape[2]:` (line 117) — a tautology comparing a value to itself, always True. The intended validation (presumably that the merged dimension matches an expected size) is absent, so the else branch raising 'cannot reshape' (lines 120-124) is dead and any 4D pair tensor is silently flattened across dims 1 and 2 regardless of correctness.
- **Recommendation:** Replace the tautological condition with the real intended check (e.g. compare against the expected n_queries*n_keys or against c_atompair layout), or remove the dead else branch and document the unconditional reshape.

### [MEDIUM] s2c3l0-008 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:239`
- **Evidence:** _create_default_z builds torch.zeros((*a.shape[:-1], *a.shape[:-1], self.c_z)). For a of shape [B,N,c_a], a.shape[:-1]=(B,N) so the result is [B,N,B,N,c_z] (the batch dim is duplicated) instead of the intended pair tensor [B,N,N,c_z]. Any path that hits this fallback (local_multihead_attention line 280) produces a mis-shaped bias.
- **Recommendation:** Construct as torch.zeros((*a.shape[:-2], a.shape[-2], a.shape[-2], self.c_z), ...) so only the atom axis is squared.

### [MEDIUM] s2c3l2-015 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:469-610`
- **Evidence:** _apply_gating contains many layers of best-effort shape adaptation that, on any failure, `warnings.warn(...)` and `return a` ('Using identity gating' at lines 514,530,536,557,560,592,597,610). Silently dropping the adaLN-Zero gating changes model semantics (the output projection is the gating per AF3) without surfacing an error, masking real shape bugs. standard_multihead_attention (lines 432-449) similarly squeezes/warns on dim mismatches.
- **Recommendation:** Compute s's expected shape deterministically and raise on mismatch; do not silently substitute identity gating in a model whose correctness depends on it.

### [MEDIUM] s2c3l2-014 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:540`
- **Evidence:** _apply_gating prints `[INSTRUMENT][Attention] s.shape=...` unconditionally whenever `torch.is_grad_enabled()` (line 539-540) — i.e. during all training/grad-enabled forwards. Instrumentation left in the hot path.
- **Recommendation:** Delete the instrument print or convert to a guarded logger.debug call.

### [MEDIUM] s2c3l0-007 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:622`
- **Evidence:** AttentionPairBias.forward begins with print(f"[DEBUG][APB] ENTRY...") at line 622, placed BEFORE the function docstring (lines 623-637) — so the triple-quoted string is a no-op expression and the real docstring is lost. Additional unconditional prints fire every forward at lines 643,656,659,664,665, and _apply_gating prints at line 540 whenever torch.is_grad_enabled(). On the inference/training hot path this is significant overhead and log spam.
- **Recommendation:** Move the docstring to the top of forward and convert all prints to logger.debug guarded by a level check; remove the [INSTRUMENT] print in _apply_gating.

### [MEDIUM] s2c3l2-024 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/encoder_feature_processing.py:1`
- **Evidence:** encoder_feature_processing.py (header '# Copied from feature_processing.py') duplicates _process_feature, adapt_tensor_dimensions and extract_atom_features from feature_processing.py but they diverge: encoder_feature_processing.extract_atom_features (lines 173-245) lacks the dimension-alignment fix, the in_features assert, and the ensure_space_uid present in feature_processing.extract_atom_features (lines 151-340). Both are live: forward_logic.py:25 imports the encoder_feature_processing version (used by _process_simple_embedding via extract_atom_features_with_config, line 312), while atom_attention_encoder.py:23 imports feature_processing's as canonical_extract_atom_features (used by the coords path, line 262). Two code paths thus extract atom features with materially different logic.
- **Recommendation:** Collapse to a single extract_atom_features implementation imported by both paths; delete the divergent copy.

### [MEDIUM] s2c3l0-012 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/feature_processing.py:30`
- **Evidence:** _process_feature silently fabricates zero/one default tensors for any missing feature and writes them back into the caller's input_feature_dict (lines 30-89; also extract_atom_features fabricates defaults lines 190-245). Missing required atom features (ref_pos, ref_element, etc.) are thus masked rather than surfaced, and the input dict is mutated as a side effect, so a genuine data-pipeline bug upstream produces plausible-but-meaningless embeddings instead of an error.
- **Recommendation:** Fail loudly (raise) on missing required features in non-test paths, or at minimum log a warning and avoid mutating the caller's dict; reserve default fabrication for an explicit opt-in flag.

### [MEDIUM] s2c3l0-011 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/feature_processing.py:33`
- **Evidence:** Default tensors for missing features are created with default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') (feature_processing.py:33, also encoder_feature_processing.py:33, forward_logic.py:324). If a CUDA device exists but the model/other inputs are on CPU (common in tests / CPU inference), the fabricated tensor lands on cuda while real features are on cpu, so the subsequent torch.cat (feature_processing.py:323) or linear_no_bias_f (line 340) raises a device-mismatch RuntimeError.
- **Recommendation:** Derive the device from an existing input tensor (e.g. ref_pos/atom_to_token_idx) or from a module parameter, never from cuda.is_available().

### [MEDIUM] s2c3l2-022 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:315-378`
- **Evidence:** _process_inputs_with_coords_impl (lines 315-378) is a near-duplicate of process_inputs_with_coords (lines 381-525) but is not referenced by the encoder forward (atom_attention_encoder.py:167 calls process_inputs_with_coords). It appears to be an unused divergent copy carrying its own num_tokens/p_lm logic. (Not exhaustively grepped repo-wide; flagged as likely-dead duplication, not confirmed-unused.)
- **Recommendation:** Confirm usage; if unused, delete _process_inputs_with_coords_impl to avoid two diverging coords paths.

### [MEDIUM] s2c3l0-006 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:109`
- **Evidence:** ConditionedTransitionBlock.forward recomputes the full activation twice: lines 110-117 compute a_norm, linear_a1, linear_a2 and b, then lines 131-134 recompute the identical adaln/linear/SiLU, discarding the first results. Combined with unconditional print() at lines 109,111,113,115,117 (and fallback prints + traceback.format_stack at 149-152), every transformer block forward both doubles its matmul work and floods stdout.
- **Recommendation:** Remove the duplicated computation and all unconditional print/traceback statements; the block should perform adaln + gated SiLU + conditioning once.

### [MEDIUM] s2c3l2-018 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:109-152`
- **Evidence:** forward emits unconditional `print(f"[INSTRUMENT][CTB.forward] ...")` (lines 109,111,113,115,117) and, in the fallback branch, prints a stack trace via `traceback.format_stack` (lines 148-152). Additionally the triple-quoted block at lines 120-129 follows executable code so it is a dead string, not the function docstring.
- **Recommendation:** Remove the INSTRUMENT prints and the traceback dump; relocate the docstring to the top of forward.

### [MEDIUM] s2c3l2-027 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer_patch.py:21-108`
- **Evidence:** patch_transformer monkey-patches AtomAttentionEncoder.forward with patched_forward(self, input_feature_dict, r_l, s, z, ...) (line 21). This signature is incompatible with the shipped encoders: the refactored encoder's forward is forward(self, *args, **kwargs) reading only args[0]/params (atom_attention_encoder.py:105), so r_l/s/z would be ignored, and the patched body references attributes that do not exist on that encoder (self.layernorm_a at line 36, self.layernorm_s at line 40 — those live on AttentionPairBias, not the encoder), so it would AttributeError if ever run. `self.layernorm_a(r_l)` (line 36) also discards its result. patch_transformer is only invoked from this module's own __main__ (line 113); a repo grep found no other caller. The Stage-1 inventory describes this as a 'Patch applied to the transformer module to fix tensor shape compatibility', but it is neither wired in nor functional.
- **Recommendation:** Remove the broken/unused patch module, or rewrite patched_forward against the real encoder API and register it where intended.

### [MEDIUM] s2c3l2-028 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/utils.py:1`
- **Evidence:** A module current/utils.py and a package current/utils/ (with __init__.py) coexist in the same directory. Python resolves the package over the module, so `import ...current.utils` always loads utils/__init__.py and current/utils.py is unreachable dead code. Both files re-export the same Protenix-derived helpers (utils.py:29-55 vs utils/__init__.py:27-50), so the shadowed module is pure redundancy and a maintenance trap (edits to utils.py have no effect).
- **Recommendation:** Delete current/utils.py (the package __init__.py already provides the re-exports).

### [MEDIUM] s2c3l0-015 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:104`
- **Evidence:** broadcast_token_to_atom._perform_gather flattens x_token to (-1, feature_dim) (line 114) and gathers along dim 0 using atom_to_token_idx_flat (line 117), but atom_to_token_idx contains per-batch token indices in [0, N_token) with no batch offset added. For batch_size>1, x_token_flat row b*N_token+idx is the correct source, yet the code uses raw idx, so every batch element reads token features from batch 0. Correct only for B==1.
- **Recommendation:** Add the per-batch offset (idx + batch_index*N_token) before the flat gather, or use torch.gather/index along the token dim without flattening the batch dim away.

### [MEDIUM] s2c3l0-016 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:173`
- **Evidence:** aggregate_atom_to_token branches on production behavior by reading the pytest environment variable: current_test = os.environ.get('PYTEST_CURRENT_TEST') (line 173) and then takes special reshape/scatter-fallback paths only when 'test_run_stageD_basic' / 'test_run_stageD_diffusion_inference_original' appear in it (lines 183, 234, 300). Production correctness thus depends on whether pytest is running; outside tests these recovery paths are disabled and the same shape mismatch will raise.
- **Recommendation:** Remove the PYTEST_CURRENT_TEST coupling; make the shape-normalization logic unconditional and correct for all callers, and move test-only behavior into the tests.

### [MEDIUM] s2c3l0-021 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/legacy/attention/block_sparse.py:81`
- **Evidence:** LocalBlockSparseAttentionNaive implements attention with a Python for-loop over every atom in forward (line 81, `for i in range(N_atom)`) and again in backward (line 129), each iteration gathering neighbors and running softmax. This is O(N_atom) Python iterations per layer with no batching; for non-trivial RNA atom counts it is prohibitively slow and is the default path when use_optimized is False (atom_transformer.py:112-116) and the only path when block_sparse_attn is not installed (block_sparse.py:163-169).
- **Recommendation:** Vectorize the neighbor gather/softmax across atoms (batched matmul over the block window) or require the optimized kernel; at minimum document the severe scaling limit.

### [MEDIUM] s2re3-004 — security
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:70,112-113`
- **Evidence:** download_file() fetches a checkpoint zip via urllib.request.urlopen(url) (:70) from a configurable checkpoint_url (default the dropbox URL in config_schema.py:271) with no checksum/signature verification, then unzip_file() calls zip_ref.extractall(extract_dir) (:112-113) with no member-path sanitization. extractall on an attacker-controlled or MITM'd archive is a classic Zip-Slip path-traversal vulnerability (members named '../...' write outside extract_dir). Distinct from the angles.py DSSR-zip findings already listed; the run_stageA checkpoint download/extract path is uncovered.

### [MEDIUM] s2re2-004 — security
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:70,188-190; rna_predict/training/rna_lightning_module.py:140; rna_predict/conf/config_schema.py:270-271`
- **Evidence:** The RFold checkpoint download (download_file/_download_file using urllib.request.urlopen, run_stageA.py:70 and rna_lightning_module.py:140) writes the response straight to disk with shutil.copyfileobj and performs only a zip-integrity (testzip) check — no cryptographic checksum/signature verification of the downloaded artifact. The default source is a personal Dropbox share 'https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1' (config_schema.py:271), an account-controlled, mutable, non-pinned URL. The unverified zip is extracted and then torch.load'ed (s2re2-003), so a swapped/MITM'd artifact leads to code execution. Distinct from the angles.py security entries already listed.

### [MEDIUM] s2c3l1-002 — security
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:70-93`
- **Evidence:** download_file() fetches the checkpoint with urllib.request.urlopen(url) -> shutil.copyfileobj into dest_path (run_stageA.py:70-71) and performs NO integrity verification (no expected SHA-256/size/signature check); the only validation is zipfile.testzip() for corruption (run_stageA.py:40-43), which does not authenticate contents. The downloaded archive is then extracted (run_stageA.py:191) and the resulting .pth (conf/model/stageA.yaml:11) is loaded downstream by StageARFoldPredictor (instantiated at run_stageA.py:210) via torch.load (pickle), which executes arbitrary code on a malicious/compromised checkpoint. Combined with a config-overridable checkpoint_url, this is a supply-chain RCE path. Default URL is a Dropbox https link (TLS) but there is no pinning or content hash.
- **Recommendation:** Pin and verify a known-good checksum (and ideally signature) of the downloaded archive before extraction; refuse to proceed on mismatch. Load checkpoints with weights_only=True (torch.load) so untrusted pickles cannot execute code, and document the trusted checkpoint source.

### [MEDIUM] s2c3l2-040 — design_defect
- **Location:** `rna_predict/pipeline/stageB/main.py:129`
- **Evidence:** run_stageB_combined hard-codes `requires_grad = False  # Set to False for integration tests` (line 129) and uses it to build init_s (line 132) and init_z_tensor (line 145). A test-oriented setting baked into the production combined-stage function means the single/pair embeddings never carry gradients, which would silently break the 'Experimental' training path that depends on Stage B differentiability.
- **Recommendation:** Drive requires_grad from config/training mode rather than a hard-coded test value.

### [MEDIUM] s2c3l0-023 — design_defect
- **Location:** `rna_predict/pipeline/stageB/main.py:152`
- **Evidence:** run_stageB_combined wraps the pairformer call in try/except Exception (lines 152-162) and on ANY failure fabricates s_up/z_up as all-ones tensors and continues, returning them as 's_embeddings'/'z_embeddings'. This silently converts model errors into plausible-but-meaningless outputs that flow downstream into Stage C, defeating error detection for the inference deliverable.
- **Recommendation:** Let genuine errors propagate (or re-raise after logging); reserve the ones-tensor fallback for an explicit test/dummy mode flag.

### [MEDIUM] s2c3l2-041 — design_defect
- **Location:** `rna_predict/pipeline/stageB/main.py:152-162`
- **Evidence:** The Pairformer forward is wrapped in `try/except Exception` that, on ANY error, logs and fabricates dummy outputs `s_up = torch.ones(...)`, `z_up = torch.ones(...)` (lines 159-162, comment 'Create dummy output for testing'). Swallowing all exceptions and substituting constant tensors masks genuine model/shape failures and yields meaningless downstream embeddings without signalling failure.
- **Recommendation:** Let real errors propagate (or narrow the except and re-raise); do not silently substitute dummy outputs in production.

### [MEDIUM] s2c3l0-022 — design_defect
- **Location:** `rna_predict/pipeline/stageB/main.py:171`
- **Evidence:** run_stageB_combined contains test-mock-specific logic in the production path: `if hasattr(pairformer_model, 'return_value') and isinstance(pairformer_model.return_value, tuple) ...: s_up, z_up = pairformer_model.return_value` (lines 171-175). 'return_value' is a unittest.mock.MagicMock attribute; real models do not have it, but the branch couples production behavior to the test framework and would misbehave if a real object exposed a `return_value` attribute.
- **Recommendation:** Remove the MagicMock-aware branch; rely solely on the actual call output (pairformer_output) and let tests assert on that.

### [MEDIUM] s2c3l2-038 — bug
- **Location:** `rna_predict/pipeline/stageB/main.py:317-322`
- **Evidence:** run_pipeline's empty-sequence handling is unreachable and contradicts its own comment. Line 299 raises ValueError when `len(sequence)==0` (precedence: `(not str and not list) or len==0`). An empty string therefore raises before reaching line 317 `if not sequence: return {coordinates: zeros, atom_count:0}` (lines 317-322), whose comment 'For empty sequences, we still ... return empty tensors' describes behaviour that can never occur.
- **Recommendation:** Decide the contract: either return empty tensors for empty input (drop the len==0 clause at line 299) or raise — and remove the dead branch.

### [MEDIUM] s2c3l2-037 — design_defect
- **Location:** `rna_predict/pipeline/stageB/main.py:366`
- **Evidence:** run_pipeline emits unconditional stdout `print(f"[CASCADE-DEBUG] ...")` at lines 366,368,372 on every run, and several `logger.info` diagnostics are ungated by debug_logging: '[DEBUG-STAGEB] N_token=...' (lines 216-217), '[DEBUG-SEQUENCE-ENTRY-STAGEB]' (line 309), '[DEBUG-SEQUENCE-BEFORE-STAGEB]' (line 365). These flood output for the README-'Functional' inference pipeline.
- **Recommendation:** Remove the CASCADE-DEBUG prints and gate the DEBUG logger.info lines behind debug_logging/logger.debug.

### [MEDIUM] s2c3l2-034 — doc_drift
- **Location:** `rna_predict/pipeline/stageB/pairwise/dummy_pairformer.py:37`
- **Evidence:** DummyPairformerModel.forward returns `torch.randn(1, 32, 32, 64, device=self.device)` (line 37) — RANDOM values — while the Stage-1 inventory describes it as a 'Stub Pairformer nn.Module that returns zero tensors', and the method docstring (line 33) says 'Returns a dummy tensor'. The output is also a hard-coded shape (1,32,32,64) independent of input sequence length, so as a fallback it would not match downstream N-dependent shapes and a random (non-deterministic) stub breaks reproducible test fallbacks.
- **Recommendation:** Return torch.zeros with a shape derived from inputs (or document it as random); reconcile with the 'zero tensors' intent.

### [MEDIUM] s2c4l0-msaconfig-fromdict-drops-hparams — bug
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer.py:92-102`
- **Evidence:** MSAConfig.from_dict only copies enable/strategy/train_cutoff/test_cutoff/train_lowerb/test_lowerb and ignores the model hyperparameters c_m, c, c_z, dropout, n_blocks, n_heads, pair_dropout. MSAModule.__init__ converts a Hydra DictConfig via `MSAConfig.from_dict(dict(cfg))` (line 707), so a DictConfig that specifies custom c_m/c/c_z/dropout/n_blocks silently reverts them to the dataclass defaults (8/8/8/0.1/1). Required-param validation that follows passes because the defaults exist, masking the data loss.
- **Recommendation:** Have from_dict also read c_m/c/c_z/dropout/n_blocks/n_heads/pair_dropout (with the existing defaults as fallbacks), or construct MSAConfig with **dict(cfg) filtered to known fields.

### [MEDIUM] s2c4l2-001 — design_defect
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer.py:983-990`
- **Evidence:** TemplateEmbedder.forward unconditionally returns torch.zeros_like(z) on every path (both the early-return at :984 and the final return at :990), with a TODO at :988 ('Implement the actual template embedding logic here when ready'). The __init__ (:940-957) still constructs linear_no_bias_z/a/u, layernorm_z/v and a full PairformerStack (self.pairformer_stack) that are never used in forward. Stage-1 intent (README/architecture) lists template embedding as part of the AF3-inspired pairwise branch, but it is a no-op stub.
- **Recommendation:** Either implement the template embedding logic or remove the unused submodules and document TemplateEmbedder as an intentional zero-output placeholder so callers do not assume template conditioning is active.

### [MEDIUM] s2c4l2-004 — design_defect
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:250-255`
- **Evidence:** The __init__ docstring (:49) claims it '...optionally freezes model parameters or prepares for LoRA integration if enabled.' The LoRA branch logs 'LoRA enabled for Pairformer (r=...). Applying LoRA layers...' (:253) but the actual application is a placeholder comment followed by `pass` (:254-255). No LoRA layers are applied, so enabling lora in config silently does nothing while logging success.
- **Recommendation:** Implement LoRA application or change the log message and docstring to state LoRA is not yet supported for the Pairformer; raise/warn if lora.enabled is set.

### [MEDIUM] s2c4l0-torsionbert-lora-target-check — bug
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:374-380`
- **Evidence:** found_any is computed via `for tm in target_modules: if hasattr(self.model, tm): found_any=True`. target_modules are nested submodule name patterns (e.g. 'query','value' as configured in lora_param_count.py:15) that match Linear layers deep inside the transformer, not top-level attributes of self.model. hasattr(model,'query')/hasattr(model,'value') is False, so found_any stays False and the whole LoRA wrapping block (lines 380-389) is skipped, leaving lora_applied=False even when LoRA is enabled and PEFT is installed. lora_param_count.py would consequently report 0 LoRA trainable params.
- **Recommendation:** Drop the hasattr top-level check (let PEFT resolve target_modules), or detect targets by scanning self.model.named_modules() for matching suffixes.

### [MEDIUM] s2c4l0-torsionbert-tokenization-divergence — intent_mismatch
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:443-449`
- **Evidence:** StageBTorsionBertPredictor._preprocess_sequence tokenizes the raw RNA sequence string directly (self.tokenizer(sequence, ...)), whereas the sibling TorsionBertModel (torsionbert_inference.py:367-394,519-525) for the same 'sayby/rna_torsionbert' (DNABERT-style) model first uppercases, replaces U->T, and builds space-separated 3-mer k-mers before tokenizing. If the predict.py inference path uses StageBTorsionBertPredictor, the model receives a tokenization that diverges from the k-mer scheme the model was trained on, which can yield incorrect torsion predictions. (Which predictor is canonical for inference is unverified from these files alone.)
- **Recommendation:** Align StageBTorsionBertPredictor tokenization with the k-mer preprocessing used by TorsionBertModel, or document why raw-string tokenization is correct for this checkpoint.

### [MEDIUM] s2c4l2-014 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:443-449,667`
- **Evidence:** Two tokenization conventions exist for the same 'sayby/rna_torsionbert' model. StageBTorsionBertPredictor tokenizes the raw sequence string directly (:443-449) and slices angle_preds[:,1:num_residues+1,:] assuming one token per residue plus a leading CLS (:667). TorsionBertModel in torsionbert_inference.py:380-394 instead builds 3-mer k-mers (_build_tokens k=3) and reconciles k-mer outputs to residues (_fill_result). These divergent conventions for the same model can produce misaligned per-residue angles depending on which class the pipeline uses.
- **Recommendation:** Standardize on the model's documented tokenization (TorsionBERT uses 3-mer per docs/pipeline/stageB/torsionbert_code.md) in the primary predictor, and remove/align the divergent path.

### [MEDIUM] s2c4l2-012 — design_defect
- **Location:** `rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:18-19,43,63,124,128`
- **Evidence:** DummyTorsionBertAutoModel emits unconditional stdout: print('[DEBUG-DUMMY-INIT]...') and traceback.print_stack(limit=5) on every construction (:18-19), and print('[DEBUG-DUMMY-FWD]...') plus angle_mode/output-shape prints on every forward (:43,:63,:124,:128). None are gated by debug_logging, so this floods stdout whenever the dummy/fallback model is active (which happens on any model-load failure, not only in tests).
- **Recommendation:** Gate these prints behind self.debug_logging or remove them; rely on logger.debug.

### [MEDIUM] s2c4l2-016 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/ml_utils.py:1-19`
- **Evidence:** Both a module file ml_utils.py and a package directory ml_utils/ exist in mp_nerf/ (confirmed via ls), and likewise rna.py and rna/ (rna.py:1-53). Python's FileFinder resolves the package directory before the same-named .py, so ml_utils.py and rna.py are shadowed/unreachable on import. Each shim duplicates the re-export surface of its corresponding package (rna.py:8-29 vs rna/__init__.py:7-20), so they are dead code masquerading as the public module.
- **Recommendation:** Delete the shadowed ml_utils.py and rna.py shim files (the packages already provide the re-exports), or rename them to avoid the collision.

### [MEDIUM] s2c4l2-021 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/ml_utils/atom_utils.py:2`
- **Evidence:** Module docstring reads 'Atom manipulation utilities for RNA structure prediction' but the entire module operates on proteins: amino-acid indices (AAS2INDEX/INDEX2AAS/AMBIGUOUS, SUPREME_INFO), the SidechainNet 14-atom-per-residue layout (ATOM_MASKS sized 14, res_idx*14 in :393-394), glycine CB handling (:211-225), and protein scn_cloud_mask (:20,128). The same 'for RNA structure prediction' header appears on protein-only modules coordinate_transforms.py:2, loss_functions.py:2, and tensor_ops.py:30-49 (c=14, sidechain_fold). This is vendored protein MP-NeRF code re-labeled as RNA.
- **Recommendation:** Correct the module docstrings to reflect that these are protein/SidechainNet utilities, or quarantine/remove the protein code if it is not part of the RNA Stage-C path.

### [MEDIUM] s2c4l2-022 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/ml_utils/coordinate_transforms.py:294-374`
- **Evidence:** noise_internals_legacy (the symbol actually re-exported by ml_utils/__init__.py:17) is documented as 'Noises the internal coordinates -> dihedral and bond angles' (:300), but the implementation never touches internal coordinates: it builds a default coords tensor and only adds Cartesian Gaussian noise (cloud = cloud + randn*noise_scale, :370-372). The full internal-noising path lives in the unexported noise_internals(config) (:246-275, which calls protein_fold). Likewise combine_noise_legacy (:796-844) ignores sidechain_reconstruct/internals and just adds coordinate noise despite its docstring.
- **Recommendation:** Make the exported legacy functions implement (or delegate to) the documented internal-coordinate noising, or update their docstrings to state they apply Cartesian noise only.

### [MEDIUM] s2c4l2-023 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/protein_utils/structure_utils.py:472`
- **Evidence:** protein_fold is implemented twice with different signatures and algorithms: proteins.py:268 (vectorized, takes cloud_mask/point_ref_mask/angles_mask/bond_mask) and structure_utils.py:472 (residue-by-residue, takes seq/angles). protein_utils/__init__.py:53 exports the structure_utils version while coordinate_transforms.py:19-23 imports the proteins.py version. The same duplication-with-divergence affects build_scaffolds_from_scn_angles, modify_scaffolds_with_coords, and modify_angles_mask_with_torsions (proteins.py:199 takes (seq,angles_mask,torsions) vs scaffold_builders.py:229 takes (angles_mask,torsions)), and the scn_* mask helpers (proteins.py:31-143 vs mask_generators.py:115-219 vs massive_pnerf.py:193). Same-named functions with divergent behavior are a wrong-import hazard.
- **Recommendation:** Consolidate each function to a single canonical implementation and import it everywhere; remove the duplicates.

### [MEDIUM] s2c4l2-025 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/protein_utils/supreme_data.py:28-53`
- **Evidence:** SUPREME_INFO (consumed by proteins.scn_cloud_mask/scn_angle_mask/scn_index_mask and the mask_generators) is generated from explicitly synthetic placeholder helpers: generate_mask/generate_bool_mask fill the first num_atoms positions (:31-40) and generate_idx_mask comments 'This is NOT biochemically accurate but fills the shape' (:43-52), header note 'more realistic placeholders' (:6). Any protein reconstruction driven by SUPREME_INFO therefore uses non-physical geometry/reference indices.
- **Recommendation:** Replace the placeholder generators with the real SidechainNet SUPREME_INFO data, or remove the protein reconstruction path if it is not used by the RNA pipeline.

### [MEDIUM] s2c4l2-020 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:267,315`
- **Evidence:** place_rna_bases sets torsion_angle = 0.0 unconditionally for every base atom placement, both in the OP1/OP2 branch (:267) and the default branch (:315), then passes it to calculate_atom_position. The function docstring (:24-36) and Stage-1 intent describe building base atoms 'using geometry if possible', but no per-atom dihedral is ever supplied, so all base atoms are placed at a fixed zero torsion rather than from real base geometry/torsions.
- **Recommendation:** Source per-atom torsion angles from BASE_GEOMETRY/connectivity or precomputed values instead of hardcoding 0.0, or document that base placement is intentionally planar/approximate.

### [MEDIUM] s2c4l0-rna-base-placement-fixed-torsion — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:267,315,324-331`
- **Evidence:** All non-backbone base atoms are placed with torsion_angle hard-coded to 0.0, and the three NeRF reference atoms used are not the chemically-bonded dihedral references but simply the last two list-ordered placed atoms (prev1=placed_atoms[prev_atoms[-1]], prev2=placed_atoms[prev_atoms[-2]]) plus the bonded ref_atom. With torsion fixed at 0 and arbitrary a/b atoms, base ring atoms are reconstructed in an approximate/planar arrangement that does not reflect real nucleobase geometry, degrading the Stage-C atomic output the pipeline is meant to produce.
- **Recommendation:** Use proper per-atom dihedral references and torsion values from the RNA geometry KB (final_kb_rna) rather than a constant 0.0 and list-order neighbors.

### [MEDIUM] s2c4l2-018 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_constants.py:29-37`
- **Evidence:** RNA_BACKBONE_TORSIONS_AFORM is defined twice with materially different values and conventions. rna_constants.py:29-37 uses signed degrees {alpha:-60, beta:180, gamma:60, delta:80, epsilon:-150, zeta:-70, chi:-160}; final_kb_rna.py:183-191 uses 0-360 degrees {alpha:300, beta:180, gamma:50, delta:85, epsilon:180, zeta:290} with no chi. epsilon differs by ~330deg (180 vs -150), gamma 50 vs 60, delta 85 vs 80. rna/__init__.py re-exports the rna_constants copy while final_kb_rna.get_backbone_torsion serves the other, so two disagreeing sources of truth feed reconstruction.
- **Recommendation:** Consolidate to a single canonical A-form torsion table (one convention) and have both modules import it; reconcile the epsilon/gamma/delta discrepancies against literature.

### [MEDIUM] s2c5l2-002 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:229-248`
- **Evidence:** `rna_fold(..., do_ring_closure=...)` only logs '[INFO-RNAFOLD] Ring closure requested. Placeholder: not yet fully implemented.' with the real call commented out (line 231), and `ring_closure_refinement` (242-248) just warns NOTIMPL and returns coords unchanged. The Stage C Hydra schema exposes `do_ring_closure` as a real bool option (stage_c_reconstruction.py:147, create_stage_c_test_config:197) and passes it down (stage_c_reconstruction.py:251,321), so a user enabling it gets a silent no-op rather than ring closure. Provisional intent (Stage C = forward-kinematics reconstruction to atomic coordinates) implies geometry refinement is part of the deliverable.
- **Recommendation:** Either implement ring closure or mark do_ring_closure as unsupported in the config schema/docs and raise/ warn loudly when set True instead of silently no-op.

### [MEDIUM] s2c5l2-003 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:43-169`
- **Evidence:** build_rna_chain_from_internal_coords only consumes scaffolds["torsions"] and recomputes all geometry via get_bond_length/get_bond_angle/get_torsion_angle_index. The bond_mask, angles_mask, point_ref_mask, cloud_mask precomputed by build_scaffolds_rna_from_torsions (rna_scaffolding.py:49-124) are never read here; only place_rna_bases later uses angles_mask. Two parallel, divergent geometry-resolution conventions coexist (MP-NeRF mask tensors vs ad-hoc per-atom lookups), wasting computation and risking divergence.
- **Recommendation:** Have the chain builder consume the precomputed scaffold masks (the MP-NeRF design intent) or drop the unused mask construction from build_scaffolds_rna_from_torsions to remove the dead second convention.

### [MEDIUM] s2c5l2-004 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_scaffolding.py:84-93`
- **Evidence:** build_scaffolds_rna_from_torsions multiplies each torsion by (math.pi/180.0) when filling angles_mask[1,...], i.e. it assumes the input torsions are in DEGREES. But build_rna_chain_from_internal_coords (rna_folding.py:25-38,136,165) treats scaffolds["torsions"] directly as RADIANS ('Torsions are expected in radians'), and the Stage C config default angle_representation is 'radians' (stage_c_reconstruction.py:171,463; create_stage_c_test_config:200). The two code paths assume contradictory units for the same `torsions` tensor; the deg->rad conversion in scaffolding is silently inconsistent with the radians contract used by the actual chain builder.
- **Recommendation:** Pick one unit convention, honor angle_representation explicitly in both build_scaffolds_rna_from_torsions and the chain builder, and assert/convert consistently.

### [MEDIUM] s2c5l2-006 — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/utils.py:385-411`
- **Evidence:** In save_structure's 3D branch, the outer guard `if coords.shape[1] != 3 or coords.shape[2] != 3:` is entered only when at least one of those dims != 3, yet the inner block immediately tests `if coords.shape[1] == 3 and coords.shape[2] == 3:` which can never be true there — so the documented default `atom_types = ['N','CA','C']` (line 390) is dead code, and the docstring claim 'If 3D and atom_types is None, defaults to [N,CA,C]' (lines 344-349) never happens. Any 3D coords whose atoms-per-residue != 3 fall through to `else: raise ValueError('shape (N,3,3)')` (line 396), and even (N,3,3) with atom_types=None raises at line 401-404. RNA reconstruction outputs (L, max_atoms, 3) with max_atoms>3 therefore cannot be saved, and the defaults are protein backbone names, contradicting the RNA intent.
- **Recommendation:** Fix the branch logic so the (N,3,3) default path is reachable, and make defaults RNA-aware (or require atom_types); update the docstring to match real behavior.

### [MEDIUM] s2c5l2-010 — bug
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:168-169,259-261,453-472`
- **Evidence:** validate_stageC_config allows device in ['auto','cpu','cuda','mps'] (line 168) and run_stageC defaults device to 'auto' when cfg is None (line 459). But run_stageC_rna_mpnerf passes device straight into build_scaffolds_rna_from_torsions (torch.zeros(..., device=device)) after only warning 'Unsupported device ... Proceeding anyway' for non cpu/cuda/mps (lines 260-261); torch will raise on device='auto'. Likewise StageCReconstruction.__init__ does torch.device(device) (line 93) which fails for 'auto'. So 'auto' is accepted by validation and is the no-cfg default yet is unusable at runtime.
- **Recommendation:** Resolve 'auto' to a concrete device (cuda if available else cpu) before constructing tensors, or remove 'auto' from the allowed/default device set.

### [MEDIUM] s2c5l0-008 — bug
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:347-367`
- **Evidence:** When place_bases is False, coords_full = coords_bb is backbone-only with shape (L, len(BACKBONE_ATOMS)=10, 3), so max_atoms=10. But the atom mask is built from STANDARD_RNA_ATOMS[res] (full atom sets, typically >10 atoms per residue): for each residue valid_atom_mask gets len(atom_list) True entries and only pads with False when len(atom_list) < coords_full.shape[1] (line 364). With len(atom_list) > 10, no padding occurs, so the per-residue mask length exceeds max_atoms and total mask length != L*max_atoms. The boolean index `coords_full.reshape(L*max_atoms, D)[mask]` (line 367) then fails with a mask-size mismatch (IndexError). The place_bases=False code path is therefore broken.
- **Recommendation:** Build the atom-name/valid_atom_mask lists from the actual atom set used to produce coords_full (backbone-only when place_bases is False), or assert/raise a clear error when STANDARD_RNA_ATOMS counts exceed coords_full.shape[1].

### [MEDIUM] s2c5l0-009 — design_defect
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:96-112,485-488`
- **Evidence:** StageCReconstruction.__call__ ignores the input torsion angles and returns all-zero coordinate tensors (`coords = torch.zeros((N*3,3))`, `coords_3d = torch.zeros((N,3,3))`) with empty atom_metadata. run_stageC dispatches to this when cfg.model.stageC.method == 'legacy' (line 485-488), and 'legacy' is an accepted method value (validate_stageC_config line 165). Selecting the legacy method thus silently yields physically meaningless (zero) structures rather than an error.
- **Recommendation:** Either raise NotImplementedError for the 'legacy' method, or remove 'legacy' from the accepted method set in validate_stageC_config so it cannot be silently selected.

### [MEDIUM] s2c5l2-013 — design_defect
- **Location:** `rna_predict/pipeline/stageD/config.py:108-134`
- **Evidence:** DiffusionConfig ships top-level dims c_atom=4, c_s=8, c_z=4, c_s_inputs=8, c_noise_embedding=4 and feature_dimensions.s_inputs=8 (lines 105,117-121), which are toy/test sizes that directly contradict the nested ModelConfig defaults c_token=768, c_s=384, c_z=128, c_s_inputs=32 (lines 74-77). The same logical dimensions are defined twice with conflicting values inside one structured config, so which one wins depends entirely on which path the code reads (DiffusionModule reads model_architecture/ModelConfig, bridging reads feature_dimensions).
- **Recommendation:** Define each diffusion dimension in exactly one place and reference it; remove the duplicated toy-valued top-level c_* / feature_dimensions fields or make them interpolations of the model block.

### [MEDIUM] s2c5l0-023 — perf
- **Location:** `rna_predict/pipeline/stageD/diffusion/bridging/residue_atom_bridge.py:188-285`
- **Evidence:** _process_pair_embedding expands residue-level pair tensors to atom-level by allocating an [.., n_atom, n_atom, C] tensor (e.g. value.new_zeros((B, n_atom, n_atom, C)) line 189/233/273) and filling it with quadruple-nested Python loops over (residue_i x residue_j x atom_i x atom_j) (lines 194-200, 236-241, 276-281). For real RNA where n_atom is tens of times n_res, this is O(n_atom^2 * C) memory and O(n_atom^2) Python-level iterations, the same quadratic blowup the diffusion_module forward comments try to guard against.
- **Recommendation:** Vectorize the residue->atom pair expansion (e.g., index_select/gather with atom_to_token_idx broadcasting) and avoid materializing dense n_atom x n_atom pair tensors where the diffusion attention can consume a bias built on the fly.

### [MEDIUM] s2c5l2-047 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/diffusion/bridging/residue_atom_bridge.py:505-515`
- **Evidence:** _process_one_trunk_embedding, when it cannot find expected feature dims in config, falls back to hardcoded values commented 'Default from stageD_diffusion.yaml': s_trunk=384, s_inputs=449, sing=384 (lines 507-515). These magic constants contradict the registered structured schema (config.py FeatureDimensionsConfig.s_inputs=8, ModelConfig.c_s=384/c_s_inputs=32) and reference a yaml whose values differ, so the bridging silently adjusts feature dims to numbers that are not the actual config's, masking misconfiguration.
- **Recommendation:** Resolve dims solely from the live config and raise on absence; remove the hardcoded 384/449/256 fallbacks (or wire them to the schema).

### [MEDIUM] s2c5l2-024 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py:163-185,369-376`
- **Evidence:** DiffusionConditioning rebuilds nn modules with fresh random weights inside forward when dims mismatch: it replaces self.layernorm_z with `LayerNorm(actual_z_dim)` (line 171), self.linear_no_bias_z with a new LinearNoBias (lines 180-183), and self.linear_no_bias_s with a new LinearNoBias (lines 373-376). Re-instantiating layers during forward discards any learned/loaded parameters and reinitializes them every mismatching call, which is incompatible with training and checkpoint loading and is non-deterministic across inputs.
- **Recommendation:** Size these layers correctly at __init__ from config and assert input dims in forward, instead of reconstructing modules at forward time.

### [MEDIUM] s2c5l0-012 — bug
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:1097-1120`
- **Evidence:** _calculate_edm_scaling_factors computes c_skip = 1/(sigma^2+1), c_out = sigma*c_skip = sigma/(sigma^2+1), c_in = 1/(sigma*(sigma^2+1)^0.5 + 1e-8). Karras/EDM preconditioning (with sigma_data) is c_skip = sigma_data^2/(sigma^2+sigma_data^2), c_out = sigma*sigma_data/sqrt(sigma^2+sigma_data^2), c_in = 1/sqrt(sigma^2+sigma_data^2). Even taking sigma_data=1, c_out here is off by a factor of sqrt(sigma^2+1) (uses sigma^2+1 instead of its square root) and c_in is off by an extra factor of sigma (blows up as sigma->0). DiffusionConditioning is constructed with sigma_data (diffusion_module.py:269) but self.sigma_data is never used in these scaling factors, so the EDM denoising/preconditioning is numerically incorrect.
- **Recommendation:** Implement the standard EDM preconditioning using sigma_data: c_skip=sigma_data^2/(sigma^2+sigma_data^2), c_out=sigma*sigma_data/sqrt(sigma^2+sigma_data^2), c_in=1/sqrt(sigma^2+sigma_data^2).

### [MEDIUM] s2c5l0-013 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:779-855`
- **Evidence:** forward() first enforces `if x_noisy.ndim not in (4,): raise ValueError(...)` (lines 779-780), guaranteeing 4D input. Yet lines 827-855 then re-handle `if x_noisy.ndim == 3: ... elif x_noisy.ndim == 4: ... else: raise`, and the comment at line 828 still claims inputs may be [B, N_atom, 3]. The 3D branch (lines 829-841) is unreachable dead code given the earlier hard 4D check, and the two shape contracts contradict each other.
- **Recommendation:** Pick one contract: either remove the strict 4D guard and keep the 3D/4D normalization, or delete the now-unreachable 3D branch and stale comments.

### [MEDIUM] s2c5l0-014 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:950-956,1021-1028`
- **Evidence:** forward() inspects the caller's stack frame via get_caller_frame() and, when the caller function name contains 'test_n_sample_handling', returns only x_denoised (skipping loss) (lines 952-956). _compute_loss does the same to short-circuit and return a dummy zero loss (lines 1023-1028). Production model behavior (return arity and loss computation) is thus conditioned on the name of the calling test function.
- **Recommendation:** Remove caller-frame/test-name introspection; control single-vs-tuple return and loss computation via explicit parameters, and move test-specific expectations into the tests.

### [MEDIUM] s2c5l2-026 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_utils.py:42-103`
- **Evidence:** validate_tensor_shapes silently pads-with-zeros or truncates the feature dimension of s_trunk/s_inputs to match config c_s/c_s_inputs (lines 65-97), and if it cannot extract those dims from config it falls back to a hardcoded 32 (lines 50,51,59,61). Silently zero-padding/truncating learned embeddings to a guessed dimension corrupts features without error, and the magic 32 fallback hides config misconfiguration.
- **Recommendation:** Raise on dimension mismatch (or require explicit config dims) rather than silently mutating feature dimensions; remove the hardcoded 32 fallback.

### [MEDIUM] s2c5l2-033 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:143-153,186-214`
- **Evidence:** ProtenixDiffusionManager.__init__ contains PYTEST_CURRENT_TEST 'Special case for test_init_with_basic_config' blocks (lines 144-153, 187-214) that log/patch diffusion_args specifically for that test. Production initialization branches on the test harness environment variable.
- **Recommendation:** Remove test-name-specific handling from the manager; configure such cases via test fixtures.

### [MEDIUM] s2c5l2-030 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:158,425-443`
- **Evidence:** The manager defaults the diffusion step count to 2: self.num_inference_steps = inference_cfg.get('num_steps', 2) (line 158) and multi_step_inference sets inference_cfg['num_steps']=2 when missing (lines 425-442). The structured config default is InferenceConfig.num_steps=100 (config.py:34). When num_steps is absent from the resolved config, inference silently runs only 2 denoising steps instead of the schema-intended 100, badly under-running the diffusion refinement.
- **Recommendation:** Align the fallback with the schema default (100) or, better, require num_steps from config and fail loudly if missing.

### [MEDIUM] s2c5l0-017 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:317-323`
- **Evidence:** _get_noise_schedule returns `torch.linspace(1.0, 0.0, steps=num_steps+1)` for both 'linear' and any unknown schedule_type. multi_step_inference uses this directly (line 510), so the diffusion sampling noise schedule always runs from 1.0 down to 0.0 regardless of config. The purpose-built EDM InferenceNoiseScheduler in generator.py (which honors s_max=160, s_min, p, and sigma_data) is never instantiated/used by the manager, and sample_diffusion initializes x_l with noise_schedule[0]=1.0. The trained noise scale (sigma_data=16 per config.py/generator.py defaults) is ignored at inference.
- **Recommendation:** Use InferenceNoiseScheduler (or an EDM schedule honoring s_max/s_min/p/sigma_data from config) to build the noise schedule, and route schedule_type accordingly.

### [MEDIUM] s2c5l2-032 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:491-498`
- **Evidence:** When stage_cfg.require_atom_level_pairs is True, the code logs 'Bridging z_trunk ... using _process_pair_embedding' but the actual bridging call is commented out and the block ends in `pass  # TODO: Provide residue_atom_map and call bridging here`. So enabling require_atom_level_pairs is a no-op that logs as if it acted — z_trunk stays residue-level.
- **Recommendation:** Implement the atom-level pair bridging or remove/guard the require_atom_level_pairs option and raise NotImplementedError when set.

### [MEDIUM] s2c5l0-018 — intent_mismatch
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:491-498`
- **Evidence:** multi_step_inference guards `if stage_cfg.get('require_atom_level_pairs', False):` then logs 'Bridging z_trunk ... using _process_pair_embedding' but the body is a commented-out call followed by `pass  # TODO: Provide residue_atom_map and call bridging here`. If a config sets require_atom_level_pairs=True, no bridging occurs; z_trunk remains residue-level and is passed unchanged to sample_diffusion, causing a silent shape mismatch downstream instead of the requested atom-level pair bridging.
- **Recommendation:** Implement the atom-level pair bridging (supply residue_atom_map and call _process_pair_embedding) or raise NotImplementedError when require_atom_level_pairs is True, rather than silently no-op'ing.

### [MEDIUM] s2c5l2-036 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/config_types.py:14-39`
- **Evidence:** There are two distinct classes both named DiffusionConfig: the Hydra structured schema in rna_predict/pipeline/stageD/config.py (lines 108-141) and this runtime data container holding partial_coords/trunk_embeddings (config_types.py). diffusion/config.py re-exports the runtime one (`from .utils.config_types import DiffusionConfig`). The name collision between a config schema and a runtime payload object is confusing and error-prone for imports.
- **Recommendation:** Rename the runtime container (e.g. DiffusionRunInputs / StageDDiffusionRequest) to disambiguate from the Hydra DiffusionConfig schema.

### [MEDIUM] s2c5l2-039 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/config_utils.py:171-291`
- **Evidence:** parse_diffusion_module_args has two divergent code paths keyed on PYTEST_CURRENT_TEST (is_test, line 181-182): under test it constructs a flattened dict of model_architecture-derived params (lines 198-288), while in production it simply returns base_cfg unchanged (line 291). The DiffusionModule init it feeds therefore receives a different config shape under test vs production, so tests validate a structure production never sees.
- **Recommendation:** Use one config-shaping path for both test and production; remove the PYTEST_CURRENT_TEST branch.

### [MEDIUM] s2c5l0-025 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/config_utils.py:180-291`
- **Evidence:** parse_diffusion_module_args checks `is_test = os.environ.get('PYTEST_CURRENT_TEST') != ''` (lines 181-182) and, when in a test, builds and returns a plain dict of diffusion_module_args (lines 198-288); otherwise it returns the raw nested base_cfg (DictConfig) (line 291). Thus DiffusionModule receives a structurally different config object (dict with flattened c_atom/c_z/... vs nested DictConfig) depending on whether pytest is running, so tests validate a code path that production never executes.
- **Recommendation:** Produce one consistent config representation for DiffusionModule regardless of environment; remove the PYTEST_CURRENT_TEST branch and have DiffusionModule consume a single canonical shape.

### [MEDIUM] s2c5l2-038 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/embedding_utils.py:54,61,94,99`
- **Evidence:** ensure_s_inputs falls back to c_s_inputs=449 (lines 54,61) and ensure_z_trunk falls back to c_z=128 (lines 94,99) as hardcoded magic numbers when config lacks the value. 449 has no basis in the structured schema (config.py FeatureDimensionsConfig.s_inputs=8, ModelConfig.c_s_inputs=32), so the fallback fabricates a dimension inconsistent with the registered config, silently masking missing config.
- **Recommendation:** Raise when the dimension cannot be resolved from config instead of inventing 449/128; or derive defaults from the schema.

### [MEDIUM] s2c5l2-044 — design_defect
- **Location:** `rna_predict/pipeline/stageD/memory_optimization/memory_fix.py:78-98`
- **Evidence:** apply_memory_fixes mutates config keys 'conditioning' and 'manager' (lines 89-96, setting hidden_dim/num_layers), but the registered Stage D schema (config.py) has no 'conditioning' or 'manager' groups — it uses model/transformer/atom_encoder/atom_decoder. These branches operate on phantom config sections that never exist in the real Hydra config, so the corresponding 'memory fixes' are dead and the real architecture (e.g. model.num_layers/transformer.n_blocks) is left untouched.
- **Recommendation:** Update apply_memory_fixes to target the actual schema keys (transformer.n_blocks/n_heads, model.num_layers) and drop the conditioning/manager phantom branches.

### [MEDIUM] s2c5l0-010 — design_defect
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:318-366`
- **Evidence:** run_stageD branches on `os.environ.get('PYTEST_CURRENT_TEST')` and specific test names ('test_run_stageD_basic', 'test_run_stageD_with_debug_logging', 'test_gradient_flow_through_stageD') to short-circuit and return a hand-built differentiable dummy `{'coordinates': total.expand(batch_size)}` instead of running the real Stage D pipeline. Production behavior is conditioned on the test harness, so tests exercise a fake path while real callers take a different code path.
- **Recommendation:** Remove test-environment special casing from production code; move dummy/gradient-check fixtures into the test suite (e.g., via dependency injection or test doubles).

### [MEDIUM] s2c5l2-016 — design_defect
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:51-68`
- **Evidence:** At import time, run_stageD.py runs a '--- PATCH: Configure all relevant loggers ---' block that force-sets five Stage A input-embedding loggers to DEBUG and attaches StreamHandlers, unconditionally and regardless of any config. Importing Stage D thus mutates global logging for unrelated Stage A modules and spews their debug output.
- **Recommendation:** Remove this import-time logging patch; configure logging via the central config/debug_logging path instead of side effects in module import.

### [MEDIUM] s2c6l2-004 — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:145,344`
- **Evidence:** Two sibling feature initializers in the same module disagree on the 'profile' feature shape: _init_feature_tensors builds features['profile'] as [batch, num_atoms, profile_dim] (:145) while initialize_features_from_config builds features['profile'] as [batch, num_residues, profile_size] (:344). Downstream consumers cannot rely on a consistent profile rank/length depending on which path produced it.
- **Recommendation:** Define a single canonical shape for 'profile' (atom-level vs residue-level) and make both initializers agree.

### [MEDIUM] s2c6l2-003 — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:48-62`
- **Evidence:** _validate_atom_metadata reaches into its caller's stack frame via inspect.currentframe().f_back and reads a local variable literally named 'config' (lines 52-58) to recover atom_metadata when None is passed. This reflection hack silently breaks if the caller renames the variable or is invoked indirectly, and hides the true data dependency.
- **Recommendation:** Pass config/atom_metadata explicitly as parameters; remove the frame-introspection fallback.

### [MEDIUM] s2c6l0-featutils-frame-hack — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:48-74`
- **Evidence:** _validate_atom_metadata reaches into the caller's stack frame via inspect.currentframe().f_back and reads frame.f_locals['config'] (lines 52-58) to recover atom_metadata. This couples the function to the variable naming of arbitrary callers and breaks silently if the caller renames 'config' or is wrapped. Line 73 also does num_residues = max(residue_indices)+1, which raises ValueError on an empty residue_indices list.
- **Recommendation:** Pass config/atom_metadata explicitly as parameters instead of frame introspection; guard max() against empty input.

### [MEDIUM] s2c6l0-init-rearrange-clobber — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:319-344,392-415,418-427`
- **Evidence:** Both fix_rearrange_qk_to_dense_trunk() and fix_rearrange_to_dense_trunk() assign to the same attribute torch.rearrange (a non-standard attribute they invent). apply_tensor_fixes() calls them in sequence (lines 422,426), so the second assignment unconditionally clobbers the first; the qk variant is unreachable after apply. Whichever caller expects torch.rearrange to be the qk version gets the wrong function.
- **Recommendation:** Use two distinct names/targets and patch the actual originating functions in their modules rather than stashing both on torch.rearrange.

### [MEDIUM] s2c6l2-009 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:319-344,392-415,422-426`
- **Evidence:** fix_rearrange_qk_to_dense_trunk (:344) and fix_rearrange_to_dense_trunk (:415) both assign to torch.rearrange, and apply_tensor_fixes calls both, so the second silently clobbers the first. Both wrappers deliberately discard all-but-the-first element of the real function's tuple return ('the test expects just a tensor', :336-341,:409-412), so the padding/mask outputs needed by real callers are dropped.
- **Recommendation:** Do not attach helpers to torch.rearrange; expose them under distinct names and return full tuples. Stop shaping production code to match test expectations.

### [MEDIUM] s2c6l2-010 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:347-367`
- **Evidence:** fix_linear_forward() globally overrides torch.nn.Linear.forward (:367) with a signature adding bogus weight=None, bias=None params (:353) and a manual N-D reshape that nn.Linear already performs natively. The override is redundant at best and process-wide at worst.
- **Recommendation:** Remove this patch; torch.nn.Linear already supports arbitrary leading dimensions.

### [MEDIUM] s2c6l0-attn-mha-positional — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/attention_fixes.py:44-65`
- **Evidence:** patched_attn_forward replaces torch.nn.MultiheadAttention.forward globally and, in the error branch, assumes args[0],args[1],args[2] are q,k,v and slices them along dim 1. MultiheadAttention.forward also accepts query/key/value as keywords and many other positional args (key_padding_mask, need_weights, attn_mask, ...); callers using keywords hit IndexError, and slicing seq len silently corrupts attention.
- **Recommendation:** Do not globally patch MultiheadAttention.forward; if needed, subclass and handle named arguments explicitly without truncating sequence length.

### [MEDIUM] s2c6l0-diff-tokenidx-max — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/diffusion_fixes.py:49-55`
- **Evidence:** fix_token_indices_after_resize() assumes self.token_indices is a dict keyed by 's_inputs'/'s_trunk'/'z_trunk' and calls self.token_indices[key].max(); if a key is absent (KeyError) or the tensor is empty (max() raises) the patched DiffusionConditioning.forward throws after the original already ran. The clamp also silently rewrites learned/used indices.
- **Recommendation:** Guard with key-existence and numel()>0 checks, and verify token_indices structure matches assumption before clamping.

### [MEDIUM] s2c6l2-014 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/diffusion_fixes.py:62-97`
- **Evidence:** fix_trunk_feature_dimensions silently truncates s_inputs/s_trunk to the smaller feature dim (min_dim) on mismatch (:82-85), masking config/embedding-dimension errors. It also patches DiffusionConditioning.forward imported from stageD.diffusion.diffusion (:66-68), while fix_token_indices_after_resize patches DiffusionConditioning.forward imported from diffusion.components.diffusion_conditioning (:14-18); if these resolve to the same class the patches stack ambiguously.
- **Recommendation:** Validate and assert feature dims rather than truncating; consolidate the two DiffusionConditioning patches and confirm they target distinct classes.

### [MEDIUM] s2c6l0-diff-trunk-truncate — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/diffusion_fixes.py:80-85`
- **Evidence:** fix_trunk_feature_dimensions() silently slices s_inputs and s_trunk to min(last_dim) when feature dims disagree, discarding channels. This hides a real conditioning-dimension mismatch and would feed truncated embeddings into the diffusion conditioner, degrading output rather than failing fast.
- **Recommendation:** Treat a feature-dim mismatch as a configuration error (raise) or project via a learned linear layer, not by truncation.

### [MEDIUM] s2c6l2-015 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/embedding_fixes.py:53-55,76-78`
- **Evidence:** fix_broadcast_token_to_atom (:53-55) and fix_batched_gather (:76-78) silently torch.clamp out-of-range indices into valid range before gathering. This converts an index-out-of-bounds bug into a silently-wrong gather (wrong atom/token mapping) instead of surfacing the error.
- **Recommendation:** Raise on out-of-range indices (or fix the index source); do not clamp-and-continue in a bridging/gather path.

### [MEDIUM] s2c6l2-017 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:41,attention_fixes.py:14,diffusion_fixes.py:10,embedding_fixes.py:10,transformer_fixes.py:8`
- **Evidence:** apply_tensor_fixes() in tensor_fixes/__init__.py (:418-427) only calls the fix_* functions defined in __init__.py. The fix functions in the sibling modules (tensor_operations, attention_fixes, diffusion_fixes, embedding_fixes, transformer_fixes) are never called anywhere in the package's production path — a full-repo grep finds references only in tests (tests/stageD/...) and within the modules themselves. They are effectively dead-in-production patch code.
- **Recommendation:** Either wire the intended fixes into apply_tensor_fixes or delete the unused modules; the current split implies fixes are active when they are not.

### [MEDIUM] s2c6l0-tenops-matmul-guard — bug
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:47-108`
- **Evidence:** fix_matrix_multiplication() guards re-patching by checking hasattr(torch.nn.functional.linear,'_patch_applied_safe_linear') (and matmul/bmm flags), but none of safe_linear/safe_matmul/safe_bmm ever set those attributes. So the guard never trips; repeated calls re-wrap the already-patched functions (the inline comment even notes 'linear is the one causing recursion here'). Additionally the retry paths call torch.matmul/torch.bmm/F.linear (the patched versions) rather than the captured originals, risking recursion.
- **Recommendation:** Set the _patch_applied_* attributes on the wrappers so the idempotency guard works, and have retries call the captured originals (_original_matmul, etc.).

### [MEDIUM] s2c6l2-016 — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/transformer_fixes.py:8-61,64-120`
- **Evidence:** fix_atom_transformer and fix_atom_attention_encoder are misleading no-ops: each computes original_forward and a patched_forward but the actual assignment is commented out and replaced by a print 'Faulty/disabled' (lines 59-61,117-120). The early-return guards reference flags (_patch_applied_forward_fix :17, _patch_applied_forward :74) that are never set. Only patched_init in fix_atom_attention_encoder (:130) is actually applied.
- **Recommendation:** Delete the dead patched_forward bodies and disabled guards, or re-enable/repair them; keep only the code that actually runs (patched_init).

### [MEDIUM] s2re3-006 — security
- **Location:** `rna_predict/predict.py:380-393`
- **Evidence:** load_partial_checkpoint() calls torch.load(checkpoint_path, map_location='cpu') (:383) with no weights_only=True. torch.load unpickles arbitrary objects, so loading a checkpoint from an untrusted/downloaded source (config_schema.py:271 defines a remote checkpoint_url; predict.py:440 is the README-recommended inference entry that loads partial checkpoints) executes arbitrary code embedded in the pickle. weights_only is unset, so this is unsafe on all torch versions prior to the 2.6 default flip. Not in the listed findings.

### [MEDIUM] s2c6l0-predict-torchload — security
- **Location:** `rna_predict/predict.py:383`
- **Evidence:** load_partial_checkpoint calls torch.load(checkpoint_path, map_location='cpu') with no weights_only=True. checkpoint_path is user/CLI supplied (cfg.checkpoint_path resolved in main). torch.load uses pickle and can execute arbitrary code from a crafted checkpoint, an RCE vector for the README-recommended inference entry point.
- **Recommendation:** Pass weights_only=True (or use safetensors) when loading external checkpoints.

### [MEDIUM] s2re2-003 — security
- **Location:** `rna_predict/predict.py:383; rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:279`
- **Evidence:** load_partial_checkpoint calls torch.load(checkpoint_path, map_location='cpu') (predict.py:383) and StageARFoldPredictor calls torch.load(checkpoint_path, map_location=self.device) (rfold_predictor.py:279) with no weights_only=True. torch.load uses pickle, so loading a checkpoint from an untrusted/downloaded source (the RFold checkpoint is fetched from a remote Dropbox URL, see s2re2-004) executes arbitrary code during unpickling. checkpoint_path is user/config supplied (predict.py CLI partial-checkpoint option). Insecure deserialization; no torch.load finding currently exists.

### [MEDIUM] s2c6l2-019 — design_defect
- **Location:** `rna_predict/predict.py:425-432`
- **Evidence:** batch_predict.write_pdb writes PDB ATOM records using the residue letter as both the atom name and the element symbol: `atom = row["resname"]` then writes `{atom:>4}` into the atom-name field and `{atom[0]:>2}` into the element field (:428-430). The resulting PDB has chemically meaningless atom names (e.g. an atom literally named 'A'/'C'/'G'/'U'), inconsistent with the README's stated PDB output deliverable.
- **Recommendation:** Emit a real atom name (e.g. "P" or "C1'") for the per-residue representative coordinate and a correct element column, or document that the PDB is a placeholder.

### [MEDIUM] s2c6l0-predict-seq-noval — bug
- **Location:** `rna_predict/predict.py:476-532`
- **Evidence:** When the input CSV uses the 'sequence' column (lines 478-480), seq_paths is never populated. The limit-mode block at line 519 truncates `sequences` but its per-sequence validation loop iterates `zip(sequences[:limit_n], seq_paths[:limit_n])` (line 523) which is empty because seq_paths is []. Thus A/C/G/U validation is silently skipped for 'sequence'-column inputs even in fast_dev_run/limit mode, and invalid characters propagate downstream.
- **Recommendation:** Populate seq_paths in the 'sequence' column branch (e.g. with None placeholders) or validate `sequences` directly independent of seq_paths.

### [MEDIUM] s2c6l2-023 — design_defect
- **Location:** `rna_predict/runners/full_pipeline.py:24`
- **Evidence:** The library module calls logging.basicConfig(level=logging.DEBUG, ...) at import time (:24). Importing runners.full_pipeline (re-exported by run_full_pipeline.py) forces global root-logger DEBUG configuration onto any process that imports it, an import side effect inappropriate for a library.
- **Recommendation:** Remove basicConfig from module import; configure logging only in __main__ entry points.

### [MEDIUM] s2c6l2-029 — intent_mismatch
- **Location:** `rna_predict/runners/pipeline_cli.py:12-16,rna_predict/runners/conf/default.yaml:1-30`
- **Evidence:** pipeline_cli.py declares @hydra.main(config_path="conf", config_name="default") (:12); relative to rna_predict/runners/ this resolves to runners/conf/default.yaml, the minimal demo config which contains only test_data/pipeline/model.stage*.enabled and lacks top-level `sequence`, `device`, and `model.stageC`. Yet pipeline_cli reads cfg.sequence (:16) and run_full_pipeline requires cfg.device (full_pipeline.py:349) and cfg.model.stageC (full_pipeline.py:391). The full-pipeline CLI is wired to the demo's stripped config rather than rna_predict/conf.
- **Recommendation:** Point pipeline_cli at the real package config (rna_predict/conf via an absolute/searchpath) or populate runners/conf/default.yaml with the fields the full pipeline requires.

### [MEDIUM] s2c6l0-hypot-leadzero-regex — bug
- **Location:** `rna_predict/scripts/hypot_test_gen.py:29-39`
- **Evidence:** fix_leading_zeros uses re.sub(r'(-?)0+(\d+)', repl, s) which is not anchored to number boundaries. It matches a zero-run anywhere inside a larger integer: e.g. '1007' matches the substring '007' at offset 1 and is rewritten to '17', corrupting numeric literals in the generated test text it is meant to clean. Any integer containing an internal '0' run followed by digits (1007, 2008, 10005, ...) is mangled.
- **Recommendation:** Anchor with word boundaries / lookbehind, e.g. r'(?<![\d.])(-?)0+(\d+)\b', so only genuine leading-zero integers are normalized.

### [MEDIUM] s2c6l2-037 — intent_mismatch
- **Location:** `rna_predict/training/rna_lightning_module.py:1008-1012`
- **Evidence:** configure_optimizers hardcodes Adam lr=1e-3 (:1012) and the docstring restates the hardcoded value, ignoring any learning rate present in the Hydra training config. Stage-1 describes training as Hydra-config-driven, so the optimizer LR being non-configurable is a config/intent mismatch.
- **Recommendation:** Read learning rate (and optimizer choice) from cfg.training with a documented default.

### [MEDIUM] s2re4-005 — security
- **Location:** `rna_predict/training/rna_lightning_module.py:138-163 and rna_predict/pipeline/stageA/run_stageA.py:68-93`
- **Evidence:** Two near-identical download helpers fetch remote files via urllib.request.urlopen(url, timeout=30) + shutil.copyfileobj with NO checksum/hash/signature verification of the downloaded artifact. The default source is a personal Dropbox link (config_schema.py:271 'https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1'); the downloaded zip is later unzipped and the checkpoint torch.load'd, so a MITM or repointed link delivers an arbitrary payload executed at load time. The ~25-line download/backoff/zip-validation block is duplicated verbatim across both modules (design defect: no shared utility).

### [MEDIUM] s2c6l1-lightning-unverified-download — security
- **Location:** `rna_predict/training/rna_lightning_module.py:140`
- **Evidence:** _download_file() does `urllib.request.urlopen(url, timeout=30)` then shutil.copyfileobj into dest_path. The url is taken from getattr(stageA_cfg,'checkpoint_url',None) (rna_lightning_module.py:199) with no scheme allowlist, no TLS pinning, and no post-download hash/integrity check. urllib honors arbitrary schemes (file://, ftp://) and follows redirects, so a config override or compromised config can cause SSRF / reading local files / fetching attacker content that is then unzipped and loaded as model weights.
- **Recommendation:** Restrict to https:// with an explicit host allowlist, verify a known SHA-256 of the downloaded artifact before use, and reject non-http(s) schemes. Treat checkpoint_url as untrusted input.

### [MEDIUM] s2re3-005 — security
- **Location:** `rna_predict/training/rna_lightning_module.py:140,173-174`
- **Evidence:** RNALightningModule._download_file uses urllib.request.urlopen(url) (:140) with no integrity check and _unzip_file calls zip_ref.extractall(extract_dir) (:173-174) with no path validation — the same Zip-Slip / unauthenticated-download pattern as run_stageA.py, duplicated verbatim here. Also a code-duplication design defect (the download+unzip helpers are copy-pasted across two modules). Not in the listed findings.

### [MEDIUM] s2re5-lightning-zipslip-dup — security
- **Location:** `rna_predict/training/rna_lightning_module.py:140,174`
- **Evidence:** rna_lightning_module contains a second copy of the same network-download-then-extract logic: urllib.request.urlopen(url, timeout=30) at :140 followed by zip_ref.extractall(extract_dir) at :174, again with no member-path validation (zip-slip) and no integrity check of the downloaded archive. rna_lightning_module.py is absent from the finding inventory.

### [MEDIUM] s2c6l0-lm-noise-schedule — intent_mismatch
- **Location:** `rna_predict/training/rna_lightning_module.py:250-277`
- **Evidence:** _sample_noise_level reads p_mean/p_std (the EDM/AF3 log-normal noise-schedule params, lines 259-260) but never uses them. It instead samples log_sigma uniformly between log(s_min) and log(s_max) (lines 272-275). The diffusion noise distribution therefore does not follow the configured (and AF3-intended) log-normal schedule; the p_mean/p_std config knobs are dead.
- **Recommendation:** Implement the intended schedule: sigma = sigma_data * exp(p_mean + p_std * randn(batch)), or document that uniform-in-log is intentional and remove the unused params.

### [MEDIUM] s2c6l0-lm-tensor-or — bug
- **Location:** `rna_predict/training/rna_lightning_module.py:612`
- **Evidence:** coords_pred_C = (output.get('coords_3d') or output.get('coords')).to(self.device_). When output['coords_3d'] is a multi-element tensor, the `or` triggers bool(tensor), which raises 'Boolean value of Tensor with more than one element is ambiguous'. This streamline-mode path breaks whenever a 'coords_3d' tensor is present in the output dict.
- **Recommendation:** Use explicit None checks: `c = output.get('coords_3d'); c = c if c is not None else output.get('coords')`.

### [MEDIUM] s2c6l0-train-accelerator — bug
- **Location:** `rna_predict/training/train.py:198-199`
- **Evidence:** L.Trainer(accelerator=cfg.device, ...) passes a torch device string (e.g. 'cuda', 'cuda:0', or even 'cpu') as the Lightning accelerator. Lightning accelerator expects 'cpu'|'gpu'|'mps'|'tpu'|'auto'; 'cuda'/'cuda:0' are not valid accelerator names and raise a MisconfigurationException, so GPU training cannot start with a typical device config.
- **Recommendation:** Map cfg.device to a valid accelerator ('gpu' for cuda) and pass the index via devices, or use accelerator='auto'.

### [MEDIUM] s2c6l0-checkpoint-nonstrict-raise — bug
- **Location:** `rna_predict/utils/checkpoint.py:61-91`
- **Evidence:** partial_load_state_dict is documented to 'skip mismatched keys'. But a shape-mismatch during own_state[name].copy_(param) is appended to error_msgs (lines 63-68), and error_msgs is raised unconditionally at lines 86-91 regardless of strict. So a single shape-mismatched key raises RuntimeError even in the default strict=False mode, defeating partial loading.
- **Recommendation:** In non-strict mode, log and skip keys whose shapes differ (check own_state[name].shape == param.shape before copy_), and only raise copy errors when strict=True.

### [MEDIUM] s2c6l0-backbone-multires — bug
- **Location:** `rna_predict/utils/rna_backbone_extraction.py:51-75,118-137`
- **Evidence:** extract_pdb_backbone_coords collects every ATOM whose name is in CANONICAL_BACKBONE_ORDER across ALL residues when residue_select is None (the mode main() uses, line 149). It then sorts solely by CANONICAL_BACKBONE_ORDER.index(atom), interleaving atoms from different residues, and 'missing' is computed from the deduplicated name set. compute_bond_lengths/compute_bond_angles then compute geometry across atoms belonging to different residues, yielding meaningless bond lengths/angles for any multi-residue file.
- **Recommendation:** Group atoms by (chain, residue number) and compute per-residue (and inter-residue P-O3') geometry, rather than globally sorting by atom-name order.

### [MEDIUM] s2c6l2-043 — design_defect
- **Location:** `rna_predict/utils/rna_backbone_extraction.py:70-75,107-112`
- **Evidence:** Both extractors sort ALL collected backbone atoms by CANONICAL_BACKBONE_ORDER.index(atom) (:70,:107) and dedup via a set (:71,:108). When residue_select is None and a file contains multiple residues, atoms from different residues with the same name collapse/scramble: the returned list is no longer per-residue ordered, the 'missing atoms' check (:72-74,:109-111) reports spurious results, and compute_bond_lengths/angles operate across residue boundaries.
- **Recommendation:** Group atoms by (chain, residue) before sorting/geometry, or require residue_select; document that multi-residue files are unsupported.

### [MEDIUM] s2c6l2-049 — design_defect
- **Location:** `rna_predict/utils/tensor_utils.py:1-24`
- **Evidence:** A module file rna_predict/utils/tensor_utils.py coexists with a package directory rna_predict/utils/tensor_utils/ (both present per Stage-1 inventory lines 459-460). In Python, the package (with __init__.py) shadows the same-named module, so utils/tensor_utils.py is unreachable: `import rna_predict.utils.tensor_utils` always resolves to the package. The shim's re-exports are dead, and the duplicate names invite confusion.
- **Recommendation:** Delete the shadowed tensor_utils.py module (the package __init__.py already re-exports the same symbols).

### [MEDIUM] s2c6l2-051 — design_defect
- **Location:** `rna_predict/utils/tensor_utils/residue_mapping.py:229-256,283-288`
- **Evidence:** _adjust_counts_to_match_total proportionally rescales per-residue atom counts when expected total != actual n_atoms and dumps the entire remainder onto the last residue (adjusted_counts[-1] += diff, :254). Triggered from derive_residue_atom_map Method 2 (:426) on a logged warning only (:283-288). This silently fabricates an atom->residue assignment that can be grossly wrong (e.g. all leftover atoms attributed to the final residue), corrupting residue-to-atom bridging downstream.
- **Recommendation:** On atom-count mismatch, require explicit atom_metadata or raise; do not invent a contiguous mapping by proportional rescaling plus last-residue padding.

### [MEDIUM] s2c6l2-050 — design_defect
- **Location:** `rna_predict/utils/tensor_utils/types.py:16-25`
- **Evidence:** STANDARD_RNA_ATOMS here defines per-residue atom names/counts (A=22,U=20,G=23,C=20) and is used by residue_mapping.derive_residue_atom_map for atom-count fallbacks. The Stage-1 inventory designates rna_predict/dataset/atom_lists.py as the 'single source of truth defining standard RNA atom ordering and per-residue max atom counts'. Two independent atom-definition sources risk silent drift in atom counts/ordering between bridging and dataset code.
- **Recommendation:** Derive STANDARD_RNA_ATOMS from dataset/atom_lists.py (single source) instead of redefining atom sets here.

### [MEDIUM] s2c6l1-analyze-curl-pipe-sh — security
- **Location:** `scripts/analysis/analyze_code.sh:109`
- **Evidence:** The script auto-installs tooling by piping a remote script directly into a shell: `curl -sSf https://downloads.codescene.io/.../install-codescene-cli.sh | sh` (analyze_code.sh:109), and similarly runs `pip install uv` / `uv run pip install ruff` (analyze_code.sh:87,144) with no checksum or signature verification. A compromised or MITM'd download executes arbitrary code with the running user's privileges.
- **Recommendation:** Download installers to a file, verify a pinned checksum/signature, then execute; or install tools from a vetted, pinned package index. Avoid curl|sh in developer/CI scripts.

### [MEDIUM] s2c7l2-002 — design_defect
- **Location:** `scripts/automation/batch_test_generator.py:28-43`
- **Evidence:** `run_test_generation(*args, **kwargs)` is a no-op stub returning None (:28-29), so in process_folder `if not result:` (:23) is always true and the script only ever prints 'Failed to generate tests for ...'. Worse, the file has NO `if __name__ == '__main__': main()` guard, so even running `python batch_test_generator.py <folder>` (the usage it prints at :35) does nothing. The module is a non-functional stub (docstring at :3 admits 'Stub implementation').
- **Recommendation:** Either delete this stub duplicate in favor of the working scripts/test_utils/batch_test_generator.py, or implement run_test_generation and add a `__main__` guard so the printed usage actually works.

### [MEDIUM] s2c7l0-001 — bug
- **Location:** `scripts/automation/commit_individual_files.sh:90-94`
- **Evidence:** STEP 2 gathers files with `find ... -exec stat -f "%z %N" {} + 2>/dev/null | sort -n`. `stat -f` is the BSD/macOS format; on Linux `stat` requires `-c "%s %n"` (the script's own comment at lines 88-89 documents both forms but only the macOS form is used). On Linux every stat call errors, the errors are swallowed by `2>/dev/null`, so file_list is empty; the script then logs 'No files found in $folder (or all already committed)' (line 97) and silently commits nothing.
- **Recommendation:** Detect the OS (or use a portable invocation such as `find ... -printf '%s %p\n'` on GNU find / `stat -c` on Linux vs `stat -f` on macOS) so the size-sorted file list is populated on both platforms; do not suppress stat's stderr unconditionally.

### [MEDIUM] s2c7l2-019 — bug
- **Location:** `scripts/automation/commit_individual_files.sh:92`
- **Evidence:** File listing uses `stat -f '%z %N'` (:92) — the macOS/BSD syntax. The script's own comment (:88-89) documents that Linux needs `stat -c '%s %n'`, yet only the macOS form is coded. On Linux (the stated platform) `stat -f` means 'filesystem status' and fails; the `2>/dev/null` (:93) swallows the error, leaving file_list empty so the script reports 'No files found' (:97) and commits nothing.
- **Recommendation:** Branch on OS (or use `find -printf '%s %p\n'`) so the size+name listing works on both macOS and Linux as the comment intends.

### [MEDIUM] s2c7l2-015 — bug
- **Location:** `scripts/automation/create_github_issues.py:62`
- **Evidence:** The remote-URL regexes `^git@github\.com:(.+)/(.+)(\.git)?$` (:62) and `^https://github\.com/(.+)/(.+)(\.git)?$` (:67) make `(\.git)?` optional after a greedy `(.+)`, so for 'git@github.com:OWNER/REPO.git' group(2) greedily captures 'REPO.git' and the optional group matches empty. The auto-detected repo therefore keeps the '.git' suffix, producing an API URL repos/OWNER/REPO.git/issues that 404s. (The sibling shell script github_automation.sh:56 explicitly strips '.git' with sed, showing the intended behavior.)
- **Recommendation:** Strip the suffix or make the group mandatory/non-greedy, e.g. capture `(.+?)(?:\.git)?$` for the repo and rstrip('.git').

### [MEDIUM] s2c7l0-002 — bug
- **Location:** `scripts/automation/create_github_issues.py:62-69`
- **Evidence:** get_repo_from_git uses `re.match(r"^git@github\.com:(.+)/(.+)(\.git)?$", ...)` (and the analogous HTTPS pattern). The second `(.+)` is greedy and the `(\.git)?` group is optional, so for `git@github.com:OWNER/REPO.git` group(2) captures `REPO.git` (the `.git` is never stripped). The returned repo name therefore includes the `.git` suffix and the GitHub API URL `repos/{owner}/{repo}` is wrong. (The sibling shell script github_automation.sh:51-57 explicitly strips `.git`, confirming intent.)
- **Recommendation:** Anchor the `.git` non-optionally for SSH or strip it explicitly, e.g. `re.match(r'^git@github\.com:(.+?)/(.+?)(?:\.git)?$', url)` with non-greedy groups, or `repo = group(2).removesuffix('.git')`.

### [MEDIUM] s2c7l2-017 — bug
- **Location:** `scripts/automation/github_automation/analyze_commit_log.py:112`
- **Evidence:** COMMIT_RE (:11) parses logs in the `%h - %an, %ar : %s` shape produced by github_automation.sh (`--pretty=format:'%h - %an, %ar : %s'` at github_automation.sh:119,130), where the time field is %ar — a RELATIVE string like '2 days ago'. But line 112 parses it with `pd.to_datetime(df['time_ago'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce')` (absolute ISO+tz, comment :111 'exact ISO format'). All relative strings coerce to NaT, dropna (:113) empties the frame, and the time-distribution plot (:116-124) is built on no data.
- **Recommendation:** Either change the log generator to emit `%ad`/`%aI` (absolute ISO date), or parse the relative %ar format instead of an ISO format string.

### [MEDIUM] s2c7l0-010 — intent_mismatch
- **Location:** `scripts/automation/github_automation/analyze_commit_log.py:112`
- **Evidence:** Time parsing uses `pd.to_datetime(df['time_ago'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce')`, but the companion log generators emit git's relative `%ar` format ('2 days ago') — see scripts/automation/github_automation.sh:119,127,130 (`--pretty=format:"%h - %an, %ar : %s"`), which is also the shape the COMMIT_RE/time group at line 11 captures. Relative strings never match the ISO format, so every value coerces to NaT and `df.dropna(subset=['time_ago'])` (line 113) empties the frame, leaving the time-distribution plot (lines 116-124) blank/meaningless.
- **Recommendation:** Generate the source log with an absolute timestamp (`%ai`/`%aI`) to match the parser, or parse relative times appropriately; assert non-empty rows after dropna and warn instead of silently producing an empty plot.

### [MEDIUM] s2c7l1-001 — security
- **Location:** `scripts/inspect_checkpoint.py:10`
- **Evidence:** main() calls torch.load(ckpt_path, map_location='cpu') on an arbitrary path supplied as argv[1] with no weights_only=True. torch.load uses Python's pickle, which executes arbitrary code embedded in the file during unpickling. This script is explicitly a tool to inspect *any* .pt/.ckpt file (including third-party/downloaded checkpoints), so a maliciously crafted checkpoint achieves arbitrary code execution when inspected.
- **Recommendation:** Pass weights_only=True to torch.load (PyTorch>=2.0) for inspection, or load with a safe unpickler / pickletools-based metadata reader. Document that only trusted checkpoints should be loaded.

### [MEDIUM] s2c7l1-002 — security
- **Location:** `scripts/inspect_pt_file.py:10`
- **Evidence:** torch.load(pt_path, map_location='cpu') is called on argv[1] (any user-supplied .pt path) without weights_only=True. Pickle deserialization of an untrusted .pt file allows arbitrary code execution. The script's stated purpose is inspecting arbitrary .pt files, maximizing the chance of pointing it at an untrusted artifact.
- **Recommendation:** Use torch.load(..., weights_only=True) or a restricted unpickler when the goal is only to enumerate keys/preview content; never unpickle untrusted files with default settings.

### [MEDIUM] s2re4-002 — intent_mismatch
- **Location:** `scripts/partial_checkpoint_full_pipeline_script.py:60-67`
- **Evidence:** Lines 56-62 carefully detect a relative config path ('rna_predict/conf' or 'conf') under PROJECT_ROOT and store it in config_path_selected, exiting with [UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND] on failure. Line 67 then ignores config_path_selected entirely and hardcodes hydra.initialize(config_path="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"). The portability logic is dead code; the script only runs on the original author's machine ('/Users/tomriddle1') and crashes everywhere else despite the [HYDRA-PROJECT-RULE] comment.

### [MEDIUM] s2re2-006 — bug
- **Location:** `scripts/partial_checkpoint_full_pipeline_script.py:67; rna_predict/pipeline/stageD/memory_optimization/test_memory.py:90`
- **Evidence:** Two shipped (non-tests/ tree) scripts hardcode hydra config_path to the developer-only absolute path "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf": scripts/partial_checkpoint_full_pipeline_script.py:67 (hydra.initialize) and rna_predict/pipeline/stageD/memory_optimization/test_memory.py:90 (get_config). The latter lives inside the installed package (rna_predict/pipeline/...), not under tests/, so it ships in the wheel yet is unrunnable on any other machine. Distinct location set from train.py (s2re2-001) and compute_ground_truth_angles.py:53.

### [MEDIUM] s2c7l0-009 — bug
- **Location:** `scripts/run_mutation_tests.sh:44`
- **Evidence:** `if ! $CMD 2>&1 | tee "$LOG_FILE"; then` evaluates the exit status of the last pipeline element (`tee`), not of `mutatest`. The script has no `set -o pipefail`, so a failing mutatest run (nonzero exit) is masked whenever tee succeeds (almost always), and the entire error-classification branch (lines 45-66) is effectively unreachable, while the final 'completed successfully' path (lines 70-74) runs on failures.
- **Recommendation:** Add `set -o pipefail` at the top, or capture mutatest's status via PIPESTATUS, e.g. `$CMD 2>&1 | tee "$LOG_FILE"; status=${PIPESTATUS[0]}; if [ "$status" -ne 0 ]; then ...`.

### [MEDIUM] s2c7l1-004 — security
- **Location:** `scripts/test_utils/hypot_test_gen.py:327-336`
- **Evidence:** run_hypothesis_write builds full_cmd = f"hypothesis write {command}" and executes subprocess.run(full_cmd, shell=True, ...). 'command' embeds method_path/module_path derived from the scanned file's filesystem path/stem (construct_module_path/generate_*_variants). A Python file whose name or directory contains shell metacharacters (e.g. ';', '$(...)', backticks) yields command injection executed via the shell when generating tests for an attacker-supplied folder.
- **Recommendation:** Invoke subprocess with an argument list (shell=False) and pass the hypothesis arguments as discrete tokens; validate/whitelist module path characters before interpolation.

### [MEDIUM] s2c7l1-005 — security
- **Location:** `scripts/test_utils/hypot_test_gen.py:785-794`
- **Evidence:** combine_and_cleanup_tests constructs ruff commands as f-strings interpolating combined_filepath (derived from the input file stem) and runs subprocess.run(cmd, shell=True, ...). A controlled file stem containing shell metacharacters or spaces breaks out of the intended command, enabling command injection / argument splitting when processing an untrusted target folder.
- **Recommendation:** Replace shell=True f-string commands with list-form subprocess calls (e.g. ['ruff','check',str(combined_filepath)]) so the path is passed as a single, non-interpreted argument.

### [MEDIUM] s2c7l2-006 — intent_mismatch
- **Location:** `scripts/test_utils/mark_slow_tests.py:6`
- **Evidence:** SLOW_TEST_THRESHOLD = 1.0 (:6) and the implied intent (mark tests slower than 1s) are never realized: the code contains no timing/measurement logic. SlowTestMarker.visit_FunctionDef marks EVERY function named test_* (is_test_function at :25-27) regardless of runtime, so the threshold constant is dead/misleading and the marker would (if it ran) flag all tests, not slow ones.
- **Recommendation:** Either feed the marker from actual pytest --durations data keyed on SLOW_TEST_THRESHOLD, or drop the threshold constant and rename the tool to reflect that it marks all tests.

### [MEDIUM] s2c7l2-008 — intent_mismatch
- **Location:** `scripts/test_utils/run_failing_tests.sh:1-6,328`
- **Evidence:** The filename and the Stage-1 inventory summary ('Shell script to run and report on failing tests') promise running failing tests, but the script runs the ENTIRE suite: `test_files=$(find tests -type f -name 'test_*.py')` (:328) then `pytest $test_files ...` (:404). Its actual purpose is progressive-coverage gating across a Kaggle timeline (header :3-6). Nothing selects or re-runs only failing tests.
- **Recommendation:** Rename to reflect its real role (e.g. run_progressive_coverage.sh) or add `--lf/--last-failed` selection if running only failures is the intent.

### [MEDIUM] s2c7l0-008 — bug
- **Location:** `scripts/test_utils/run_failing_tests.sh:317-319`
- **Evidence:** get_coverage_goal() prints several human-readable status lines (e.g. 'Days since last run: 1') and finally `echo $COVERAGE_GOAL` such as '85.00'. The caller captures the ENTIRE multi-line output into COVERAGE_GOAL, then extracts with `grep -o '[0-9]\+' | tail -1`. For a fractional goal like 85.00 the last digit-run is the decimal fraction '00', so COVERAGE_GOAL becomes '00' and the pytest invocation runs `--cov-fail-under=00`, effectively setting a 0% gate and disabling the coverage threshold the script exists to enforce.
- **Recommendation:** Return only the numeric goal from get_coverage_goal (write status to stderr, value to stdout) and parse the integer part explicitly, e.g. `printf '%.0f' "$RAW_GOAL"`, instead of `grep -o '[0-9]\+' | tail -1`.

### [MEDIUM] s2re1-003 — doc_drift
- **Location:** `setup.py:1-32 vs pyproject.toml:5-38`
- **Evidence:** A second, conflicting build manifest exists. setup.py declares name="rna_predict", version="1.0.0", python_requires=">=3.8" and a totally different dependency set (numpy, scipy, pandas, tqdm, PySimpleGUI). pyproject.toml declares name="rna-predict", version="2.0.8", requires-python=">=3.10" with build-backend=setuptools.build_meta. Because PEP 621 [project] metadata in pyproject takes precedence under setuptools.build_meta, setup.py is dead/misleading config: its version, python floor and deps (e.g. PySimpleGUI) are never applied, yet it advertises a different package identity to any reader/tool that parses it.

### [MEDIUM] s2c7l2-011 — doc_drift
- **Location:** `setup.py:6`
- **Evidence:** setup.py declares version='1.0.0' (:6) while pyproject.toml:7 declares version='2.0.8' and rna_predict/VERSION contains 2.0.8. Two coexisting build configs with contradictory version metadata; whichever build path is used yields an inconsistent package version.
- **Recommendation:** Pick one source of truth (pyproject.toml is canonical per the setuptools backend) and either delete setup.py or sync its version/metadata to 2.0.8.

### [MEDIUM] s2c7l0-011 — design_defect
- **Location:** `setup.py:6-21`
- **Evidence:** setup.py declares `version="1.0.0"` and an install_requires that lists GUI/screen-finder deps (opencv-python, mss, PySimpleGUI, pyautogui) as CORE requirements while omitting the pipeline's real runtime deps (no hydra-core, pytorch-lightning, omegaconf). The authoritative build config is pyproject.toml (version 2.0.8, verified). Two divergent build-metadata sources for the same package cause version/dependency drift and ambiguous installs depending on which path the toolchain uses.
- **Recommendation:** Consolidate on pyproject.toml (delete or reduce setup.py to a shim), or sync setup.py's version and dependency list with pyproject.toml; move GUI-only deps to an optional extra.

### [LOW] s2c0l2-0022 — doc_drift
- **Location:** `.augement_code_rules`
- **Evidence:** The rules filename is misspelled `.augement_code_rules` (should be 'augment'), and the file's own internal references call it `.augment coderules` / `AUGMENT CODE_RULES` (lines 337-338, 357), none of which match the actual filename. Augment Code looks for a specific rules filename, so the misspelled name likely means the tool never loads it.
- **Recommendation:** Rename to the filename Augment Code expects and make the in-file self-references match.

### [LOW] s2c0l2-0019 — other
- **Location:** `.coverage_config.json.bak:1`
- **Evidence:** .coverage_config.json.bak is byte-identical to .coverage_config.json (both 56 lines, same content) — a committed editor backup file. Both are also listed in .gitignore (.gitignore:195,198,199) yet are tracked in the repo, an internal contradiction.
- **Recommendation:** Delete the .bak backup from version control and either untrack or stop gitignoring the canonical .coverage_config.json.

### [LOW] s2c0l2-0017 — doc_drift
- **Location:** `.coverage_config.json:31-38`
- **Evidence:** The phased coverage schedule ends at final_submission end_date 2025-05-29 and last_updated is 2025-05-10; all phase windows are in the past relative to the current date (2026-06-17). The phase-driven targets are stale and no longer actionable, and current_coverage 89.99 is a hard-coded snapshot with no link to actual measured coverage.
- **Recommendation:** Update or retire the phase schedule; derive current_coverage from a live coverage run rather than a static value.

### [LOW] s2c0l2-0018 — doc_drift
- **Location:** `.coverage_config.json:51-53`
- **Evidence:** module_categories.utility_modules lists `rna_predict.scripts` for coverage tracking, but rna_predict/scripts/ contains only __init__.py and hypot_test_gen.py (verified by ls), and .coveragerc explicitly omits `*/scripts/*` from coverage (.coveragerc:8). Tracking a scripts package that coverage is told to omit is contradictory.
- **Recommendation:** Remove rna_predict.scripts from the tracked categories or stop omitting scripts in .coveragerc — pick one consistent policy.

### [LOW] s2c0l1-014 — design_defect
- **Location:** `.coveragerc:11; .coverage_config.json:3`
- **Evidence:** Coverage enforcement is contradictory: .coveragerc sets `fail_under = 0` (no gate), while .coverage_config.json declares base_coverage 80 / current 89.99 / phase targets up to 95 (lines 3-5,31-38). The effective CI gate (`make test` -> pytest --cov-config .coveragerc, Makefile:48) enforces nothing, so the documented coverage policy is unenforced.
- **Recommendation:** Either wire .coverage_config.json thresholds into the actual gate or set .coveragerc fail_under to the intended floor; reconcile the two sources of truth.

### [LOW] s2c0l2-0039 — doc_drift
- **Location:** `.github/FUNDING.yml:12`
- **Evidence:** The custom funding list includes the placeholder URL `'https://www.example.com/sponsor2'` alongside the real GitHub sponsors link. example.com is a non-functional placeholder that would render as a broken sponsor link on the repository.
- **Recommendation:** Remove the example.com placeholder entry.

### [LOW] s2c0l2-0038 — design_defect
- **Location:** `.github/dependabot.yml:3`
- **Evidence:** Dependabot is configured only for the `github-actions` ecosystem. The project has substantial Python dependencies (requirements*.txt, pyproject.toml) and a Node dependency set (package.json), none of which Dependabot is configured to monitor, so security/version updates for the actual application dependencies are never proposed.
- **Recommendation:** Add `pip` (and optionally `npm`) update entries to dependabot.yml to cover the real dependency surfaces.

### [LOW] s2c0l0-008 — bug
- **Location:** `.github/init.sh:62`
- **Evidence:** Line reads `echo "Applying ${template} template to this project"}` — a stray closing brace `}` is appended outside the quoted string. Bash prints it literally (output ends with `...project}`), indicating a copy/paste corruption; the following line then unconditionally execs `./.github/templates/${template}/apply.sh`.
- **Recommendation:** Remove the trailing `}` on line 62.

### [LOW] s2c0l1-012 — security
- **Location:** `.github/rename_project.sh:24-31`
- **Evidence:** The loop `for filename in $(git ls-files)` is unquoted (word-splits on whitespace) and runs `sed -i "s/$original_author/$author/g" $filename` with both the replacement values ($author/$name/$urlname/$description, from -a/-n/-u/-d args) and $filename unquoted and unescaped. Filenames with spaces break, and replacement values containing `/` or other sed metacharacters corrupt the substitution or could alter unintended files. In CI these values come from the repo owner/name (rename_project.yml:36), limiting external attacker control.
- **Recommendation:** Quote $filename, iterate safely (e.g. git ls-files -z | while IFS= read -r -d ''), and escape sed metacharacters in substitution values.

### [LOW] s2c0l1-009 — security
- **Location:** `.github/workflows/main.yml:104,38-47; .github/workflows/release.yml:40; .github/workflows/rename_project.yml:38`
- **Evidence:** Third-party GitHub Actions are pinned only to mutable major-version tags rather than immutable commit SHAs: codecov/codecov-action@v5 (main.yml:104), actions/upload-artifact@v4 (main.yml:44), softprops/action-gh-release@v2 (release.yml:40), stefanzweifel/git-auto-commit-action@v5 (rename_project.yml:38). A moved/compromised tag could inject code into CI, which here holds write tokens and (release.yml) the PyPI token.
- **Recommendation:** Pin third-party actions to full commit SHAs and use Dependabot (already configured for github-actions in .github/dependabot.yml) to bump them.

### [LOW] s2c0l0-006 — bug
- **Location:** `.github/workflows/main.yml:40`
- **Evidence:** `pip-audit -r requirements.txt --severity high --exit-code 1 ...` passes a `--severity` flag. pip-audit's CLI does not expose a `--severity` option (it has no built-in severity filtering), which would make the step error out on argument parsing. UNVERIFIED against the exact pip-audit version installed in CI (no version pin), so flagged as a likely-invalid flag rather than confirmed.
- **Recommendation:** Confirm the installed pip-audit version's supported flags; remove `--severity high` (filter severities downstream from the JSON output) or pin a pip-audit version that supports the intended behavior.

### [LOW] s2c0l1-007 — security
- **Location:** `.github/workflows/main.yml:50-70`
- **Evidence:** The linter job auto-applies `ruff check --fix --unsafe-fixes` (:53,:64) then configures a bot identity and runs `git add -A`/`git commit`/`git push` (:57-59,:67-70) inside CI, twice, under `continue-on-error: true`. This performs automated writes/pushes to the repository using GITHUB_TOKEN as a side effect of CI, masks failures, and applies *unsafe* (potentially behavior-changing) auto-fixes without human review before pushing.
- **Recommendation:** Move auto-fix to a gated, reviewable flow (open a PR rather than direct push), drop `--unsafe-fixes` from CI auto-commit, and avoid `continue-on-error` hiding push/commit failures.

### [LOW] s2c0l2-0040 — design_defect
- **Location:** `.github/workflows/mkdocs.yml:23`
- **Evidence:** The docs-deploy workflow pins Python 3.10 and installs only `mkdocs mkdocs-material`, while the main CI uses Python 3.11 (.github/workflows/main.yml:21) and mkdocs.yml relies on pymdownx extensions / arithmatex (mkdocs.yml:16-28). pymdown-extensions is declared in pyproject dev (pyproject.toml:53) but is not explicitly installed in this workflow (it ships transitively with mkdocs-material, so the build is fragile to that transitive dependency and the Python version is inconsistent with the rest of CI).
- **Recommendation:** Explicitly install pymdown-extensions and align the Python version with the main CI matrix to make doc builds reproducible.

### [LOW] s2c0l0-022 — design_defect
- **Location:** `.github/workflows/release.yml:38`
- **Evidence:** Release builds use `python setup.py sdist bdist_wheel` (lines 38 and 63). setup.py does exist (verified `ls setup.py`), so this runs, but `setup.py` invocations are deprecated by setuptools/PyPA in favor of `python -m build` given the project already declares a PEP517 build-backend (pyproject.toml:1-3). The direct setup.py path risks breakage on newer setuptools.
- **Recommendation:** Switch to `python -m build` (add `build` to the install step) for a PEP 517-compliant release flow consistent with the pyproject build-backend.

### [LOW] s2c0l1-015 — bug
- **Location:** `.gitignore:195-199,220-221,226-227`
- **Evidence:** Files that are committed and tracked are also listed in .gitignore: `.coverage_config.json` (:195,:198), `.coverage_config.json.bak` (:199), `package.json` (:221), and even `.gitignore` itself (:226-227). Because they are already tracked the ignore has no effect now, but it is misleading and risks accidental loss/non-tracking of regenerated copies and confuses contributors about which files are canonical.
- **Recommendation:** Remove ignore entries for files intended to be tracked (package.json, .coverage_config.json*, .gitignore), or untrack them deliberately if they are meant to be local-only.

### [LOW] s2c0l0-017 — design_defect
- **Location:** `.gitignore:220-227`
- **Evidence:** .gitignore lists already-tracked files: `package-lock.json` (220), `package.json` (221) — both are committed config files (package.json is an assigned tracked file) — and `.gitignore` itself appears twice (226-227). gitignore has no effect on tracked files, so these entries are inert and the self-ignore of .gitignore is nonsensical, signaling accidental edits. .coverage_config.json is likewise ignored (195,198) yet committed.
- **Recommendation:** Remove ignore entries for files that are intentionally tracked, or `git rm --cached` them if they should be untracked; delete the duplicate/self `.gitignore` lines.

### [LOW] s2c0l2-0036 — design_defect
- **Location:** `.gitignore:226-227`
- **Evidence:** .gitignore lists `.gitignore` itself (twice, lines 226-227). A file cannot meaningfully ignore itself once tracked; this is dead/nonsensical configuration. The file also contains many duplicated entries (e.g. `.DS_Store` repeated ~7 times across lines 141-152, `rna_predict/.DS_Store` twice, `.coverage_config.json` twice).
- **Recommendation:** Remove the self-referential .gitignore entries and de-duplicate the file.

### [LOW] s2c0l2-0044 — design_defect
- **Location:** `.gitmodules:1-3`
- **Evidence:** The RooFlow submodule is declared (path RooFlow, url https://github.com/ImmortalDemonGod/RooFlow.git) but is uninitialized — the working tree RooFlow/ is an empty directory (verified by ls; contains no files). A fresh clone without `--recurse-submodules` leaves RooFlow empty, and nothing in the build/runtime documents that this external dependency is required, so its purpose and necessity are unclear.
- **Recommendation:** Either initialize/commit the submodule pointer with documentation of why it is needed, or remove the RooFlow submodule if it is unused tooling.

### [LOW] s2c0l2-0024 — doc_drift
- **Location:** `.windsurfrules:2`
- **Evidence:** The debugging rule points to `docs/comprehensive_debugging_guide.md`, but the file actually lives at docs/guides/best_practices/debugging/comprehensive_debugging_guide.md (per the inventory at audit/01-understanding.md and mkdocs.yml:46). The referenced top-level path does not exist.
- **Recommendation:** Update the reference to the real docs path.

### [LOW] s2c0l2-0026 — doc_drift
- **Location:** `.windsurfrules:38`
- **Evidence:** Version-control rule states 'Do not commit changes to version control.', which directly contradicts the CI workflow that auto-commits and pushes ruff fixes on every push/PR (.github/workflows/main.yml:56-59,68-70). The guidance and the automation disagree.
- **Recommendation:** Reconcile the policy: either stop CI auto-commits or relax the rule for the CI bot.

### [LOW] s2re2-005 — design_defect
- **Location:** `MANIFEST.in:4-5`
- **Evidence:** MANIFEST.in does `graft tests` and `graft rna_predict`, which recursively include the entire test suite and every binary blob under the package tree in the source distribution. rna_predict/ contains large vendored binaries — rna_predict/dataset/preprocessing/dssr-basic-linuxMacWindows-v2.5.3.zip plus rna_predict/dataset/preprocessing/dssr/*.zip (linux/macOS/windows DSSR archives) and example .cif/.pt/.pdb fixtures — so the built sdist bundles multi-MB third-party DSSR archives (whose redistribution licensing is also questionable) and the full test tree. No MANIFEST.in finding exists in the current set.

### [LOW] s2re3-007 — design_defect
- **Location:** `MANIFEST.in:5`
- **Evidence:** MANIFEST.in contains 'graft tests' (:5), which bundles the entire tests/ tree into the built sdist/wheel. That ships ~223 test files plus large fixtures and the developer-only absolute-path tests (e.g. tests carrying '/Users/tomriddle1/...') into the published package, bloating the distribution and leaking dev-environment paths to consumers. The package distribution config has no existing finding.

### [LOW] s2c0l0-019 — doc_drift
- **Location:** `Makefile:107`
- **Evidence:** The `switch-to-poetry` target runs `poetry init --no-interaction --name=a_flask_test --author=ImmortalDemonGod`, hardcoding the unrelated template name 'a_flask_test' as the project name. If executed it would mislabel the package.
- **Recommendation:** Use the actual project name (rna-predict) in the poetry init invocation, or remove the obsolete switch-to-poetry target.

### [LOW] s2c0l2-0010 — intent_mismatch
- **Location:** `Makefile:107-110`
- **Evidence:** The switch-to-poetry target runs `poetry init --no-interaction --name=a_flask_test --author=ImmortalDemonGod` (line 107) — a leftover flask-template placeholder name — and appends a poetry script `rna_predict = 'rna_predict.__main__:main'` (line 110) targeting the same non-existent __main__ module (see s2c0l2-0003).
- **Recommendation:** If poetry support is wanted, set name to rna-predict and a valid entry point; otherwise remove the switch-to-poetry target.

### [LOW] s2c0l2-0012 — design_defect
- **Location:** `Makefile:2-3`
- **Evidence:** `USING_POETRY=$(shell grep "tool.poetry" pyproject.toml && echo "yes")` runs unconditional grep whose stdout (the matched line, if any) leaks into every `make` invocation; combined with `.ONESHELL`, this prints grep output as noise. pyproject.toml has no [tool.poetry] section, so USING_POETRY is always empty — the poetry branches in install/show/virtualenv are dead.
- **Recommendation:** Suppress grep output with `grep -q ... && echo yes` and remove the unreachable poetry branches.

### [LOW] s2c0l2-0011 — doc_drift
- **Location:** `Makefile:28-30`
- **Evidence:** The `fmt` target's help text is `## Format code using black & isort.` but its recipe only runs `$(ENV_PREFIX)isort rna_predict/` — black is never invoked. Separately the project uses ruff for formatting/import-sorting elsewhere (Makefile:43-44, .github/workflows/main.yml:73-79), so black/isort here are inconsistent with the actual toolchain.
- **Recommendation:** Update the help text to match the recipe, or standardize formatting on ruff and remove the redundant black/isort target.

### [LOW] s2c0l1-010 — security
- **Location:** `mkdocs.yml:30-31`
- **Evidence:** `extra_javascript: - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js` loads remote JavaScript into the published docs site from a third-party CDN with no Subresource Integrity hash and only a major-version range. A compromised/poisoned CDN asset would execute arbitrary JS for every docs visitor.
- **Recommendation:** Pin to an exact version and add an SRI integrity hash, or vendor MathJax locally under docs/.

### [LOW] s2c0l0-020 — doc_drift
- **Location:** `mkdocs.yml:81`
- **Evidence:** Nav entry 'MP-NeRF Integration: pipeline/stageC/Unified, Comprehensive Plan for Integrating MP-NeRF into Stage C.md' references a filename containing a literal comma and spaces. The Stage-1 inventory flags this exact file as UNRESOLVED (audit/01-understanding.md:48), and the on-disk name uses a non-breaking space variant; a mismatch will trigger mkdocs 'doc not found in nav'/missing-file warnings during `mkdocs build` (Makefile:99) and gh-deploy (.github/workflows/mkdocs.yml).
- **Recommendation:** Rename the doc to an ASCII-safe filename without commas/special spaces and update the nav entry to match exactly.

### [LOW] s2c0l0-013 — doc_drift
- **Location:** `mutatest.ini:3`
- **Evidence:** Comment states 'This configuration requires coverage==5.5 and pytest-cov==2.12.1', but requirements-test.txt:2,9 pin `coverage>=7.6.12` and `pytest-cov>=6.0.0`, and pyproject.toml:44,51 do the same. The stated mutatest prerequisite is years out of date relative to the actual pinned versions.
- **Recommendation:** Update or remove the stale version comment in mutatest.ini to reflect the coverage/pytest-cov versions actually used.

### [LOW] s2c0l2-0033 — design_defect
- **Location:** `pyproject.toml:41-76`
- **Evidence:** Two distinct dev dependency mechanisms coexist with different contents: [project.optional-dependencies].dev (lines 42-54: black, coverage, flake8, gitchangelog, isort, mkdocs, mypy, pytest, pytest-cov, pytest-xdist, pymdown-extensions) and [dependency-groups].dev (lines 63-76: cosmic-ray, mkdocs, mutatest, pydeps, pytest-asyncio, pytest-cov, pytest-faulthandler, pytest-memprof, pytest-timeout, snoop, types-requests). The two 'dev' sets barely overlap, so which tools you get depends on installer (pip extras vs uv groups).
- **Recommendation:** Merge into a single dev dependency declaration to avoid installer-dependent dev environments.

### [LOW] s2c0l0-015 — design_defect
- **Location:** `pyproject.toml:42-76`
- **Evidence:** Two parallel and inconsistent dev dependency declarations exist: `[project.optional-dependencies].dev` (lines 42-54: black, coverage, flake8, gitchangelog, isort, mkdocs, mypy, pytest, pytest-cov, pytest-xdist, pymdown-extensions) and `[dependency-groups].dev` (lines 63-76: cosmic-ray, mkdocs, mutatest, pydeps, pytest-asyncio, pytest-cov, pytest-faulthandler, pytest-memprof, pytest-timeout, snoop, types-requests). The two sets barely overlap, so `pip install .[dev]` (used by Makefile:26 and CI) and the uv dependency-group give different environments.
- **Recommendation:** Consolidate dev dependencies into a single canonical list to avoid environment skew between pip and uv installs.

### [LOW] s2c0l0-016 — doc_drift
- **Location:** `pyproject.toml:8`
- **Evidence:** `description = "Add your description here"` is the unedited template placeholder. This string is published as the package summary on any wheel/sdist build (release.yml).
- **Recommendation:** Replace with a real one-line project description.

### [LOW] s2c0l2-0015 — design_defect
- **Location:** `pytest.ini:15`
- **Evidence:** addopts includes `-p no:warnings` (line 15) which disables the pytest warnings plugin entirely, yet the file also defines a `filterwarnings` block (lines 27-31) intended to ignore specific warning categories. With the warnings plugin disabled, the filterwarnings rules are inert/contradictory.
- **Recommendation:** Drop `-p no:warnings` and rely on filterwarnings, or remove the filterwarnings block since the plugin is disabled.

### [LOW] s2c0l2-0029 — design_defect
- **Location:** `requirements-test.txt:1`
- **Evidence:** requirements-test.txt overlaps and diverges from pyproject dev declarations: it pins runtime dep hydra-core==1.3.2 in a test-only file (line 8) and lists black/gitchangelog/memory-profiler/pytest-asyncio/pytest-timeout that are spread differently across pyproject [project.optional-dependencies].dev and [dependency-groups].dev. There is no single authoritative test-dependency set.
- **Recommendation:** Consolidate test deps into one location (pyproject dev extra) and drop the redundant requirements-test.txt or generate it from pyproject.

### [LOW] s2c0l1-011 — security
- **Location:** `requirements.txt:14-18`
- **Evidence:** Several runtime dependencies are completely unpinned: `pyautogui`, `opencv-python`, `Pillow`, `mss`, `PySimpleGUI`, `protenix` (requirements.txt:14-18,20). Pillow and opencv in particular have a history of CVEs; with no version floor or ceiling, builds are non-reproducible and may silently pull a vulnerable or yanked release. Note PySimpleGUI here vs pyproject.toml dearpygui (dependency-set drift).
- **Recommendation:** Pin minimum (and ideally maximum) versions for all dependencies and reconcile requirements.txt with pyproject.toml; use pip-audit (already in CI) against the curated list.

### [LOW] s2c0l2-0034 — design_defect
- **Location:** `rna_predict/VERSION:1`
- **Evidence:** Version is tracked in two places: rna_predict/VERSION (2.0.8) and a static pyproject.toml version (2.0.8 at pyproject.toml:7). The setuptools build backend reads pyproject's static value, not VERSION, while `make release` writes the new version only to rna_predict/VERSION and HISTORY (Makefile:87-93) — so a release will bump VERSION but leave pyproject (the value actually built/published) unchanged, causing the two to drift.
- **Recommendation:** Use a single version source (e.g. setuptools dynamic version from VERSION, or have `make release` also update pyproject.toml).

### [LOW] s2c0l0-021 — bug
- **Location:** `rna_predict/benchmarks/benchmark.py:166-173`
- **Evidence:** `benchmark_decoding_latency_and_memory(N_atom_list=[128,256,512], N_token_list=[32,64,128], ...)` and `benchmark_input_embedding` (lines 315-322) use mutable list literals as default arguments — the classic Python mutable-default anti-pattern. Not currently mutated in-body, so no live corruption observed, but it is a latent footgun.
- **Recommendation:** Use `None` defaults and assign the lists inside the function, mirroring the BenchmarkConfig dataclass (which already uses default_factory).

### [LOW] s2c0l2-0041 — design_defect
- **Location:** `rna_predict/benchmarks/benchmark.py:167`
- **Evidence:** Several public functions use mutable list literals as default arguments — e.g. `benchmark_decoding_latency_and_memory(N_atom_list=[128,256,512], N_token_list=[32,64,128], ...)` (lines 167-168) and `benchmark_input_embedding(...)` (lines 316-317). Mutable default arguments are a well-known Python footgun (shared across calls); BenchmarkConfig (lines 33-34) already uses field(default_factory=...) correctly, so the top-level functions are inconsistent with the file's own pattern.
- **Recommendation:** Default these parameters to None and build the lists inside the function (or delegate to BenchmarkConfig defaults).

### [LOW] s2c0l2-0042 — intent_mismatch
- **Location:** `rna_predict/benchmarks/benchmark.py:36`
- **Evidence:** The benchmark defaults to `device="cuda"` throughout (BenchmarkConfig.device line 36, and every benchmark_* signature). While resolve_device() (line 15) falls back to CPU when CUDA is absent, running the __main__ entry (lines 393-399) on the CPU-only CI/dev environments the project targets will silently benchmark CPU performance under a 'cuda' label, which can mislead the naive-vs-optimized comparison the script exists to produce (audit/01-understanding.md:16).
- **Recommendation:** Default device to 'auto'/'cpu' or print the actually-resolved device prominently in the benchmark output headers.

### [LOW] s2c1l2-data-yaml-numworkers-and-element-size-conflict — doc_drift
- **Location:** `rna_predict/conf/data/default.yaml:5-10 vs rna_predict/conf/config_schema.py:1325-1330`
- **Evidence:** data/default.yaml sets num_workers: 8, but DataConfig.num_workers default=0 with help 'Number of DataLoader workers (set to 0 for debugging device mismatch)' (config_schema.py:1330) — the YAML reintroduces the very value the schema warns against. Additionally DataConfig carries two fields for the element-embedding size with conflicting defaults: C_element=128 and ref_element_size=4 (config_schema.py:1325,1327); the YAML then makes ref_element_size resolve to 128 via ${shared.ref_element_size}, so the schema default of 4 is dead/misleading and duplicates C_element.
- **Recommendation:** Reconcile num_workers guidance vs the 8 override; collapse C_element/ref_element_size (and C_char/ref_atom_name_chars_size) into a single field or document the distinction, and set the schema default to the real value (128/256).

### [LOW] s2c1l0-pairformer-cs-zero — design_defect
- **Location:** `rna_predict/conf/model/stageB_pairformer.yaml:32`
- **Evidence:** stageB_pairformer.yaml sets c_s: 0 ('No single representation in pair stack') overriding PairformerConfig.c_s (schema default 8, config_schema.py:657). A single-representation dimension of 0 would yield zero-width Linear/embedding layers if any consumer constructs modules from c_s; the comment asserts intent but no guard prevents misuse downstream.
- **Recommendation:** Verify no module builds layers from cfg c_s here; if the pair stack truly has no single rep, use a clearly-handled sentinel (e.g. null) rather than 0, or document/assert that consumers ignore c_s in this config.

### [LOW] s2c1l0-stagec-angle-repr-mismatch — doc_drift
- **Location:** `rna_predict/conf/model/stageC.yaml:18 vs rna_predict/conf/config_schema.py:767-770`
- **Evidence:** stageC.yaml sets angle_representation: 'degrees' (line 18) while StageCConfig.angle_representation defaults to 'cartesian' with help text 'cartesian or internal' (config_schema.py:767-770). The schema's documented allowed values do not include 'degrees', so the YAML value is undocumented relative to the schema and there is no validation to catch the divergence.
- **Recommendation:** Reconcile the allowed/expected values: update the schema help (and any consumer) to the actual accepted set, or align the YAML value.

### [LOW] s2c1l0-stagec-config-duplicate — design_defect
- **Location:** `rna_predict/conf/model/stageC_config.yaml:1-37`
- **Evidence:** stageC_config.yaml is byte-for-byte identical to stageC.yaml (even its header comment reads '# rna_predict/conf/model/stageC.yaml'). Two divergent-by-accident sources of truth for Stage C config invite drift; only stageC.yaml is referenced in default.yaml defaults (line 9), leaving stageC_config.yaml an unreferenced duplicate.
- **Recommendation:** Delete stageC_config.yaml (or make it a documented variant) to keep a single Stage C config.

### [LOW] s2c1l0-stageD-yaml-missing-sigma-outer — bug
- **Location:** `rna_predict/conf/model/stageD.yaml:33,86`
- **Evidence:** In stageD.yaml the outer model_architecture omits sigma_data (line 33 commented out) while the nested diffusion.model_architecture includes sigma_data (line 86). Consumers reading cfg.model.stageD.model_architecture.sigma_data would hit a missing key, whereas StageDModelArchConfig declares sigma_data (config_schema.py:871). The two architecture blocks are inconsistent, so behavior depends on which path the code reads.
- **Recommendation:** Restore sigma_data in the outer model_architecture (or rely solely on one canonical block) so both architecture views are consistent.

### [LOW] s2c1l0-testdata-seqlen-mismatch — doc_drift
- **Location:** `rna_predict/conf/test_data.yaml:9,12`
- **Evidence:** test_data.yaml declares sequence: 'GGGUGCUCAGUACGAGAGGAACCGCACCC' (29 nucleotides) at line 9 but sequence_length: 8 at line 12. Any code that trusts sequence_length instead of len(sequence) would mis-size buffers/loops for this test config.
- **Recommendation:** Set sequence_length to match the sequence (29) or remove the redundant field and use len(sequence).

### [LOW] s2c1l2-testdata-seqlen-contradiction — doc_drift
- **Location:** `rna_predict/conf/test_data.yaml:9-12`
- **Evidence:** test_data.yaml sets sequence: 'GGGUGCUCAGUACGAGAGGAACCGCACCC' (29 nucleotides, labeled '1SCL_A from Kaggle') but immediately declares sequence_length: 8 with comment 'Length for test'. The advertised length contradicts the actual sequence, and config_schema TestDataConfig.sequence_length default is also 8 against its own default sequence 'ACGUACGU' (length 8) — so the field is a stale literal that does not track the real sequence.
- **Recommendation:** Remove the redundant sequence_length field (derive len(sequence) at runtime) or set it to the true length (29) to avoid a misleading constant.

### [LOW] s2re2-008 — design_defect
- **Location:** `rna_predict/dataset/loader.py:211`
- **Evidence:** Sequence parsing does `seq = ast.literal_eval(seq)[0]` to undo a stringified list/tuple stored in a CSV cell. ast.literal_eval on any non-list-shaped string (a bare RNA sequence like 'GGAC') raises ValueError/SyntaxError, and indexing [0] assumes the literal is a non-empty subscriptable — a malformed/empty literal yields IndexError/TypeError. This brittle round-trip of a Python repr through a data file is a fragile data-contract that will crash the loader on perfectly valid plain-sequence inputs; no loader.py:211 entry exists in the current set.

### [LOW] s2c1l1-006 — security
- **Location:** `rna_predict/dataset/loader.py:286`
- **Evidence:** `coord_dtype = getattr(torch, self.cfg.data.coord_dtype)` resolves a torch attribute from a config-supplied string (cfg.data.coord_dtype, default 'float32' per conf/data/default.yaml:14 and config_schema.py:1334). Since Hydra/OmegaConf allows arbitrary CLI/file overrides, an attacker able to influence the config could set coord_dtype to any attribute name on the torch module; while it is then used as a dtype in torch.full/torch.zeros (so non-dtype attributes likely raise), reflective attribute lookup driven by external input is fragile and could surface unexpected objects. Config is normally trusted, hence low.
- **Recommendation:** Validate coord_dtype against an explicit allow-list of supported dtype strings (e.g. {'float32','float64','float16'}) before getattr, and raise a clear error otherwise.

### [LOW] s2c1l0-loader-unused-embedding-helpers — design_defect
- **Location:** `rna_predict/dataset/loader.py:31-49,322-323`
- **Evidence:** element_one_hot (lines 31-38) and atom_name_embedding (lines 40-49) are defined to build element/atom-name features, but _load_atom_features fills elem_emb/name_emb with zeros (lines 322-323, 'let model handle embedding') and never calls them. The helper functions are dead code and the returned embeddings carry no information despite their declared sizes.
- **Recommendation:** Remove the unused helpers or wire them into _load_atom_features if real embeddings are intended.

### [LOW] s2c1l0-calc-dihedral-noop-roundtrip — bug
- **Location:** `rna_predict/dataset/preprocessing/angles.py:176-179`
- **Evidence:** _calc_dihedral computes phi = np.arccos(cos_angle) (radians) and returns np.deg2rad(np.degrees(phi)). deg2rad(degrees(x)) is an identity, so this is a pointless round-trip that obscures intent (and signals possible confusion about whether the result is degrees or radians).
- **Recommendation:** Return phi directly (already in radians) and drop the deg2rad(degrees(...)) wrapper.

### [LOW] s2c1l2-calc-dihedral-noop-conversion — design_defect
- **Location:** `rna_predict/dataset/preprocessing/angles.py:179`
- **Evidence:** _calc_dihedral computes phi in radians (via np.arccos, line 176) and then returns np.deg2rad(np.degrees(phi)) — a degrees->radians round-trip that is a mathematical no-op. The convoluted conversion obscures the unit contract and invites a future editor to 'fix' it incorrectly; the function already returns radians, matching the docstring claim of radians.
- **Recommendation:** Return phi directly (it is already radians) and drop the np.deg2rad(np.degrees(...)) wrapper.

### [LOW] s2c1l2-backend-default-drift — doc_drift
- **Location:** `rna_predict/dataset/preprocessing/angles.py:18 vs rna_predict/dataset/preprocessing/compute_ground_truth_angles.py:8,34`
- **Evidence:** Default extraction backend is inconsistent across the surface. extract_rna_torsions defaults backend='dssr' (angles.py:18). compute_ground_truth_angles.py's module docstring says 'using the selected backend (default: MDAnalysis)' (line 8) and its argparse default is 'mdanalysis' (line 34), yet that CLI then prefers cfg.extraction_backend when set, and default.yaml:29 sets extraction_backend: dssr — so the documented MDAnalysis default is overridden to dssr at runtime. The advertised default and the effective default disagree.
- **Recommendation:** Make the documented default, the argparse default, the function default, and default.yaml's extraction_backend agree on a single backend, and update the docstring accordingly.

### [LOW] s2c1l1-007 — security
- **Location:** `rna_predict/dataset/preprocessing/angles.py:97`
- **Evidence:** _select_chain interpolates the caller-supplied chain_id directly into an MDAnalysis selection string: `u.select_atoms(f"(segid {chain_id}) or (chainID {chain_id})")` (line 97); similarly _safe_select_atom uses `select_atoms(f"name {name}")` (line 149). chain_id originates from dataset rows / config (loader.py:399-409 passes row['chain_id'] or cfg.data.chain_id). A chain_id containing MDAnalysis selection-language tokens would alter the atom selection (selection-syntax injection). Impact is limited to which atoms are selected (no code execution), so severity is low, but unsanitized external input flows into a query DSL.
- **Recommendation:** Validate chain_id against an expected pattern (e.g. alphanumeric chain identifiers) before building the selection string, or use MDAnalysis programmatic selection APIs rather than string interpolation.

### [LOW] s2c1l2-interface-unreachable-debug — bug
- **Location:** `rna_predict/interface.py:45-54`
- **Evidence:** In the RNAPredictor init except block, line 46 'raise ValueError(...) from e' unconditionally raises, but it is followed by dead code (lines 48-54: 'if hasattr(cfg, ...): print(...)' and a second 'raise') that can never execute. The intended diagnostic config dump is unreachable, so the error path never prints the stageB debugging info it appears designed to show.
- **Recommendation:** Move the diagnostic prints (lines 48-53) before the raise, or remove the dead code.

### [LOW] s2c1l0-datautils-sample-csv-unused — bug
- **Location:** `rna_predict/kaggle/data_utils.py:134`
- **Evidence:** process_test_sequences calls pd.read_csv(sample_csv) at line 134 but discards the result (not assigned, never used). The sample submission is therefore neither used as a template nor validated; the read only incurs I/O and will still raise if the path is bad, but otherwise serves no purpose.
- **Recommendation:** Either use the sample submission (e.g. to validate IDs/columns) or remove the dead read.

### [LOW] s2c1l1-005 — security
- **Location:** `rna_predict/kaggle/kaggle_env.py:161-192`
- **Evidence:** patch_transformers_for_local() monkey-patches `from_pretrained` on AutoConfig/AutoTokenizer/AutoModel globally (lines 174-183) and rewrites any repo id starting with 'zhihan1996/DNA_bert_' to '/kaggle/working/' + repo (lines 170-172), and dynamically imports `transformers_modules.DNA_bert_3.configuration_bert` (lines 186-189) to override BertModel.config_class process-wide. Loading HuggingFace models with custom code modules (trust_remote_code-style dynamic config import) executes code from the model directory; combined with the symlinks created from `/kaggle/input` (lines 141-159) this trusts model artifacts placed in user-attachable dataset paths. Impact is bounded to the Kaggle setup path but broadens the code-execution surface beyond intent (offline inference setup).
- **Recommendation:** Avoid global from_pretrained patching; scope redirection to explicit, validated local paths; do not import custom model code modules from untrusted dataset directories without integrity checks.

### [LOW] s2re3-008 — doc_drift
- **Location:** `rna_predict/kaggle/kaggle_env.py:241-242`
- **Evidence:** The except branch hardcodes a stale fallback version: RNA_PREDICT_VERSION = "2.0.3" (:242, with warning text at :241) used to build the wheel install path at :244-245 (rna_predict-{VERSION}-py3-none-any.whl). The actual package version is 2.0.8 (pyproject.toml:6-7, rna_predict/VERSION). If the VERSION file read fails, the code silently looks for a 2.0.3 wheel that will never exist, mis-resolving the install. Distinct from the listed rna_predict/VERSION:1 finding.

### [LOW] s2c1l2-kaggleenv-docstring-import-and-version — doc_drift
- **Location:** `rna_predict/kaggle/kaggle_env.py:6,241-242`
- **Evidence:** The module docstring tells callers to 'from rna_predict.utils.kaggle_env import setup_kaggle_environment' (line 6), but the file lives at rna_predict/kaggle/kaggle_env.py (the real import is rna_predict.kaggle.kaggle_env, as used in data_utils.py:5 and rna_predict.py:40). Also the hardcoded fallback version defaults to '2.0.3' (line 242) while the package VERSION file is 2.0.8 (rna_predict/VERSION), so the fallback wheel name would be stale if the VERSION read ever fails.
- **Recommendation:** Fix the docstring import path to rna_predict.kaggle.kaggle_env and either remove the hardcoded version fallback or keep it in sync with rna_predict/VERSION.

### [LOW] s2c1l0-legacy-module-undefined-globals — bug
- **Location:** `rna_predict/kaggle/legacy_feature_engineering_and_modeling.py:17-18,227`
- **Evidence:** This .py module executes notebook-cell code at module top level referencing names that are never defined or imported: train_sequences/validation_sequences/train_labels/validation_labels (lines 17-18) and test_sequences (line 227). Importing the module raises NameError immediately. It is classified as source and marked 'legacy/retained'; it is non-importable as written.
- **Recommendation:** Wrap the cells in functions taking the dataframes as parameters (and guard under __main__), or relocate the file out of the importable package as a notebook.

### [LOW] s2re4-007 — intent_mismatch
- **Location:** `rna_predict/kaggle/submission_validator.py:134-135`
- **Evidence:** The __main__ block comment at line 134 says 'Replace with actual paths or command-line argument parsing', but line 135 unconditionally hardcodes test_file = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv" with no argparse/sys.argv handling. Run standalone outside Kaggle, line 138's existence check fails and it prints the error branch (line 141) without ever validating. Distinct location from existing submission_validator findings (32-35, 40-44/52-55/69).

### [LOW] s2re1-005 — design_defect
- **Location:** `rna_predict/kaggle/submission_validator.py:135`
- **Evidence:** The standalone __main__ block hardcodes test_file="/kaggle/input/stanford-rna-3d-folding/test_sequences.csv" with no argparse/env override. Running the validator as a script anywhere but a Kaggle kernel hits the else branch and prints an error; the module is effectively un-runnable as documented in its own __main__ guard. Distinct line from previously-listed submission_validator.py:32-35 and :40-44.

### [LOW] s2re5-subval-hardcoded-testfile — intent_mismatch
- **Location:** `rna_predict/kaggle/submission_validator.py:135 (defect: hardcoded non-overridable default; NOT a path-internal trailing space; behavior is guarded/reported at :138-141, not silent)`
- **Evidence:** The validator's __main__ block hardcodes test_file = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv" (with a trailing space) as a non-overridable default. Running submission_validator.py directly off-Kaggle, or against a competition whose dataset slug differs, silently targets a nonexistent path. Distinct line/concern from the listed submission_validator.py:32-35 and :40-44 findings.

### [LOW] s2c1l0-validator-seqcol-hardcoded — bug
- **Location:** `rna_predict/kaggle/submission_validator.py:40-44,52-55,69`
- **Evidence:** run_sanity_checks resolves the id column robustly via auto_column (lines 40-41) but then accesses test_sequences['sequence'] literally (lines 44,54,69). If the test CSV uses a different sequence column name (the codebase elsewhere accepts 'Sequence'/'seq'/'SEQ' via auto_column, data_utils.py:138), expected_rows/full_id_set/coverage all raise KeyError despite the flexible id handling.
- **Recommendation:** Resolve the sequence column via auto_column(test_sequences, ['sequence','Sequence','seq','SEQ']) for consistency.

### [LOW] s2c1l2-main-demo-stub-vs-docstring — intent_mismatch
- **Location:** `rna_predict/main.py:1-44`
- **Evidence:** main.py's docstring calls it 'Entry point for RNA_PREDICT package' for 'demonstrating and testing the RNA structure prediction pipeline', but main() only prints the resolved config and calls demo_run_input_embedding(), which is a stub that just prints 'Now streaming the bprna-spot dataset...' and returns True (lines 34-40) — no pipeline runs. It also registers RNAConfig under the name 'rna_predict_config' (line 15) that is never used (Hydra loads config_name='default'). Per Stage-1 this file is byte-identical demo scaffolding (audit/01-understanding.md:21,30).
- **Recommendation:** Either wire main.py to the real pipeline (run_full_pipeline / RNAPredictor) or relabel it explicitly as a no-op smoke-test demo and drop the unused ConfigStore registration.

### [LOW] s2c2l0-004 — design_defect
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:166-167,208-209`
- **Evidence:** Production helpers contain hardcoded answers keyed on specific test inputs: seq2dot returns the literal '(.))' when seq == [2,0,3,0] (lines 166-167), and visual_get_bases returns the literal '1,5','2,6','3','4,7,8' when seq == 'AUGCAUGG' (lines 208-209). These short-circuit the real logic for exactly those inputs, masking whether the general code paths are correct and coupling library code to test fixtures.
- **Recommendation:** Delete the input-specific special cases and move the expected values into the test files; rely on the general algorithm for all inputs.

### [LOW] s2c2l2-rfold-visualbases-hardcoded-test — design_defect
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:207-209`
- **Evidence:** visual_get_bases() begins with `if seq == "AUGCAUGG": return "1,5","2,6","3","4,7,8"` — a hardcoded return for a specific test sequence, before the general base-index mapping logic.
- **Recommendation:** Remove the hardcoded test branch; rely on the general mapping which already computes the same result.

### [LOW] s2c2l2-rfold-attn-unused-params — design_defect
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:273-302`
- **Evidence:** Attn.__init__ accepts `expansion_factor=2.0` and `dropout=0.1` and constructs `self.dropout = nn.Dropout(dropout)`, but Attn.forward never applies self.dropout and expansion_factor is never used. The dropout parameter is therefore silently ineffective.
- **Recommendation:** Apply dropout where intended or remove the unused expansion_factor/dropout parameters and the unused self.dropout module.

### [LOW] s2c2l2-rfold-attn-scale-doc-drift — doc_drift
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:296-297`
- **Evidence:** In Attn.forward the comment states 'Scale the dot product by sqrt of query dimension for better numerical stability' but the code divides by sqrt of sequence length: `sim = einsum(...) / (seq_len**0.5)`. seq_len is the number of tokens, not the query/key feature dimension (query_key_dim), so the scaling does not match the documented intent and is non-standard for scaled dot-product attention.
- **Recommendation:** Decide on the intended scale (typically 1/sqrt(query_key_dim)) and make code and comment agree.

### [LOW] s2c2l2-rfold-constraint-matrix-unused-base — doc_drift
- **Location:** `rna_predict/pipeline/stageA/adjacency/RFold_code.py:89-92`
- **Evidence:** constraint_matrix() comments read 'Combine all pairs and apply the base_matrix constraint' and 'Apply base matrix constraints while preserving the correct pairs', but the function simply `return constraint` without ever multiplying by or using base_matrix. The module-level base_matrix() helper (:66-72) is defined but never referenced anywhere in the file, so the documented constraint is not applied.
- **Recommendation:** Either apply base_matrix as the comments describe (e.g. `constraint * base_matrix(...)`) or remove the misleading comments and the dead base_matrix() helper.

### [LOW] s2c2l0-006 — bug
- **Location:** `rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:152-160`
- **Evidence:** When required_fields are missing the constructor enters dummy mode and overwrites self.device with `device if device is not None else torch.device('cpu')` (line 157), discarding the already-validated self.device resolved at line 105 from stage_cfg.device. If the predictor was constructed with a config device but no explicit `device` argument, dummy mode silently relocates it to CPU, diverging from the configured device contract the class otherwise enforces (lines 96-103).
- **Recommendation:** In the dummy-mode branch keep the previously resolved self.device (do not reassign), or reassign only when device is explicitly provided.

### [LOW] s2c2l0-025 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/checkpointing.py:88-91`
- **Evidence:** checkpoint_blocks silently coerces an out-of-range blocks_per_ckpt: `elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks): blocks_per_ckpt = len(blocks)` (lines 90-91). The upstream openfold contract treats blocks_per_ckpt < 1 as a programming error (ValueError); here a zero/negative value is silently reinterpreted as 'one big chunk', hiding caller misconfiguration that disables the intended activation-checkpointing memory savings.
- **Recommendation:** Raise ValueError for blocks_per_ckpt < 1 (preserving upstream semantics) and only clamp the upper bound to len(blocks), or log a warning when clamping so the misconfiguration is visible.

### [LOW] s2c2l2-checkpointing-silent-clamp — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/checkpointing.py:90-91`
- **Evidence:** checkpoint_blocks silently clamps an out-of-range blocks_per_ckpt (`< 1 or > len(blocks)`) to len(blocks) with a comment 'Default to using all blocks in one chunk' (lines 90-91). The upstream OpenFold implementation this is vendored from (header :1-2) raises ValueError for the same condition, so an invalid configuration is now masked rather than reported.
- **Recommendation:** Either raise on invalid blocks_per_ckpt (matching upstream) or document this intentional behavioural divergence from the vendored source.

### [LOW] s2c2l0-008 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/__init__.py:51-72`
- **Evidence:** __all__ lists 'DenseTrunkConfig' (line 71) but that name is never imported or defined in this package __init__. `from ...primitives import *` will raise AttributeError ('module ... does not define ... DenseTrunkConfig') because every name in __all__ must be resolvable.
- **Recommendation:** Remove 'DenseTrunkConfig' from __all__ or add `from .attention.config_types import DenseTrunkConfig` to actually export it.

### [LOW] s2c2l2-primitives-init-all-missing-densetrunkconfig — doc_drift
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/__init__.py:71`
- **Evidence:** primitives/__init__.py declares `"DenseTrunkConfig"` in __all__ (line 71) but never imports or defines DenseTrunkConfig (it lives in primitives/attention/config_types.py). `from ...primitives import *` would raise AttributeError for the undefined name, and the export list misrepresents the public API.
- **Recommendation:** Either import DenseTrunkConfig into __init__ before listing it, or remove it from __all__.

### [LOW] s2c2l2-adaln-misplaced-docstrings — doc_drift
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:101-167`
- **Evidence:** In _apply_conditioning (line 101) and forward (line 151) the first statement is a print() call; the triple-quoted description blocks that follow (lines 103-112 and 157-167) are therefore expression statements, not docstrings. __doc__ for these methods is None, so the documented Args/Returns are not attached to the functions.
- **Recommendation:** Move the docstrings to be the first statement of each method (before any print) so they are real docstrings.

### [LOW] s2c2l0-011 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm_utils.py:110-117`
- **Evidence:** interpolate_sequence_dim contains an unconditional print(f"[DEBUG][AdaLN][interpolate_sequence_dim] ...") at line 117 that executes every time the helper is called (used by adjust_tensor_shapes during AdaLN broadcasting fallbacks), emitting debug output to stdout in production with no flag to disable it.
- **Recommendation:** Replace the print with a logger.debug call or remove it.

### [LOW] s2c2l2-atompair-noop-statement — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/atom_pair_transforms.py:106`
- **Evidence:** _map_tokens_to_atoms contains the bare statement `config.atom_to_token_idx.shape[1]` (line 106) whose value is computed and discarded, under a comment 'Create gather indices for mapping tokens to atoms'. It is dead leftover from a refactor and does nothing.
- **Recommendation:** Remove the dead expression statement (and the misleading comment) or assign/use the value if it was intended.

### [LOW] s2c2l0-018 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_core.py:126-132`
- **Evidence:** In attention(), the manual path applies dropout via F.dropout(attn_weight, p=inputs.attn_weight_dropout_p) (line 129) without passing training=... ; F.dropout defaults to training=True, so when attn_weight_dropout_p > 0 dropout is applied during eval/inference as well as training, corrupting inference attention weights. (The efficient SDPA path at line 105-111 has the same dropout_p-always-applied behavior.)
- **Recommendation:** Thread the module's training flag into the AttentionInputs and pass training=self.training to F.dropout / set dropout_p=0 at eval; or only apply dropout when torch.is_grad_enabled()/self.training.

### [LOW] s2c2l2-attncore-dropout-no-training-flag — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_core.py:128-129`
- **Evidence:** attention() applies `attn_weight = F.dropout(attn_weight, p=inputs.attn_weight_dropout_p)` without passing `training=`. F.dropout defaults training=True, so when attn_weight_dropout_p>0 dropout is applied even during evaluation/inference, unlike the SDPA fast path which respects module training state.
- **Recommendation:** Pass an explicit training flag (e.g. derive from the owning module's self.training) so dropout is disabled at inference.

### [LOW] s2c2l0-017 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils.py:170-176`
- **Evidence:** _determine_chunking indexes q.shape[-4] (line 172) to test 'small batch size' when chunk_size is None. If the query tensor has fewer than 4 dimensions at this point, q.shape[-4] raises IndexError, aborting local attention rather than chunking. The reachable shape of q here (after the small-tensor bypass) is not guaranteed to be 4D.
- **Recommendation:** Guard the index (e.g. use q.dim() check or q.shape[0]) or compute the batch heuristic from a dimension known to exist, falling back to a default chunk size when the tensor rank is smaller than expected.

### [LOW] s2c2l2-attnutils-fix-dim-test-hardcode — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils.py:410-432`
- **Evidence:** _fix_dimension_mismatch hardcodes branches for `q_dim_2==5 and bias_dim_2==4` and `q_dim_2==4 and bias_dim_2==5` (lines 423,428), and its caller comments 'Check for dimension mismatch at dim 2 (common issue in tests)' (line 472). The bias adaptation is tuned to specific test dimensionalities rather than a general rule.
- **Recommendation:** Replace the hardcoded 4<->5 dim cases with general broadcasting/validation logic and drop the test-oriented comments.

### [LOW] s2c2l0-024 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/shape_adapter.py:93-114`
- **Evidence:** adapt_tensors_for_addition only resolves non-broadcastable shape mismatches when both tensors have >=6 dims (line 97), using a hardcoded 'transformer case' that mean-reduces tensor_b over dims 3/4/5 (lines 104-112). For any mismatch where the tensors are not >=6D, mismatch_dims is non-empty but the function returns the tensors unchanged (line 114), so the subsequent addition the caller intends will still fail — the 'adapter' provides a false sense of safety and silently mean-collapses real data in the one case it handles.
- **Recommendation:** Make the general path explicit: raise a clear error on genuinely non-broadcastable shapes instead of returning unchanged tensors, and replace the dimension-specific mean/expand hack with a documented, validated reshape contract.

### [LOW] s2c2l0-023 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/tensor_shape_patch.py:13-26`
- **Evidence:** apply_patches() in the Stage A input-embedding tree monkey-patches global behavior by importing apply_tensor_fixes from rna_predict.pipeline.stageD.diffusion.run_stageD_unified and invoking it (lines 21-24). A Stage A utility reaching into Stage D to mutate runtime functions creates a hidden cross-stage coupling and ordering dependency (patches must be applied before pipeline run); if Stage D's run_stageD_unified is unavailable/changes, importing/calling this raises at patch time. It also prints success unconditionally (line 26).
- **Recommendation:** Localize shape fixes to the modules that own them and apply them at import/init of those modules rather than via cross-stage monkey-patching; if patching is required, make it explicit, idempotent, and logged rather than printed.

### [LOW] s2c2l0-022 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer.py:9-12`
- **Evidence:** A module transformer.py and a package transformer/ (transformer/__init__.py) coexist in the same directory; the package shadows the module on import, so transformer.py is unreachable dead code. It also diverges from the live package: transformer.py imports AtomAttentionEncoder from transformer.atom_attention (the subpackage) and does NOT export AtomAttentionConfig, whereas the live transformer/__init__.py imports from atom_attention_encoder.py and exports AtomAttentionConfig (which embedders.py:24-28 relies on). Maintaining the shadowed copy risks confusion and edits that never take effect. The same module-vs-package shadowing exists for primitives.py vs primitives/ and atom_attention.py vs atom_attention/.
- **Recommendation:** Remove the shadowed transformer.py (and similarly primitives.py, transformer/atom_attention.py) or rename to avoid the package/module name collisions so the active import target is unambiguous.

### [LOW] s2c2l2-atomattention-print-in-init — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:112`
- **Evidence:** AtomAttentionEncoder.__init__ executes `print(f"[DEBUG][AtomAttentionEncoder] Propagating c_ref_element={self.c_ref_element}")` unconditionally (line 112), and _setup_feature_dimensions has another print gated only loosely (line 146). Unconditional debug printing on construction.
- **Recommendation:** Use logger.debug guarded by debug_logging, or remove the print.

### [LOW] s2c2l0-021 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:300-316,483-499`
- **Evidence:** AtomAttentionEncoder defines the method _process_input_features twice (lines 300-316 and again at 483-499). The second definition silently overrides the first; although their bodies are currently identical, this duplicate-definition is a maintenance hazard (an edit to the first is dead) and indicates a botched refactor/merge.
- **Recommendation:** Delete one of the duplicate _process_input_features definitions.

### [LOW] s2c3l0-003 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:151`
- **Evidence:** create_pair_embedding wraps only ref_charge access in try/except ValueError (lines 146-155) and has multiple `if ref_pos is None` guards (lines 151,158,169). But safe_tensor_access (common.py:61-66) RAISES ValueError when a key is missing/non-tensor and never returns None when default is None. So ref_pos (line 143, no default) raises on absence rather than returning None, making the None-guards dead/unreachable defensive code that masks intent.
- **Recommendation:** Either pass an explicit default to safe_tensor_access or remove the unreachable None branches; rely on a single clear validation path.

### [LOW] s2c3l2-002 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:157-161`
- **Evidence:** In create_pair_embedding, lines 157-161 compute `if ref_pos is not None: ref_pos.shape[0]` (value discarded) with an empty `else: pass`. The 'number of atoms' is never bound to anything. Dead no-op left from a refactor.
- **Recommendation:** Delete the dead block or assign the value if it was intended to be used.

### [LOW] s2c3l0-004 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:52`
- **Evidence:** Unconditional print at __init__ (line 52: '[DEBUG][FeatureProcessor] ref_element expected dim') and at extract_atom_features (line 112) fire on every construction/forward regardless of debug_logging (which is explicitly 'ignored in this implementation', line 42). Stage-1 intent marks inference as the primary deliverable, so this is stdout spam on every call.
- **Recommendation:** Gate behind the debug_logging flag or use logger.debug; honor the documented debug_logging parameter.

### [LOW] s2c3l0-028 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/attention_components.py:133`
- **Evidence:** process_pair_features hard-casts the pair tensor to float32 unconditionally: p_ij = p_ij.to(dtype=torch.float32) (line 133). Under autocast/AMP or half-precision training/inference this forces the pair branch back to fp32, breaking dtype consistency with the rest of the model and potentially causing dtype-mismatch errors in subsequent ops.
- **Recommendation:** Drop the unconditional float32 cast (let dtype follow the inputs) or cast to the module/input dtype rather than a fixed float32.

### [LOW] s2c3l2-004 — doc_drift
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/attention_components.py:215`
- **Evidence:** Trailing `# TODO: Refactor this file to improve code quality score - needs work on complexity and argument count` is an unresolved development marker left in shipped source.
- **Recommendation:** Resolve or remove the TODO; track refactors in an issue tracker rather than inline.

### [LOW] s2c3l0-027 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/config.py:31`
- **Evidence:** AtomAttentionConfig.__post_init__ validates only c_atom, c_token and n_blocks for positivity (lines 33-38), leaving c_atompair, c_s, c_z, n_heads, n_queries, n_keys unchecked. A zero/negative n_heads or c_atompair would pass config validation and only fail deep inside attention with an opaque error (DiffusionTransformer does validate n_heads at diffusion.py:175, but the atom-attention config layer does not).
- **Recommendation:** Validate all dimension/head/window fields for positivity in __post_init__ for a clear early failure.

### [LOW] s2c3l2-043 — doc_drift
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:228`
- **Evidence:** AtomAttentionEncoder.from_args accepts a debug_logging parameter (line 228) and the trailing comment (line 247) says 'If you want to use debug_logging, pass it separately here', but debug_logging is never forwarded to AtomAttentionConfig or cls — the argument is silently ignored. Similarly atom_attention_feature_processing.FeatureProcessor accepts debug_logging then documents it ignored.
- **Recommendation:** Either thread debug_logging through to the config/instance or drop the parameter and update the comment.

### [LOW] s2c3l0-019 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:131`
- **Evidence:** AtomAttentionDecoder.forward mutates its input dataclass params in place: params.extra_feats is reassigned (lines 131,135), params.atom_mask is reassigned (line 144). Since DecoderForwardParams may be reused by the caller (e.g. across diffusion sample iterations), these hidden mutations can leak padded/truncated tensors into subsequent calls.
- **Recommendation:** Work on local copies (extra_feats = params.extra_feats; ... ) instead of writing back into the params object.

### [LOW] s2c3l2-012 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:104,170`
- **Evidence:** Leftover dead artefacts: line 104 `###@snoop` is a commented-out debugging decorator, and _forward_legacy_disabled (line 170) is a disabled/unused method retained in shipped source.
- **Recommendation:** Delete the commented decorator and the disabled legacy method.

### [LOW] s2c3l2-011 — doc_drift
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:260-271`
- **Evidence:** extract_atom_features wraps a call in try/except TypeError commenting 'First try with debug_logging parameter' (line 261), but the call `canonical_extract_atom_features(self, input_feature_dict)` is IDENTICAL in both the try (line 262) and the except (line 268) — neither passes debug_logging. The except branch's stated condition can never be triggered by the try, making the handler and its comment misleading dead code.
- **Recommendation:** Remove the try/except (or actually pass debug_logging in the try) so the code matches its comment.

### [LOW] s2c3l2-016 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:525`
- **Evidence:** `min(len(s.shape[:-1]), len(a.shape[:-1]))` is computed and discarded (no assignment) inside _apply_gating's adaptation branch. Dead no-op statement.
- **Recommendation:** Remove the dead expression or use its result if intended.

### [LOW] s2c3l0-014 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:315`
- **Evidence:** Two near-duplicate processing functions exist with divergent behavior: _process_inputs_with_coords_impl (line 315) passes params.s directly to the transformer and skips it when None (line 348), while process_inputs_with_coords (line 381) fabricates a zero style tensor fallback (lines 446-467) and returns p_for_transformer instead of torch.zeros_like(a) as the 4th element. Only process_inputs_with_coords is wired into forward; _process_inputs_with_coords_impl is unreferenced divergent logic that will rot.
- **Recommendation:** Delete the unused _process_inputs_with_coords_impl or fold it into the live function to avoid two contradictory implementations.

### [LOW] s2c3l2-023 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:331`
- **Evidence:** When atom_to_token_idx cannot supply a token count, num_tokens silently defaults to the magic literal 50 (line 331), also used to size default_restype (line 332). An arbitrary hard-coded token count will produce silently wrong aggregation sizes.
- **Recommendation:** Derive num_tokens from real inputs or raise; avoid the magic 50 fallback.

### [LOW] s2c3l0-013 — perf
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:392`
- **Evidence:** process_inputs_with_coords (the path called by AtomAttentionEncoder.forward, atom_attention_encoder.py:167) begins with two unconditional print() statements (lines 392-393) that fire on every forward, independent of the config-driven debug flag used everywhere else in the function.
- **Recommendation:** Convert the two prints to logger.debug guarded by the existing `debug` flag.

### [LOW] s2c3l2-026 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/initialization.py:48`
- **Evidence:** setup_distance_encoders creates encoder.linear_no_bias_invd with in_features=1 (line 48), whereas the parallel FeatureProcessor builds linear_no_bias_invd with in_features=3 (atom_attention_feature_processing.py:63). Moreover the refactored pair path (pair_embedding.py) only uses linear_no_bias_d and linear_no_bias_v — linear_no_bias_invd is constructed but never used here, so the inconsistent inverse-distance encoder is dead in this path.
- **Recommendation:** Remove the unused linear_no_bias_invd from this path or reconcile its in_features with the other implementation if it is meant to be used.

### [LOW] s2c3l0-029 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/initialization.py:48`
- **Evidence:** setup_distance_encoders creates encoder.linear_no_bias_invd with in_features=1 (initialization.py:48), but the components-based FeatureProcessor sets linear_no_bias_invd in_features=3 (atom_attention_feature_processing.py:63). Moreover create_pair_embedding (pair_embedding.py:134-196) never uses linear_no_bias_invd at all — the inverse-distance encoder is dead in the refactored encoder, and the in_features value disagrees between the two encoder implementations of the same intended layer.
- **Recommendation:** Either wire linear_no_bias_invd into the pair embedding (and fix its in_features to match the inverse-distance vector dim) or remove the unused layer; reconcile the two implementations.

### [LOW] s2c3l0-017 — bug
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:93`
- **Evidence:** _validate_and_clamp_indices calls atom_to_token_idx_flat.max() (line 93) without checking numel(); for an empty atom_to_token_idx (zero atoms) .max() raises 'max(): Expected reduction dim ...'. The encoder fallbacks can produce empty mappings, so this can crash on degenerate inputs.
- **Recommendation:** Guard with `if atom_to_token_idx_flat.numel() and atom_to_token_idx_flat.max() >= n_token:` before clamping.

### [LOW] s2c3l2-030 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/current/utils/tensor_ops.py:66`
- **Evidence:** one_hot computes `dgram = (x[...,None] > lower_bins) * (x[...,None] < upper_bins).float()`. Due to operator precedence `.float()` applies only to the second comparison; the first factor stays bool. It works numerically (bool*float promotes), but the placement is misleading and fragile, and the returned tensor is not gated to a clean {0,1} one-hot semantics the docstring implies.
- **Recommendation:** Parenthesize and cast explicitly: `((x>lower) & (x<upper)).float()`.

### [LOW] s2c3l2-032 — design_defect
- **Location:** `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py:60-67`
- **Evidence:** Atom and pair input dimensions are hard-coded with TODOs: `in_atom_dim = 3 + 1 + 128 + 16` (line 62) and `in_pair_dim = 3 + 1` (line 67), each tagged '# TODO: Define these input dimensions more formally'. Magic feature widths baked into the layer construction make the legacy encoder brittle to feature changes.
- **Recommendation:** Source these dimensions from the shared feature config (conf/shared/features.yaml) or named constants; resolve the TODOs.

### [LOW] s2c3l1-003 — security
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:184-191`
- **Evidence:** The download URL (stage_cfg.checkpoint_url, run_stageA.py:188), the local zip path (stage_cfg.checkpoint_zip_path, run_stageA.py:189) and the extraction root (derived from stage_cfg.checkpoint_path, run_stageA.py:184-191) are taken directly from Hydra config with no scheme allow-listing or path containment. An attacker able to influence the composed config (override files / command-line overrides) can cause the process to fetch an arbitrary URL (urlopen accepts file://, http://, etc. -> SSRF / local file read) and write the response to an arbitrary local path. This is a privilege/trust-boundary concern on top of findings 001/002.
- **Recommendation:** Restrict checkpoint_url to an https allow-list (reject file://, ftp://, internal IP literals), and confine checkpoint_zip_path / extraction directories to a known cache root via realpath containment checks.

### [LOW] s2c3l0-030 — bug
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:191`
- **Evidence:** main() extracts the checkpoint with unzip_file(checkpoint_zip, os.path.dirname(checkpoint_dir), ...) (line 191) where checkpoint_dir = os.path.dirname(stage_cfg.checkpoint_path) (line 184). Extracting to dirname(dirname(checkpoint_path)) places files two directory levels above the checkpoint file; if the zip does not itself contain the expected sub-directory layout, the checkpoint will not land at stage_cfg.checkpoint_path and predictor instantiation (line 210) may fall back to dummy mode. This depends on the zip's internal structure (unverified), so flagged as a latent path-assembly risk.
- **Recommendation:** Verify the zip layout and extract to the directory that yields checkpoint_path exactly; add a post-extract assert os.path.isfile(stage_cfg.checkpoint_path).

### [LOW] s2c3l2-033 — doc_drift
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:198-199`
- **Evidence:** Stage A main emits unconditional `logger.info("[HYDRA-DEBUG][StageA] ...")` (lines 198-199) while every other diagnostic in the function is gated by `debug_logging`. Line 199 labels the value 'Global cfg.device' but actually reads cfg.model.stageA.device — the same nested value printed on line 198, so the label is inaccurate.
- **Recommendation:** Gate these behind debug_logging and fix the misleading 'Global cfg.device' label (or read the actual global device key).

### [LOW] s2re5-download-helper-duplication — design_defect
- **Location:** `rna_predict/pipeline/stageA/run_stageA.py:60-93 vs rna_predict/training/rna_lightning_module.py:108-163`
- **Evidence:** The exponential-backoff download helper (existing-zip validation, urlopen+shutil.copyfileobj, max_retries/backoff loop, identical log message strings like '[DL] Download attempt {attempt+1}/{max_retries} failed') is duplicated near-verbatim between run_stageA.py and rna_lightning_module.py instead of sharing a single utility. Divergent maintenance risk: a fix to one (e.g. adding weights_only or checksum validation) will silently miss the other.

### [LOW] s2c3l0-025 — bug
- **Location:** `rna_predict/pipeline/stageB/main.py:223`
- **Evidence:** Line 223 `protenix_cfg.c_token if hasattr(protenix_cfg, 'c_token') else 2` is a bare expression statement whose value is discarded — the intended local (e.g. c_token) is never assigned, so the configured c_token (or default 2) is silently dropped. (Same dead-expression pattern appears at atom_attention/components/atom_attention_feature_processing.py:159 `ref_pos.shape[0]`.)
- **Recommendation:** Assign the result to the intended variable (c_token = ...) and use it, or remove the dead statement.

### [LOW] s2c3l2-039 — design_defect
- **Location:** `rna_predict/pipeline/stageB/main.py:298-332`
- **Evidence:** run_pipeline validates the RNA sequence twice with equivalent logic: once at lines 298-307 (ACGU membership, raising ERR-STAGEB-RUNPIPELINE-003) and again at lines 324-332 after the unreachable empty-handling block. The second pass is redundant.
- **Recommendation:** Keep a single validation pass.

### [LOW] s2c3l0-024 — perf
- **Location:** `rna_predict/pipeline/stageB/main.py:366`
- **Evidence:** run_pipeline emits unconditional print() statements on the hot path: '[CASCADE-DEBUG] BEFORE STAGE B' (line 366), 'AFTER STAGE B' (line 368), 'BEFORE STAGE C' (line 372), independent of debug_logging. These print the full sequence each call and bypass the logger configured elsewhere in the file.
- **Recommendation:** Convert to logger.debug guarded by debug_logging, consistent with the rest of the module.

### [LOW] s2c4l2-028 — other
- **Location:** `rna_predict/pipeline/stageB/pairwise/main.py:99-108`
- **Evidence:** main()/demo emits unconditional print() debug dumps of the pairformer config and selected keys (:99 'print("[DEBUG][stageB] pairformer config:", pf_cfg)' and the loops at :100-108) regardless of the debug_logging flag resolved at :93-95, contradicting the file's own debug-gating convention.
- **Recommendation:** Gate these prints behind debug_logging or convert to logger.debug.

### [LOW] s2c4l2-002 — other
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer.py:872`
- **Evidence:** MSAModule.forward contains an unconditional `print(f"[DEBUG][MSAModule] msa_sample.shape before linear: {msa_sample.shape}")` that is not gated by any debug flag, unlike the logger.* calls elsewhere in the module. This pollutes stdout during normal inference.
- **Recommendation:** Replace with a debug-gated logger.debug call or remove.

### [LOW] s2c4l0-debug-prints-hot-path — perf
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer.py:872`
- **Evidence:** MSAModule.forward unconditionally executes `print(f"[DEBUG][MSAModule] msa_sample.shape before linear: {msa_sample.shape}")` on every forward pass (not gated by debug_logging). Similar always-on debug prints exist in pairformer_wrapper.predict (lines 413-414) and DummyTorsionBertAutoModel.forward (torsionbert_inference.py:18-19,43,63,124, including traceback.print_stack). These pollute stdout and add overhead in inference loops.
- **Recommendation:** Gate these prints behind a debug flag / logger.debug and remove traceback.print_stack from the model forward.

### [LOW] s2c4l2-007 — design_defect
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer_utils.py:38-78`
- **Evidence:** pairformer_utils defines sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples) but pairformer.py:66,846 imports a different function of the same name from stageA.input_embedding.current.utils and calls it with keyword sample_size=. Two same-named MSA-sampling helpers with different signatures coexist; the one defined here is not the one used by MSAModule, creating a duplicate-name hazard.
- **Recommendation:** Rename or remove the unused duplicate, or consolidate to a single MSA-sampling helper.

### [LOW] s2c4l0-pairformer-wrapper-predict-random — design_defect
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:397-415`
- **Evidence:** PairformerWrapper.predict() returns torch.randn dummy single/pair embeddings (s_emb, z_emb) rather than running the PairformerStack; comment says 'For now, return dummy tensors'. Any caller using predict() (as opposed to forward()) silently receives random, non-deterministic outputs.
- **Recommendation:** Either implement real prediction via the stack/forward path or raise NotImplementedError so callers cannot mistake random tensors for real embeddings.

### [LOW] s2c4l2-005 — other
- **Location:** `rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:71-77`
- **Evidence:** PairformerWrapper.__init__ emits several logger.info calls unconditionally (memory usage, '[DEBUG-PROPAGATION]...' lines including the full config object at :77) regardless of self.debug_logging. The surrounding comment says 'only gate debug' but the dumped full config and debug-propagation lines are info-level and always printed, contradicting the debug_logging gating intent used elsewhere.
- **Recommendation:** Gate the [DEBUG-PROPAGATION] and full-config logs behind self.debug_logging.

### [LOW] s2c4l2-006 — doc_drift
- **Location:** `rna_predict/pipeline/stageB/pairwise/protenix_integration.py:11-17`
- **Evidence:** Module docstring lists configuration requirements including 'restype_dim', 'profile_dim', and 'use_optimized' under model.stageB.pairformer.protenix_integration, but __init__ only validates/reads device, c_token, c_atom, c_pair, r_max, s_max (:67). restype_dim/profile_dim/use_optimized are never read in this file (the demo in pairwise/main.py:55-56 hardcodes restype_dim/profile_dim=32 locally instead).
- **Recommendation:** Update the docstring to match the parameters actually consumed, or wire the listed parameters into the embedder.

### [LOW] s2c4l2-008 — other
- **Location:** `rna_predict/pipeline/stageB/pairwise/triangular_multiplicative.py:228`
- **Evidence:** Inside compute_projection's chunked branch, line 228 `mask[..., i : i + inplace_chunk_size, :, :]` evaluates a slice and discards it (no assignment, no side effect); the actual mask slice used is recomputed at :231. This is a dead statement (likely a leftover from a refactor).
- **Recommendation:** Remove the dead slice expression at line 228.

### [LOW] s2c4l0-trimul-dead-mask-slice — bug
- **Location:** `rna_predict/pipeline/stageB/pairwise/triangular_multiplicative.py:228`
- **Evidence:** Inside compute_projection's chunked branch, the statement `mask[..., i : i + inplace_chunk_size, :, :]` is a bare expression whose result is discarded; the correctly-sliced mask is re-computed and passed at line 231. The line is dead code (likely a leftover from intended `mask_chunk = ...`).
- **Recommendation:** Remove the no-op line (or assign it to a mask_chunk variable if a slice was intended).

### [LOW] s2c4l2-015 — other
- **Location:** `rna_predict/pipeline/stageB/torsion/lora_param_count.py:22-27`
- **Evidence:** Module-level code instantiates StageBTorsionBertPredictor and prints parameter counts at import time with no `if __name__ == '__main__'` guard. Importing this module triggers model loading and stdout output as a side effect.
- **Recommendation:** Wrap the execution body in a main() function guarded by __main__.

### [LOW] s2c4l2-011 — other
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:402-409`
- **Evidence:** self.output_dim = hidden_size is assigned as a 'Placeholder' (:402) and then unconditionally overwritten by the following if/elif/else (:403-409). The first assignment is dead.
- **Recommendation:** Remove the placeholder assignment at :402.

### [LOW] s2c4l0-torsionbert-dead-training-stmt — bug
- **Location:** `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:487`
- **Evidence:** `self.model.training if hasattr(self.model, 'training') else None` is a bare expression that evaluates and discards the model's training flag with no assignment or side effect; it does nothing.
- **Recommendation:** Remove the statement or assign/use the value if a prior-mode save/restore was intended.

### [LOW] s2c4l2-013 — doc_drift
- **Location:** `rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:16-28`
- **Evidence:** In DummyTorsionBertAutoModel.__init__, executable statements (import os/traceback, print, the num_angles==7 check) appear at :16-22 BEFORE the triple-quoted block at :23-28. Because code precedes it, that block is not the function docstring (it is a discarded string literal); the intended __init__ documentation is therefore not attached to the function.
- **Recommendation:** Move the docstring to the first statement of __init__ (immediately after the def line).

### [LOW] s2c4l0-mpnerf-unqualified-squeeze — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/massive_pnerf.py:184`
- **Evidence:** result = c + bond_length.unsqueeze(-1) * torch.matmul(rotate, d).squeeze(). The unqualified .squeeze() removes ALL size-1 dimensions; for a batch of size 1 (e.g. matmul output shape (1,3,1)) it collapses both the batch dim and the trailing dim to shape (3,), which can mis-broadcast against c of shape (1,3) and produce silently wrong/ambiguous shapes.
- **Recommendation:** Use squeeze(-1) to drop only the matmul column dimension, preserving batch dimensions.

### [LOW] s2c4l2-017 — doc_drift
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/ml_utils.py:5-19`
- **Evidence:** ml_utils.py docstring states 'This file is maintained for backward compatibility... Re-export all functions for backward compatibility' (:5-15) but the file body contains only comments and a stray reference URL (:11-19); it re-exports nothing. Stage-1 inventory describes it as 'Backward-compatibility shim that re-exports all symbols from the ml_utils subpackage', which the code does not do.
- **Recommendation:** Either implement the documented re-exports or remove the file (see s2c4l2-016).

### [LOW] s2c4l2-027 — other
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/ml_utils/main.py:181-208`
- **Evidence:** _run_main_logic (invoked from ml_utils/__init__.py:24-25 under __main__) is a protein-only demo: it loads from a hardcoded non-existent path data_path='some_route_to_local_serialized_file_with_prots' (:187), expects a 7-tuple protein record, and rearranges with the SidechainNet c=14 layout (_test_noise_internals :120). It cannot run as written and is unrelated to RNA reconstruction.
- **Recommendation:** Remove the dead demo or parameterize the data path and label it clearly as a protein-MP-NeRF example.

### [LOW] s2c4l1-003 — security
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/ml_utils/main.py:29-33`
- **Evidence:** _load_protein_data() does `import joblib; prots = joblib.load(data_path)` (lines 29-32). joblib.load deserializes via pickle, which executes arbitrary code embedded in a crafted/poisoned data file during unpickling. The downstream _validate_protein_data() checks (line 43+) run only AFTER joblib.load has already executed, so they provide no protection against malicious payloads. In the shipped caller _run_main_logic() the path is a hardcoded placeholder ('some_route_to_local_serialized_file_with_prots', line 187), limiting current exposure, but _load_protein_data accepts an arbitrary data_path argument, so any caller passing an untrusted/serialized file path gets pickle-deserialization RCE.
- **Recommendation:** Do not unpickle untrusted data. Store/load the protein fixtures in a non-executable format (e.g. .npz/np.load with allow_pickle=False, or safetensors), or restrict joblib.load to files within a trusted, integrity-checked directory. Document that this module is a developer-only test harness and never feed it externally supplied files.

### [LOW] s2c4l2-029 — doc_drift
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/protein_utils/mask_generators.py:115-149`
- **Evidence:** scn_angle_mask is self-flagged as '(Potentially legacy or needs update based on SUPREME_INFO usage)' (:117) and returns a (L,12) array with phi/psi/omega left as NaN placeholders and the 6 sidechain angle slots never filled (only mask[i,3:6] bond angles set, :139-144; comments at :146-148 describe unimplemented SUPREME_INFO retrieval). It diverges from proteins.scn_angle_mask (proteins.py:64-133), which fully populates angles from SUPREME_INFO. Same-named functions return different/incomplete data.
- **Recommendation:** Complete or remove the placeholder scn_angle_mask and unify with the proteins.py implementation to avoid silently producing NaN/zero angle masks.

### [LOW] s2c4l0-symmetry-utils-unverified-indices — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/protein_utils/symmetry_utils.py:29-53`
- **Evidence:** get_symmetric_atom_pairs hardcodes SidechainNet atom-index pairs per residue with the authors' own repeated 'check indices' comments and a TODO stating the mapping may be wrong (e.g. F/Y use (6,10),(7,9); H uses (6,9),(7,8)). If used by rename_symmetric_atoms (ml_utils/atom_utils.py) these unverified indices would swap the wrong atoms. (Protein-path utility; relevance to the RNA pipeline is unverified.)
- **Recommendation:** Verify the indices against SC_BUILD_INFO atom-name ordering and derive pairs from atom names instead of hardcoded integers.

### [LOW] s2c4l2-026 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/protein_utils/symmetry_utils.py:29-54`
- **Evidence:** get_symmetric_atom_pairs uses hardcoded SidechainNet atom indices for symmetric pairs with explicit uncertainty markers: TODO 'these indices ... seem hardcoded ... Verify if this mapping is robust' (:29-31) and repeated '- check indices' comments on F/Y/R/H/V/L (:37-47), plus a note that prior (4,5) pairs were removed as 'seems incorrect' (:50-51). Correctness of the symmetric-atom renaming is unverified. This is also a second divergent implementation versus ml_utils/atom_utils.py:514 get_symmetric_atom_pairs (which uses AMBIGUOUS).
- **Recommendation:** Verify the index mapping against the SidechainNet atom ordering (or derive indices from atom names dynamically) and unify with the atom_utils implementation.

### [LOW] s2c4l0-rna-nan-check-preassign — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:336-339`
- **Evidence:** The 'universal NaN check after any atom placement' tests torch.isnan(full_coords[i, idx, :]) at line 336, but full_coords[i, idx] still holds its zero-initialized value because the computed `pos` is only written into full_coords at line 339 (after the check). The guard therefore always inspects zeros and never detects a NaN produced in `pos`. (A final nan_to_num at line 349 mitigates downstream impact.)
- **Recommendation:** Check torch.isnan(pos).any() before assigning, or move the NaN check after the full_coords[i, idx, :] = pos assignment.

### [LOW] s2c4l2-019 — doc_drift
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_constants.py:30-36`
- **Evidence:** The inline atom-quadruplet comments for each torsion are shifted by one relative to standard RNA backbone definitions: alpha is annotated 'P-O5'-C5'-C4'' (the beta atoms), beta as 'O5'-C5'-C4'-C3'' (gamma's), gamma as 'C5'-C4'-C3'-O3'' (delta's), delta as 'C4'-C3'-O3'-P' (epsilon's), epsilon as 'C3'-O3'-P-O5'' (zeta's), zeta as 'O3'-P-O5'-C5'' (alpha's). final_kb_rna.py:185-191 carries the correct labels. The numeric values appear correct; only the comments are mislabeled.
- **Recommendation:** Correct the atom-quadruplet comments to match the canonical alpha..zeta definitions (as in final_kb_rna.py).

### [LOW] s2c5l2-001 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:17`
- **Evidence:** Module-level `print(f"!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED FROM: {__file__} !!!!!!!!!!")` executes on every import. Further unconditional `print` debug banners at line 180 (build_rna_chain_from_internal_coords completion) and line 247 (ring_closure_refinement). This is leftover 'CASCADE' debug instrumentation that pollutes stdout of the README-documented Stage C reconstruction path regardless of any debug flag.
- **Recommendation:** Remove the module-level and function-level print() banners or gate them behind the existing debug_logging flag / logger.debug.

### [LOW] s2c5l0-003 — other
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:17,180,247`
- **Evidence:** Unconditional module-level `print("!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED ...")` at import (line 17), a per-call `print("!!! CASCADE: build_rna_chain_from_internal_coords COMPLETED ...")` (line 180), and a CASCADE print inside ring_closure_refinement (line 247). These execute on every import/call regardless of any debug flag, polluting stdout in production/inference runs.
- **Recommendation:** Remove the stray debug prints or gate them behind a debug_logging flag / logger.debug.

### [LOW] s2c5l0-002 — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:177-181`
- **Evidence:** After building coordinates, `if torch.isnan(residue_coords).any(): logger.error(...)` logs an error but does NOT raise; the function then returns the NaN-containing tensor (line 181). Callers (stage_c_reconstruction.run_stageC_rna_mpnerf) receive NaN coordinates silently, which then flow into place_bases and downstream Stage D.
- **Recommendation:** Either raise on NaN detection or return an explicit validity flag so callers do not silently propagate NaN coordinates.

### [LOW] s2c5l2-005 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_scaffolding.py:56-93`
- **Evidence:** backbone_triplets are built with `range(len(RNA_CONNECT['backbone']) - 2)` (line 56), dropping the final triplet, and dihedral torsions are written to fixed hard-coded slots angles_mask[1,i,1..6] and angles_mask[1,i,9] (lines 87-93), skipping indices 7 and 8 with no documented rationale. This fixed slot mapping is fragile and unrelated to the get_torsion_angle_index mapping used in rna_folding.py:130, so the two modules encode torsion ordering differently.
- **Recommendation:** Document and unify the torsion-index mapping between scaffolding and folding; replace magic indices with named constants and verify the -2 triplet bound is intentional.

### [LOW] s2c5l0-004 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_scaffolding.py:95-112`
- **Evidence:** point_ref_mask is filled for non-P atoms as `[i*B + (j-3), i*B + (j-2), i*B + (j-1)]` (lines 110-112). For j==1 and j==2 this yields negative within-residue offsets (j-3 = -2 and -1; j-2 = -1 and 0), producing reference indices that point into the previous residue's atoms or to negative positions. The masks (bond_mask, point_ref_mask) are also not consumed by the actual folding path (rna_folding.build_rna_chain_from_internal_coords reads only scaffolds['torsions']), so this construction is both incorrect and dead for the mp_nerf path.
- **Recommendation:** Either remove the unused bond_mask/point_ref_mask construction or fix the j<3 reference indices (clamp/guard the negative offsets) if a future code path will consume them.

### [LOW] s2c5l2-008 — design_defect
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/utils.py:10-31`
- **Evidence:** Bio.PDB is imported twice: first unconditionally via importlib.util.find_spec guard (lines 11-15) and again inside a try/except that defines the BIOPYTHON_AVAILABLE flag and dummy classes (lines 23-31). The first import is redundant and the commented-out import block (lines 17-20, 26) is leftover noise.
- **Recommendation:** Keep only the try/except import that sets BIOPYTHON_AVAILABLE; remove the redundant find_spec import and commented blocks.

### [LOW] s2c5l0-006 — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/utils.py:71-72`
- **Evidence:** to_zero_two_pi returns `torch.where(x > np.pi, x % np.pi, 2*np.pi + x % np.pi)`. For x in [0, pi] (the not-greater branch) it returns 2*pi + (x % pi), i.e. values >= 2*pi, which cannot be a wrapped angle in [0, 2*pi). The intended wrap to [0, 2*pi) is not produced. (Vendored utility; verify usage before relying on it.)
- **Recommendation:** Correct the angle-wrap formula (e.g., `x % (2*np.pi)`), or remove if unused.

### [LOW] s2c5l2-007 — doc_drift
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/utils.py:76-136`
- **Evidence:** get_prot pulls a PROTEIN from sidechainnet (vocab int2char, padding token == 20, coords stride * 14, angles) — protein-specific code vendored from the original mp_nerf library (file header 'Author: Eric Alcaide') sitting in an RNA pipeline. It also has an unreachable `return None` at line 136 after a `while True:` loop. This is dead/irrelevant code relative to the RNA structure-prediction intent.
- **Recommendation:** Remove get_prot (and other protein-only helpers) or clearly quarantine/document them as unused vendored code; delete the unreachable return.

### [LOW] s2c5l0-005 — bug
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/utils.py:88-136`
- **Evidence:** get_prot wraps its logic in `while True:` (line 88) with the only exit being a `return` inside the loop; the trailing `return None` at line 136 is unreachable. If the dataloader yields no matching protein the function spins forever rather than terminating. Vendored sidechainnet helper (protein-oriented) embedded in the RNA mp_nerf tree.
- **Recommendation:** Add a termination condition (e.g., StopIteration handling / max attempts) so the loop cannot spin indefinitely, and drop or make the trailing return reachable.

### [LOW] s2c5l2-011 — doc_drift
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:53-63,105-110`
- **Evidence:** The class docstring states Stage C 'intentionally produces a sparse atom representation (21 atoms total)' and that 'Stage D expects a dense atom representation (44 atoms per residue)'. But __call__ returns coords of shape (N*3,3) i.e. 3 atoms/residue (not 21 total), and the real MP-NeRF path returns per-residue STANDARD_RNA_ATOMS counts. The '21 atoms total' and '44 per residue' figures are not reflected by either code path here (compute_max_rna_atoms returns 21 as a per-residue max, not a total), making the docstring misleading.
- **Recommendation:** Update the class docstring to reflect the actual per-residue atom counts produced by each method and the bridging target, or remove the stale numeric claims.

### [LOW] s2c5l2-012 — design_defect
- **Location:** `rna_predict/pipeline/stageC/stage_c_reconstruction.py:72-76,256-258,403-408`
- **Evidence:** Several INFO-level logs are emitted unconditionally (independent of debug_logging): per-instance memory logs in __init__ (lines 72-76), '[DEBUG][StageC] stage_cfg.device ...' emitted via logger.info at lines 256-258 (debug content at INFO level), and a duplicate summary forced onto the ROOT logger at line 408 ('ROOT: StageC completed ...'). These 'SYSTEMATIC DEBUGGING' artifacts spam the documented inference path's logs.
- **Recommendation:** Demote these to logger.debug gated by debug_logging and remove the explicit root-logger emission.

### [LOW] s2c5l0-029 — design_defect
- **Location:** `rna_predict/pipeline/stageD/config.py:108-162`
- **Evidence:** stageD/config.py defines a dataclass `DiffusionConfig` (a Hydra schema with mode/device/model_architecture-style fields) and registers it in the ConfigStore at import (cs.store at lines 149-162), while a completely different `DiffusionConfig` dataclass (a runtime input container with partial_coords/trunk_embeddings) lives in diffusion/utils/config_types.py and is what diffusion/config.py and utils/__init__.py re-export. Two unrelated classes share the name DiffusionConfig, inviting import/type confusion. ConfigStore.store also runs as an import side effect.
- **Recommendation:** Rename one of the DiffusionConfig classes (e.g., DiffusionSchemaConfig vs DiffusionRunConfig) to disambiguate, and move ConfigStore registration into an explicit register function rather than executing at import time.

### [LOW] s2c5l2-014 — design_defect
- **Location:** `rna_predict/pipeline/stageD/config.py:114`
- **Evidence:** DiffusionConfig.debug_logging defaults to True. Stage D is documented as a research prototype (README per Stage-1 intent), and the codebase has extensive debug_logging-gated prints/logs; defaulting this True makes verbose diagnostic output the out-of-the-box behavior for every Stage D run.
- **Recommendation:** Default debug_logging to False to match the other stages and avoid shipping verbose diagnostics by default.

### [LOW] s2c5l2-015 — design_defect
- **Location:** `rna_predict/pipeline/stageD/config.py:18,117`
- **Evidence:** sigma_data is defined inconsistently: NoiseScheduleConfig.sigma_data=16.0 (line 18) vs DiffusionConfig.sigma_data=1.0 (line 117); generator.py defaults sigma_data=16.0 with comment 'in EDM, this is 1.0' (generator.py:37,88). Three different sigma_data sources with two different values create ambiguity about the actual data scale used by the EDM noise process.
- **Recommendation:** Consolidate sigma_data to a single config field consumed by both the noise scheduler and the diffusion module.

### [LOW] s2c5l0-022 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/bridging/residue_atom_bridge.py:504-522`
- **Evidence:** _process_one_trunk_embedding, after failing to find feature_dimensions in config, falls back to hardcoded expected_dim values (s_trunk=384, s_inputs=449, sing=384) with only a warning (lines 507-515), then adjusts tensor feature dims to those constants. This contradicts the module/project's stated 'strictly config-driven, no hardcoded fallbacks' intent and can silently coerce tensors to the wrong dimension when config is misconfigured.
- **Recommendation:** Raise the ValueError for missing feature_dimensions in all cases (as already done in the else branch at line 517-522) instead of substituting hardcoded 384/449 values.

### [LOW] s2c5l2-041 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/bridging/sequence_utils.py:59`
- **Evidence:** _try_extract_from_input_features unconditionally executes `print(f"[CASCADE-DEBUG][SEQ-EXTRACT] type={type(result)}, value={result}")` (line 59) on every successful sequence extraction, with no debug flag gating. This prints the full sequence to stdout during normal Stage D bridging.
- **Recommendation:** Remove the print or gate it behind a debug flag / logger.debug.

### [LOW] s2c5l2-025 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py:139,221,236,239,269,272`
- **Evidence:** Several print() calls in the conditioning forward path are not gated by debug_logging: e.g. '[WARNING] Unexpected batch size mismatch...' (line 139) and the '[DIFFUSION-FIX]' bridging prints (lines 221,236,239,269,272). These emit to stdout on normal shape-mismatch handling during every inference, mixing diagnostics into the output stream.
- **Recommendation:** Route through self.logger and gate behind self.debug_logging.

### [LOW] s2c5l0-015 — bug
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:124-132`
- **Evidence:** In the `is_test and 'test_init_with_basic_config' in current_test` early-init branch, when 'transformer' is not in kwargs the code calls `self.logger.debug(...)` (line 128). self.logger is only assigned later at line 259, after this branch returns at line 132, so this path raises AttributeError: 'DiffusionModule' object has no attribute 'logger'. Latent crash in the test-special-case branch.
- **Recommendation:** Use the module-level logger (defined at line 84) instead of self.logger here, or set self.logger before this branch; better, remove the test-special-case branch entirely (see s2c5l0-014).

### [LOW] s2c5l2-028 — bug
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:124-132,258-259`
- **Evidence:** In the test_init_with_basic_config early-return branch, when 'transformer' is not in kwargs the code calls `self.logger.debug(...)` (line 128), but self.logger is not assigned until line 259 (after this branch returns at line 132). This raises AttributeError ('DiffusionModule' object has no attribute 'logger') on that path.
- **Recommendation:** Use the module-level logger (defined line 84) instead of self.logger before it is set, or set self.logger before any branch — and ideally remove the test branch per s2c5l2-027.

### [LOW] s2c5l2-029 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:70-81`
- **Evidence:** DiffusionModule.__init__ unconditionally prints debug instrumentation on every instantiation: type(cfg), cfg.keys(), cfg.model_architecture, and kwargs (lines 70-81), independent of debug_logging. This spams stdout each time the diffusion model is built.
- **Recommendation:** Gate these prints behind debug_logging via the logger or remove them.

### [LOW] s2c5l1-stageD-diffmodule-envvar-init-002 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:99-132`
- **Evidence:** DiffusionModule.__init__ inspects os.environ.get('PYTEST_CURRENT_TEST') (:100) and, when it contains 'test_init_with_basic_config' (:103), executes an alternate initialization path that sets only self.c_atom/self.c_z/self.transformer from kwargs and RETURNS EARLY at :132, skipping construction of the conditioning module, encoder, transformer and decoder. The same env-var-driven special-casing also appears in protenix_diffusion_manager.py:144-153,187-213 and diffusion_module forward()/_compute_loss() (inspect-based caller-name checks at :952-956,1023-1028). Behavior of a core model component is therefore determined by an ambient, externally settable environment variable rather than by configuration.
- **Recommendation:** Eliminate environment-variable and caller-frame (inspect) based branching from model construction and forward passes. Use explicit, config-driven flags so the instantiated module is deterministic and independent of the runtime environment.

### [LOW] s2c5l0-027 — bug
- **Location:** `rna_predict/pipeline/stageD/diffusion/context_objects.py:214-228`
- **Evidence:** EmbeddingContext.get_z_trunk reads the fallback pair dimension as `c_z_dim = self.stage_cfg['model_architecture']['c_z']` (line 224) via subscript, whereas the sibling get_s_inputs (lines 166-174) reads c_s_inputs through attribute/feature_dimensions paths with graceful fallbacks. If stage_cfg lacks a 'model_architecture' key (the manager passes cfg.model.stageD.diffusion as stage_cfg), this raises KeyError when the 'pair' embedding is missing, instead of a clean fallback or clear config error.
- **Recommendation:** Use the same robust config-extraction logic as get_s_inputs (attribute/feature_dimensions/dict paths) for c_z, and raise a descriptive error if it cannot be resolved.

### [LOW] s2c5l2-021 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/diffusion.py:1-38`
- **Evidence:** diffusion.py contains only the Apache license header and comments stating that DiffusionConditioning, DiffusionModule, DiffusionSchedule and utility functions were 'moved to components/'. The file has no executable code — it is an empty stub left behind after refactoring.
- **Recommendation:** Delete the empty module (and update any references) rather than keeping a comment-only file.

### [LOW] s2c5l2-022 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/diffusion/generator.py:162-178,359-375`
- **Evidence:** Docstrings describe these RNA diffusion routines in protein terms: sample_diffusion 'Generates denoised protein structure coordinates' (line 162) and sample_diffusion_training 'Performs diffusion-based training by adding noise to ground-truth coordinates' framed around protein structure. The file header is the ByteDance/Protenix protein generator; the protein wording is stale for the RNA pipeline.
- **Recommendation:** Update docstrings to reference RNA structures, and note provenance/adaptation from Protenix.

### [LOW] s2c5l0-026 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/diffusion/inference/inference_mode.py:58-82,137-140`
- **Evidence:** A comment states 'Hydra best practice: always use config-driven value, never fallback to hardcoded default' (line 58), but the code falls back to a hardcoded `test_residues_per_batch = 25` (lines 76,79) when not found in config. seq_len is set to this value (line 82) and used in a hard `assert coords.shape[1] == atom_count` (line 137) plus a warning comparison to seq_len (line 138).
- **Recommendation:** Either remove the contradictory comment or genuinely require the config value (raise if absent); avoid the magic 25 default in production inference.

### [LOW] s2c5l0-019 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:143-153,186-213`
- **Evidence:** ProtenixDiffusionManager.__init__ contains two `if 'test_init_with_basic_config' in current_test:` blocks driven by PYTEST_CURRENT_TEST that mutate diffusion_args (copying c_atom/c_z/c_s/... from model_architecture) only in the test environment. Production initialization therefore takes a different argument-assembly path than tests.
- **Recommendation:** Always normalize diffusion_args (move the model_architecture flattening out of the test guard so production and tests share one path); remove the PYTEST_CURRENT_TEST branching.

### [LOW] s2c5l2-034 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py:308`
- **Evidence:** demo_run_diffusion hardcodes device = 'mps' (Apple Metal) and allocates all demo tensors on it (lines 308-318). On non-macOS / non-MPS machines this demo raises at tensor creation. The same hardcoded-developer-environment pattern is noted elsewhere in the repo (Stage-1: training/train.py absolute macOS path).
- **Recommendation:** Default the demo to cpu (or auto-detect) rather than hardcoding 'mps'.

### [LOW] s2c5l2-035 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py:66-73`
- **Evidence:** get_unified_cfg uses `with resources.path('rna_predict.conf', '') as cfg_path:` to obtain the Hydra config dir and passes it to hydra.initialize(config_path=str(cfg_path)). importlib.resources.path is deprecated and passing '' as the resource name plus an absolute path to Hydra's initialize (which expects a path relative to the caller) is brittle and likely to fail.
- **Recommendation:** Use initialize_config_dir with importlib.resources.files('rna_predict.conf') (or hydra's documented absolute-dir API).

### [LOW] s2c5l2-037 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/config_types.py:29,34-36`
- **Evidence:** The runtime DiffusionConfig hardcodes feature defaults test_residues_per_batch=25, ref_element_size=128, ref_atom_name_chars_size=256, profile_size=32 as plain magic numbers. These duplicate (and can drift from) the Hydra schema's input_features sizes (config.py:97-99 ref_element=[128], ref_atom_name_chars=[256], profile=[32]) without any linkage.
- **Recommendation:** Source these from the Hydra config rather than hardcoding duplicates, or document them as fallbacks tied to the schema.

### [LOW] s2c5l2-040 — design_defect
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/config_utils.py:118-133`
- **Evidence:** validate_stageD_config contains a 'Special case for tests' that, when 'model.stageD' is absent but a top-level 'stageD' key exists, rewrites the whole cfg in place via OmegaConf.update loops (lines 119-133) — mutating the caller's config to relocate stageD under model. Test-shaped configs are being silently transformed by a validation function, which is surprising and couples validation to test layouts.
- **Recommendation:** Validate without mutating the input config; have tests provide the canonical model.stageD structure.

### [LOW] s2c5l0-028 — bug
- **Location:** `rna_predict/pipeline/stageD/diffusion/utils/tensor_utils.py:72-75`
- **Evidence:** normalize_tensor_dimensions, when tensor.shape[0] != batch_size, silently truncates with `tensor = tensor[:batch_size]` after only a warning (lines 73-75). If the input batch dimension is genuinely larger (e.g., an N_sample/residue dim was misinterpreted as batch), this silently discards data rather than failing, and can hide upstream shape errors during bridging.
- **Recommendation:** Raise on an unexpected batch dimension mismatch (or expand when batch_size>shape[0]); do not silently slice rows away.

### [LOW] s2c5l2-045 — design_defect
- **Location:** `rna_predict/pipeline/stageD/memory_optimization/memory_fix.py:33-40,101-107`
- **Evidence:** preprocess_inputs silently truncates 3D coords/embeddings to max_seq_len=25 (lines 35-36, 52-57) with no warning for the 3D case (only the 2D path warns), so longer inputs lose residues without notice; and run_stageD_with_memory_fixes defaults device='cuda' (line 106), which fails on CPU-only hosts. Combined, the 'memory-efficient' wrapper can silently drop data and assume CUDA.
- **Recommendation:** Warn (or make configurable) on any truncation including 3D, and default device to 'cpu'/auto-detect.

### [LOW] s2c5l2-043 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/memory_optimization/run_stageD_memory_efficient.py:26-53`
- **Evidence:** The example dummy data and config use dims s_inputs=449, s_trunk=384, pair=64, c_token=832, c_atom=128, num_steps=100 etc. (lines 27-52). These magic numbers are inconsistent with the registered structured schema (config.py: c_s=8/384, c_z=4/128, c_s_inputs=8/32, c_token=768) and with the 'conditioning'/'manager' keys this config invents (see s2c5l2-044). The example does not match the canonical Hydra config shape.
- **Recommendation:** Build the demo config from the registered Hydra schema (or align the magic dims with it) so it documents real usage.

### [LOW] s2c5l0-021 — bug
- **Location:** `rna_predict/pipeline/stageD/memory_optimization/run_stageD_memory_efficient.py:69-73`
- **Evidence:** In train mode `x_denoised, loss, sigma = result` unpacks the return of run_stageD_with_memory_fixes -> run_stageD_diffusion -> run_training_mode, which returns `(x_denoised, sigma, x_gt_augment)` (training_mode.py:107). So the second element is sigma (mislabeled 'loss') and the third is x_gt_augment (mislabeled 'sigma'). `loss.item()` / `sigma.item()` then operate on the wrong tensors (x_gt_augment is not scalar, so sigma.item() would raise under debug_logging).
- **Recommendation:** Align the unpacking with run_training_mode's actual (x_denoised, sigma, x_gt_augment) ordering and naming.

### [LOW] s2re1-009 — bug
- **Location:** `rna_predict/pipeline/stageD/memory_optimization/test_memory.py:90`
- **Evidence:** get_config(config_path="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf") hardcodes the original author's absolute machine path, so this Stage-D memory test cannot resolve Hydra configs on any other machine or in CI. A distinct machine-specific-path site from the previously-listed compute_ground_truth_angles.py:53 and from s2re1-001 (train.py).

### [LOW] s2c5l2-019 — design_defect
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:303-307,543-546`
- **Evidence:** _run_stageD_impl returns context.result if set, otherwise falls back to returning context.diffusion_cfg (a config object) as the function's 'refined coordinates' result (lines 304-307). The caller _run_stageD_main_logic_context then logs 'Coordinates were NOT refined as a tensor (training mode or error)' (line 546). Returning a config object in the coordinate-result slot is a confusing contract that masks failures as non-tensor 'results'.
- **Recommendation:** Return None or raise on failure rather than returning the config object; make the return type explicit and consistent.

### [LOW] s2c5l1-stageD-pytest-envvar-bypass-001 — design_defect
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:319-366`
- **Evidence:** run_stageD() reads the runtime environment variable PYTEST_CURRENT_TEST (os.environ.get("PYTEST_CURRENT_TEST","") at :319-320) and, when it is set and contains one of the literal test names ('test_run_stageD_basic','test_run_stageD_with_debug_logging','test_gradient_flow_through_stageD') at :341, SHORT-CIRCUITS the entire diffusion pipeline and returns a fabricated dummy tensor {"coordinates": out} (:343-366) instead of executing _run_stageD_impl. Production control flow is thus gated on an attacker/operator-controllable environment variable: anyone able to set PYTEST_CURRENT_TEST to a string containing those substrings makes the refinement stage silently return non-physical placeholder coordinates rather than real output. This is test logic embedded in a production code path.
- **Recommendation:** Remove the PYTEST_CURRENT_TEST branch from production code. Drive test-specific behavior through dependency injection / explicit constructor flags or monkeypatching in the test suite, never via an ambient environment variable that is honored at deploy time.

### [LOW] s2c5l2-020 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:392`
- **Evidence:** Stage D's @hydra.main uses config_name="default.yaml" (with the .yaml extension), whereas every other stage main uses config_name="default" (e.g. stage_c_reconstruction.py:491). Inconsistent config_name spelling across the otherwise-parallel stage entry points.
- **Recommendation:** Use config_name="default" (no extension) for consistency with the other Hydra mains.

### [LOW] s2c5l0-011 — design_defect
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:80-98`
- **Evidence:** set_stageD_logger_level sets the ROOT logger's level (`root_logger.setLevel(level)`, line 85) and mutates every handler on the root logger (lines 95-98) based on Stage D's debug_logging flag. This is invoked from run_stageD (line 330) and changes global logging verbosity for the whole process as a side effect of calling Stage D.
- **Recommendation:** Scope level changes to the Stage D package logger; do not mutate the root logger / global handlers from a stage runner.

### [LOW] s2c5l2-018 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/run_stageD.py:9-26`
- **Evidence:** The module docstring's Configuration Requirements list flat keys under model.stageD such as 'ref_element_size', 'ref_atom_name_chars_size' and 'inference: num_steps', but the actual structured schema (config.py) puts these under nested groups (input_features.ref_element.size, inference.num_steps inside DiffusionConfig.inference). The documented config shape does not match the registered schema.
- **Recommendation:** Update the docstring to reflect the real nested Hydra structure (model.stageD.diffusion.* with input_features sub-group).

### [LOW] s2c5l0-024 — perf
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/bridging_utils.py:107-122`
- **Evidence:** check_and_bridge_embeddings bridges residue-level s_inputs to atom-level with a double Python loop `for b in range(batch_size): for atom_idx in range(n_atoms):` doing per-element assignment (lines 108-119). This is O(batch*n_atoms) Python iterations per call. It also silently leaves atoms with residue_idx >= s_inputs.shape[1] as zeros (guard at line 118) without warning, which can mask mapping errors.
- **Recommendation:** Replace the loop with a vectorized gather (s_inputs[b, atom_to_token_idx]) and emit a warning when any residue_idx is out of range rather than silently zero-filling.

### [LOW] s2c5l2-046 — perf
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/bridging_utils.py:108-119`
- **Evidence:** check_and_bridge_embeddings bridges s_inputs from residue- to atom-level with a doubly-nested Python loop over batch_size and n_atoms, calling .item() per atom (lines 108-119). For realistic atom counts this is an O(B*N_atom) per-element Python loop on the inference hot path, where a vectorized index_select via atom_to_token_idx would suffice.
- **Recommendation:** Replace the per-atom Python loop with a vectorized gather (e.g. s_inputs[b].index_select(0, atom_to_token_idx)).

### [LOW] s2c6l2-002 — doc_drift
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/config_utils.py:15-20`
- **Evidence:** flatten_stageD_config_to_dict docstring states it flattens the config '...omitting tensor fields' but the body is a single OmegaConf.to_container(stage_cfg, resolve=True) call (:20) that omits nothing; the comment at :19 ('Exclude tensor fields') is never implemented.
- **Recommendation:** Either implement tensor-field exclusion or correct the docstring/comment to match the actual behavior.

### [LOW] s2c6l2-001 — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/config_utils.py:8,36`
- **Evidence:** Two near-identically-named stageD config validators with divergent behavior coexist: public validate_and_extract_stageD_config (:8) only checks cfg.model.stageD, while private _validate_and_extract_stageD_config (:36) also accepts a top-level cfg.stageD fallback (_extract_stageD :45) and silently injects defaults via setattr (_validate_required_stageD_params :70-73). Callers picking one vs the other get different validation semantics.
- **Recommendation:** Consolidate into one validator; make the default-injection explicit/opt-in and fail loudly on truly-missing required config rather than mutating stage_cfg with hardcoded defaults.

### [LOW] s2c6l0-featutils-profile-shape — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:145,343-346`
- **Evidence:** Within the same module 'profile' is created with inconsistent leveling: _init_feature_tensors builds profile as [batch, num_atoms, profile_dim] (line 145), while initialize_features_from_config builds profile as [batch, num_residues, profile_size] (line 344). Downstream extract_atom_features (lines 388-405) enforces all features share the same atom count; a residue-level 'profile' would fail that check, so the two builders are not interchangeable.
- **Recommendation:** Decide whether 'profile' is residue- or atom-level and make both builders consistent (and consistent with extract_atom_features' expectations).

### [LOW] s2c6l2-005 — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/validation_utils.py:31`
- **Evidence:** validate_run_stageD_inputs computes `_ = n_atoms // n_residues if n_residues else None` and discards the result; the value is never used. Dead computation that suggests an intended atoms-per-residue check was never completed.
- **Recommendation:** Remove the dead statement or finish the intended validation it was meant to support.

### [LOW] s2c6l1-tensorfixes-global-monkeypatch — design_defect
- **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py:251-293,347-367,370-386 (invoked via run_stageD_unified.py:137)`
- **Evidence:** apply_tensor_fixes and the *_fixes modules monkeypatch global PyTorch primitives process-wide: torch.Tensor.__add__ (tensor_operations.py:38, attention_fixes/__init__.py), torch.matmul/torch.bmm/torch.nn.functional.linear (tensor_operations.py:106-108), torch.nn.Linear.forward (attention_fixes.py / tensor_fixes/__init__.py:367) and torch.nn.Module.forward (tensor_fixes/__init__.py:386). On a RuntimeError these silently slice/truncate or even return one operand unchanged (tensor_operations.py:29-32), masking shape-correctness failures rather than surfacing them. Not an injection vector, but it is a global integrity/correctness hazard that can silently corrupt model outputs for any code in the process.
- **Recommendation:** Avoid patching global torch operators; fix shape handling at call sites or in dedicated wrapper modules. If retained, gate behind an explicit opt-in flag and fail loudly instead of returning silently coerced results.

### [LOW] s2c6l2-018 — design_defect
- **Location:** `rna_predict/predict.py:326-377`
- **Evidence:** RNAPredictor.predict_submission_original is a full alternate implementation superseded by predict_submission (:219). A full-repo review shows batch_predict (:413) and predict_submission are the live path; predict_submission_original is not called from production code.
- **Recommendation:** Remove the dead predict_submission_original method (or document it as legacy and exclude it from the public surface).

### [LOW] s2c6l0-predict-orig-index — bug
- **Location:** `rna_predict/predict.py:357-366`
- **Evidence:** predict_submission_original builds resname via [sequence[i] for i in residue_indices]; residue_indices falls back to list(range(n_atoms)) (line 358) when metadata is absent. If n_atoms > len(sequence) (typical: many atoms per residue), sequence[i] raises IndexError. Even with metadata, indices are not bounds-checked against len(sequence).
- **Recommendation:** Bound-check residue_indices against len(sequence) or map atoms to residues correctly before indexing the sequence string.

### [LOW] s2c6l2-020 — design_defect
- **Location:** `rna_predict/predict.py:380-395`
- **Evidence:** load_partial_checkpoint duplicates the partial-state-dict logic already provided by rna_predict/utils/checkpoint.py:partial_load_state_dict, with divergent semantics (here it pre-filters by matching shape and uses load_state_dict(strict=False); the util uses per-key copy_ with try/except). Two checkpoint loaders risk inconsistent behavior.
- **Recommendation:** Use the shared utils/checkpoint.partial_load_state_dict in predict.py to keep one checkpoint-loading policy.

### [LOW] s2c6l1-predict-seqpath-fileread — security
- **Location:** `rna_predict/predict.py:485`
- **Evidence:** main() reads arbitrary filesystem paths taken from the 'sequence_path' column of the input CSV: `with open(seq_path, 'r') as f: lines = f.readlines()` (predict.py:483-486). The path is not constrained to a base directory, so a crafted input_csv can make the process open any file the user can read (limited impact: only A/C/G/U lines are kept and the path is echoed to logs on error at predict.py:514).
- **Recommendation:** Resolve seq_path against a configured input root and reject paths that escape it (realpath containment check); validate the column values before opening.

### [LOW] s2c6l2-021 — design_defect
- **Location:** `rna_predict/predict.py:79,97-99,412`
- **Evidence:** The primary, README-recommended inference path prints raw debug to stdout on every prediction (e.g. predict_3d_structure :79,:97-99 and batch_predict :412), polluting CLI/notebook output for end users.
- **Recommendation:** Route debug output through logger.debug guarded by a config flag rather than unconditional print().

### [LOW] s2c6l2-026 — design_defect
- **Location:** `rna_predict/runners/batch_runner.py:111`
- **Evidence:** batch_runner's additional_files list includes runners/full_pipeline.py (:111) and runs it via `uv run <file>` (run_python_file :27). Per the Stage-1 inventory, full_pipeline.py is a library module with no __main__/CLI; executing it as a script does nothing useful (it only triggers import side effects such as the global logging.basicConfig) and produces no pipeline output in the combined log.
- **Recommendation:** Remove full_pipeline.py from the batch list or invoke runners/pipeline_cli.py (the actual full-pipeline entry point) instead.

### [LOW] s2c6l2-027 — doc_drift
- **Location:** `rna_predict/runners/demo_entry.py:1-6`
- **Evidence:** The module docstring opens 'main.py - Entry point for RNA_PREDICT package.' but the file is runners/demo_entry.py (Stage-1 notes it is byte-for-byte identical to rna_predict/main.py). The header was copy-pasted and no longer names the actual file.
- **Recommendation:** Update the docstring to reference demo_entry.py and its demo purpose.

### [LOW] s2c6l2-028 — intent_mismatch
- **Location:** `rna_predict/runners/demo_entry.py:34-40`
- **Evidence:** demo_run_input_embedding() prints 'Now streaming the bprna-spot dataset...' and 'Showing the full dataset structure for the first row...' (:38-39) and returns True, but performs no dataset streaming or input-embedding work. The output claims behavior that does not occur, and the function docstring calls it 'a simple demonstration of the input embedding functionality'.
- **Recommendation:** Either implement the demonstrated behavior or change the messages/docstring to make clear this is an empty stub.

### [LOW] s2c6l2-025 — design_defect
- **Location:** `rna_predict/runners/full_pipeline.py:399-414,479-490`
- **Evidence:** There are two Stage-D guard blocks. The first (:401-414) only logs and creates a default atom_metadata then `pass`es without calling Stage D; the actual Stage D invocation and an identical atom_metadata default block are repeated later (:480-499). The first block's work is largely dead/duplicated.
- **Recommendation:** Remove the first redundant run_stageD block; keep a single atom_metadata-default + Stage-D-call path.

### [LOW] s2c6l2-030 — design_defect
- **Location:** `rna_predict/runners/pipeline_cli.py:9-12`
- **Evidence:** pipeline_cli registers RNAConfig in the ConfigStore under name 'default' (:10) while also using config_name='default' with config_path='conf', where runners/conf/default.yaml also exists. Registering a structured config under the same primary name as a config file creates an ambiguous/conflicting Hydra composition for the entry point.
- **Recommendation:** Register the schema under a distinct name (e.g. 'rna_predict_config') and reference it from a defaults list, instead of colliding with the 'default' config file name.

### [LOW] s2c6l2-032 — design_defect
- **Location:** `rna_predict/scripts/hypot_test_gen.py:8-39`
- **Evidence:** remove_logger_lines and fix_leading_zeros are duplicated across at least three modules per the Stage-1 inventory: rna_predict/scripts/hypot_test_gen.py, scripts/automation/hypot_test_gen.py, and scripts/test_utils/hypot_test_gen.py (plus tests/common/mock_hypot_test_gen.py). Divergent copies of the same helper risk drift.
- **Recommendation:** Consolidate to one canonical implementation and import it everywhere.

### [LOW] s2c6l0-lm-lr-hardcoded — design_defect
- **Location:** `rna_predict/training/rna_lightning_module.py:1008-1012`
- **Evidence:** configure_optimizers hardcodes Adam(lr=1e-3) and ignores any learning rate / optimizer settings in cfg, contradicting the otherwise config-driven design and preventing hyperparameter control during the 'Experimental' training mode.
- **Recommendation:** Read optimizer type and lr from cfg.training with sensible defaults.

### [LOW] s2c6l2-038 — design_defect
- **Location:** `rna_predict/training/rna_lightning_module.py:368,374-377,459,480,540-541,585-598`
- **Evidence:** The training forward/training_step paths emit many unconditional print() statements (e.g. '[DEVICE-PATCH]' :368,:377; '[DEBUG][FORWARD]' :374,:459; '[NOISE-PRINT]' :480; '[TRAIN DEBUG]' :540-541; per-parameter grad-norm prints :585-598), independent of debug_logging, producing heavy stdout noise during normal training.
- **Recommendation:** Replace print() with logger.debug gated on self.debug_logging; remove the per-step backward()/grad-norm diagnostic from the hot path.

### [LOW] s2c6l0-angleloss-norm — bug
- **Location:** `rna_predict/utils/angle_loss.py:29-32`
- **Evidence:** With a mask, loss = (per-element MSE over [B,L,num_angles]) summed and divided by mask.sum()+1e-8. mask.sum() counts valid (B,L) positions only, but the numerator sums over the num_angles feature axis too, so the mean is inflated by a factor of num_angles versus the true per-element mean. (The Lightning training_step at rna_lightning_module.py:573 multiplies the denominator by the feature count, so the two loss implementations disagree.)
- **Recommendation:** Divide by (mask.sum() * num_angles) + eps, or expand the mask and divide by the masked element count, to get a true per-element mean.

### [LOW] s2c6l2-039 — design_defect
- **Location:** `rna_predict/utils/angle_loss.py:5-35`
- **Evidence:** angle_loss is referenced only by tests/utils/test_angle_loss.py (full-repo grep for imports/usage). The production training loss is reimplemented inline in rna_lightning_module.training_step (MSE + mask, :564-576) rather than calling this utility, so the two angle-loss definitions can diverge and the util is unused in production.
- **Recommendation:** Either use angle_loss() in training_step or remove it; keep one angle-loss definition.

### [LOW] s2c6l2-040 — design_defect
- **Location:** `rna_predict/utils/checkpointing.py:17-32`
- **Evidence:** save_trainable_checkpoint prints the entire model state_dict key list and all named_parameters to stdout on every save (:17-23,:30-32). For real models this dumps thousands of lines per checkpoint save.
- **Recommendation:** Demote these dumps to logger.debug or remove them.

### [LOW] s2c6l0-devmgmt-struct-mutate — design_defect
- **Location:** `rna_predict/utils/device_management.py:118-124`
- **Evidence:** handle_device_error mutates the Hydra config in place (cfg.device_management.force_components_to_cpu = [] / .append(component_path)). If cfg is a struct-flagged OmegaConf DictConfig (the norm for structured configs), adding a new key raises ConfigAttributeError; and the comment 'for future runs' is misleading since the config object is not persisted across process runs.
- **Recommendation:** Avoid mutating the config; track forced-CPU components in a local/runtime structure, or OmegaConf.set_struct(cfg, False) deliberately with documentation.

### [LOW] s2c6l2-041 — design_defect
- **Location:** `rna_predict/utils/device_management.py:15-127`
- **Evidence:** device_management's get_device_for_component/handle_device_error are used only by rna_predict/pipeline/stageB/torsion/torsionbert_inference.py (full-repo grep). The central orchestrators ignore this module: runners/full_pipeline.py and training/rna_lightning_module.py do ad-hoc cfg.device handling and define their own move_to_device (rna_lightning_module.py:1014), duplicating logic and leaving the conf/device_management config group largely unconsumed.
- **Recommendation:** Standardize device selection/movement on this module across stages, or remove it and its config group if not adopted.

### [LOW] s2c6l2-042 — bug
- **Location:** `rna_predict/utils/rna_backbone_extraction.py:57`
- **Evidence:** extract_pdb_backbone_coords contains the statement `line[17:20].strip()` (:57) which slices the residue-name field but discards the result (no assignment, no side effect). Dead code indicating an intended residue-name capture that was dropped.
- **Recommendation:** Remove the dead expression or assign and use the residue name.

### [LOW] s2c6l0-backbone-cif-index — bug
- **Location:** `rna_predict/utils/rna_backbone_extraction.py:57,84,91-106`
- **Evidence:** extract_cif_backbone_coords detects the atom_site loop via lines[lines.index(line)+1] (line 84) — lines.index returns the FIRST matching line (wrong for repeated lines) and can IndexError if 'loop_' is the last line — and then parses fixed column positions fields[2]/[4]/[5]/[6..8], assuming a specific mmCIF column order that is not guaranteed. Also line 57 (`line[17:20].strip()`) in the PDB parser is a dead expression whose result is discarded (residue name never captured).
- **Recommendation:** Parse the _atom_site loop header to map column names to indices instead of assuming positions; iterate with enumerate instead of lines.index; remove or use the dead line[17:20] read.

### [LOW] s2c6l2-044 — design_defect
- **Location:** `rna_predict/utils/scatter_utils.py:16-20`
- **Evidence:** layernorm() returns torch.zeros_like(x) whenever the last dim is 1, with the inline comment 'Return zeros to ensure zero mean (test will pass the mean check)' (:19). The behavior is admittedly coded to satisfy a test rather than from a numerically-motivated convention, and silently discards the input for dim-1 features.
- **Recommendation:** Define a principled dim-1 behavior (e.g. return input unchanged or raise) and decouple it from test expectations.

### [LOW] s2c6l2-045 — design_defect
- **Location:** `rna_predict/utils/scatter_utils.py:57-69,68-69,83-86`
- **Evidence:** scatter_mean silently grows dim_size to max(index)+1 (:58-60) and clamps indices into range (:65), masking caller errors (e.g. wrong segment count) instead of failing. It also prints index contents and dim_size on every call (:68-69, fallback :83-86), noisy in a hot per-token path.
- **Recommendation:** Validate index range against dim_size and raise on violation; remove or gate the per-call prints behind a debug flag.

### [LOW] s2c6l2-047 — design_defect
- **Location:** `rna_predict/utils/shape_utils.py:16-125`
- **Evidence:** adjust_tensor_feature_dim (:40-55) and adjust_attention_bias (:92-117) silently zero-pad or slice tensors to coerce shapes, the same defect-masking pattern as the stageD tensor_fixes; mismatches are hidden rather than surfaced.
- **Recommendation:** Restrict silent coercion to known-safe cases and log/assert otherwise so genuine shape bugs are not hidden.

### [LOW] s2c6l2-048 — design_defect
- **Location:** `rna_predict/utils/submission.py:99-103`
- **Evidence:** coords_to_df copies the identical x/y/z values into every repeat column x_1..x_n,y_1..,z_1.. (:100-103), so all 'prediction_repeats' are byte-identical duplicates rather than distinct predictions. (This helper backs the dead predict_submission_original path, predict.py:331.)
- **Recommendation:** Accept per-repeat coordinates (or document that this helper intentionally duplicates a single prediction across columns).

### [LOW] s2c6l0-resmap-meta-seqlen — bug
- **Location:** `rna_predict/utils/tensor_utils/residue_mapping.py:399-412,145-147`
- **Evidence:** In derive_residue_atom_map Method 1, n_residues_meta is computed from max(residue_indices)+1 and passed to _derive_map_from_metadata, which builds a map of length n_residues_meta and then calls _validate_residue_atom_map iterating that length while indexing sequence_list[res_idx] (line 147). If the provided sequence is shorter than n_residues_meta (metadata implies more residues than the sequence), this raises IndexError in the warning path.
- **Recommendation:** Reconcile n_residues_meta with len(sequence_list) (raise a clear error on mismatch) and guard sequence_list indexing in _validate_residue_atom_map.

### [LOW] s2c6l2-053 — bug
- **Location:** `scripts/analysis/analyze_code.sh:274-282,315,320-324`
- **Evidence:** coverage is installed into the uv-managed environment (`uv run pip install coverage`, :277) and other tools are invoked via `uv run`, but the coverage commands are called as bare `coverage run/report/xml/json` (:315,:320-324) and availability is checked with `command -v coverage` (:305) against the system PATH. If coverage lives only in the uv env, these bare invocations fail (the script tolerates it via `|| true`, silently skipping coverage).
- **Recommendation:** Invoke coverage consistently via `uv run coverage ...` (and check with `uv run coverage --version`).

### [LOW] s2c6l0-analyze-undefvar — bug
- **Location:** `scripts/analysis/analyze_code.sh:404,420`
- **Evidence:** Coverage reporting references $TARGET_BASENAME (lines 404 and 420), but that variable is never assigned anywhere in the script (the defined variable is BASENAME at line 288). Under `set -u` it would error; as-is it expands to empty, producing misleading messages like 'Coverage for  is ...'.
- **Recommendation:** Use the defined $BASENAME (or define TARGET_BASENAME) in the coverage messages.

### [LOW] s2c7l0-012 — bug
- **Location:** `scripts/automation/batch_test_generator.py:31-43`
- **Evidence:** main() is defined (lines 31-42) and prints a usage/processing flow, but there is no `if __name__ == '__main__': main()` guard, so executing `python batch_test_generator.py <folder>` does nothing. Additionally run_test_generation (lines 28-29) is a `pass` stub returning None, so in process_folder `result` is always falsy and it always prints 'Failed to generate tests for {file}' (lines 23-24). The file is labeled a stub, but the missing entrypoint makes even the stub path unreachable.
- **Recommendation:** Add the `if __name__ == '__main__': main()` guard; have run_test_generation return a truthy success indicator (or raise NotImplementedError) so the stub's control flow is meaningful.

### [LOW] s2c7l1-007 — security
- **Location:** `scripts/automation/commit_individual_files.sh:62-139`
- **Evidence:** The script iterates over data folders (RNA_NET, SPOT_RNA_PDB_dataset, bpRNA, kaggle), runs `git add "$file"` and `git commit` for every file found, then unconditionally `git push origin main`. There is no filtering/.gitignore enforcement or review step, so any secrets, credentials, or PII inadvertently present in those dataset directories would be committed and published to the remote automatically.
- **Recommendation:** Add an explicit allowlist/denylist and a dry-run/confirmation gate; never auto-push. Respect .gitignore and scan staged files for secrets before committing.

### [LOW] s2c7l2-016 — doc_drift
- **Location:** `scripts/automation/create_github_issues.py:11`
- **Evidence:** Usage docstring example references a foreign project: `python quick-fixes/automation-scripts/create_github_issues.py ImmortalDemonGod ProjectEquiSurv /Users/tomriddle1/ProjectEquiSurv/issue.json` (:11). This repo is RNA_PREDICT, the path quick-fixes/automation-scripts/ does not exist here, and ProjectEquiSurv is an unrelated repo — stale copied documentation.
- **Recommendation:** Update the usage example to this repository and a valid in-repo path.

### [LOW] s2c7l2-018 — design_defect
- **Location:** `scripts/automation/github_automation/analyze_commit_log.py:8`
- **Evidence:** LOG_FILE is hardcoded to 'all_commits.log' (:8) read from CWD, but github_automation.sh writes per-branch logs to $OUTPUT_DIR/logs/<branch>.log (github_automation.sh:246), never a combined 'all_commits.log'. The analyzer's input is thus orphaned from how logs are actually produced. Also imports Counter/defaultdict/datetime/timedelta (:2-3) that are unused.
- **Recommendation:** Accept the log path as an argv parameter (defaulting sensibly) and remove the unused imports.

### [LOW] s2c7l0-015 — bug
- **Location:** `scripts/automation/github_automation/pull_requests/convert_prs_to_markdown.py:75-77,108`
- **Evidence:** create_markdown_content does `pr['author'].get('name', ...)` etc.; GitHub returns a null author for PRs by deleted ('ghost') users, in which case pr['author'] is None and `.get` raises AttributeError, aborting conversion of that PR (no per-item try/except in main, lines 113-124). Separately, main reads `Path('pull_requests_full_pretty.json')` (line 108) from CWD, but the generator writes that file under a subdirectory `$OUTPUT_DIR/pull_requests/pull_requests_full_pretty.json` (github_automation.sh:226), so the expected input path does not match what the pipeline produces.
- **Recommendation:** Guard author access (`author = pr.get('author') or {}`); align the input path with the generator's output location (or accept it as an argument).

### [LOW] s2c7l2-003 — design_defect
- **Location:** `scripts/automation/hypot_test_gen.py:1-30`
- **Evidence:** Three divergent copies of hypot_test_gen.py exist: scripts/automation/hypot_test_gen.py and rna_predict/scripts/hypot_test_gen.py each define only the helper stubs fix_leading_zeros/remove_logger_lines, while scripts/test_utils/hypot_test_gen.py is the full 897-line generator. Likewise two batch_test_generator.py copies (automation stub vs test_utils real). The reorganize_scripts.sh `mv` (lines 10-13) was supposed to relocate these out of rna_predict/scripts/, yet rna_predict/scripts/hypot_test_gen.py still exists, leaving stale duplicates that confuse imports (see s2c7l2-001).
- **Recommendation:** Consolidate to a single canonical hypot_test_gen.py / batch_test_generator.py and remove the stale stub copies under scripts/automation/ and rna_predict/scripts/.

### [LOW] s2c7l0-013 — bug
- **Location:** `scripts/coverage/show_coverage.py:14-24`
- **Evidence:** run_command passes `check=True` only on the non-capture branch (line 17); when capture_output=True (line 15) it omits check, so subprocess.CalledProcessError is never raised on a nonzero exit and the except block at lines 19-24 (which references e.stdout/e.stderr) is dead for captured calls. Callers show_least_covered (line 268) and filter_coverage (line 346) then proceed to parse `result.stdout`, which on a failed `coverage report -m` is empty, silently yielding 'No files...' rather than surfacing the error.
- **Recommendation:** Pass check=True (and capture stderr) in both branches, or explicitly inspect result.returncode after capture and raise/report on failure.

### [LOW] s2c7l2-028 — design_defect
- **Location:** `scripts/coverage/show_coverage.py:231`
- **Evidence:** Coverage scripts disagree on the memory-profiling plugin: show_coverage.py invokes pytest with `--memprof-top-n=10 --memprof-csv-file=...` (pytest-memprof) at :231, while run_failing_tests.sh uses `--memray --most-allocations=10 --stacks=5` (pytest-memray) at run_failing_tests.sh:383,405. run_command uses check=True (:17) so an absent plugin aborts the whole run before the coverage report. The two memory tooling choices are inconsistent across the coverage toolset.
- **Recommendation:** Standardize on one memory-profiling plugin across the coverage scripts and ensure it is declared in requirements-test.txt.

### [LOW] s2c7l2-027 — design_defect
- **Location:** `scripts/coverage/show_coverage.py:54-191`
- **Evidence:** show_coverage.py contains two parallel implementations of the same report logic. The helper set parse_coverage_report (:27), filter_coverage_report (:54), get_least_covered_report (:88) and parse_missing_lines (:147) are never called; main() instead uses the inline reimplementations show_least_covered (:265) and filter_coverage (:343). The unused functions are dead/duplicate code.
- **Recommendation:** Delete the unused helper functions or refactor main()'s inline logic to call them, keeping a single implementation.

### [LOW] s2c7l2-030 — design_defect
- **Location:** `scripts/demo_stochastic_inference.py:62-68`
- **Evidence:** The uniqueness check hardcodes 5 repeats (`range(5)`, columns x_1..x_5/y_/z_ at :62-66 and the `len(unique_structs) == 5` assertion at :68), but predict_submission is called with no prediction_repeats (predict.py:219 default None → driven by config). If config sets repeats != 5, the x_5/y_5/z_5 columns won't exist (KeyError) or the success criterion is wrong. The demo silently assumes a config value it does not pass or read.
- **Recommendation:** Read the repeat count from cfg (or pass prediction_repeats=5 explicitly) and derive the column range and success threshold from that value.

### [LOW] s2re1-006 — security
- **Location:** `scripts/inspect_checkpoint.py:10; scripts/inspect_pt_file.py:10; scripts/partial_checkpoint_full_pipeline_script.py:100`
- **Evidence:** Three developer/utility scripts call torch.load(...) (map_location='cpu' / default) with no weights_only=True against arbitrary user-supplied checkpoint/.pt paths. Same unsafe-pickle deserialization class as s2re1-002 but in tooling rather than the inference path; an operator pointing these at an untrusted .pt executes embedded pickle payloads.

### [LOW] s2c7l1-003 — security
- **Location:** `scripts/partial_checkpoint_full_pipeline_script.py:100`
- **Evidence:** checkpoint = torch.load(partial_ckpt_path) is called without weights_only=True. Here partial_ckpt_path is a self-created tempfile (line 88-91), so the immediate risk is low, but the default-pickle pattern is unsafe and would be a deserialization vector if the path were ever pointed at an externally produced checkpoint.
- **Recommendation:** Add weights_only=True (or map_location and a safe loader) to torch.load to harden the pattern even for self-produced checkpoints.

### [LOW] s2re4-003 — security
- **Location:** `scripts/partial_checkpoint_full_pipeline_script.py:88`
- **Evidence:** partial_ckpt_path = tempfile.mktemp(suffix="_partial.ckpt") uses tempfile.mktemp(), deprecated since Python 2.3 due to a TOCTOU race (the returned name can be created/symlinked by another process between mktemp() and the subsequent write at line 91 / read at line 100). Should use tempfile.mkstemp()/NamedTemporaryFile.

### [LOW] s2c7l2-029 — design_defect
- **Location:** `scripts/reorganize_scripts.sh:10-13`
- **Evidence:** This one-time migration `mv`s scripts out of rna_predict/scripts/ (e.g. hypot_test_gen.py at :13) but updates none of the moved files' internal import paths or PROJECT_ROOT computations, directly causing the broken import in scripts/test_utils/batch_test_generator.py (s2c7l2-001) and the wrong PROJECT_ROOT in scripts/run_all_pipeline.py (s2c7l2-009). It also left rna_predict/scripts/hypot_test_gen.py in place (still present), so the 'move' was partial and stale duplicates remain.
- **Recommendation:** After such moves, update intra-repo imports and path assumptions, and verify rna_predict/scripts/ no longer holds duplicated relocated files; treat this script as historical (it is non-idempotent and would fail re-running).

### [LOW] s2c7l1-009 — security
- **Location:** `scripts/run_mutation_tests.sh:42-44`
- **Evidence:** CMD="mutatest -n $NUM_MUTATIONS -m $MUTATION_MODE -o $OUTPUT_FILE" is later executed via unquoted `$CMD` expansion (`if ! $CMD 2>&1 | tee ...`). NUM_MUTATIONS/MUTATION_MODE come from unvalidated -n/-m CLI args and undergo word-splitting, allowing argument injection (e.g. extra mutatest flags) into the invoked command.
- **Recommendation:** Use an array (cmd=(mutatest -n "$NUM_MUTATIONS" -m "$MUTATION_MODE" -o "$OUTPUT_FILE")) and invoke "${cmd[@]}"; validate that NUM_MUTATIONS is an integer and MUTATION_MODE is in an allowed set.

### [LOW] s2c7l0-014 — doc_drift
- **Location:** `scripts/screen_finder_app/config.py:27-58`
- **Evidence:** The documented example templates_config.json structure uses an action object keyed by `"type"` with values like 'text'/'clipboard' (e.g. `"action": {"type": "click", ...}`). However execute_action reads `action_config.get("action", "")` (scripts/screen_finder_app/main.py:29) and the real templates/templates_config.json uses `"action": {"action": "click"}`. A config authored from the docstring example (key 'type') would always fall through to 'No recognized action' (main.py:57-58). The action-type names ('text' vs 'text_input') also disagree with execute_action's branches.
- **Recommendation:** Update the config.py docstring example to use the `action` key and the action-type names actually handled by execute_action (click, text_input, clipboard, double_click), or make execute_action accept both 'type' and 'action' keys.

### [LOW] s2c7l0-016 — bug
- **Location:** `scripts/screen_finder_app/gui_launcher.py:58,61,100,142`
- **Evidence:** periodic_search_thread runs on a background daemon thread (started at gui_launcher.py:161) yet makes direct Dear PyGui calls from that thread — dpg.get_value('-INTERVAL-') (line 58) and multiple dpg.set_value('-STATUS-', ...) (lines 61,120,142) and execute_action side effects. The code's own comments (lines 46,51,251-253) acknowledge that direct DPG calls from non-main threads are unsafe and can crash/corrupt the GUI; the main render loop (lines 250-263) already polls status, making the threaded set_value calls both redundant and hazardous.
- **Recommendation:** Restrict DPG access to the main thread: have the worker thread update plain Python state (status_message, interval read once or via a thread-safe value) and let the render loop apply it, removing dpg.get_value/set_value calls from periodic_search_thread.

### [LOW] s2c7l1-008 — security
- **Location:** `scripts/screen_finder_app/main.py:25-58`
- **Evidence:** execute_action() drives pyautogui to click, double-click, type arbitrary text (action_config.get('text')), and send copy/paste hotkeys based entirely on templates_config.json (loaded in template_loader.py:30-32 with no validation of the action payload). If templates_config.json or a template image is tampered with, the tool will autonomously inject keystrokes/clicks into whatever application is focused, a local input-automation abuse vector.
- **Recommendation:** Validate and constrain action types/payloads against a strict schema, and treat templates_config.json as trusted input with restrictive file permissions; warn the user that text_input/clipboard actions execute arbitrary keystrokes.

### [LOW] s2c7l2-024 — design_defect
- **Location:** `scripts/screen_finder_app/screenshot.py:17`
- **Evidence:** The package's two real entry points (main.py:60 main(), gui_launcher.py:36 periodic_search_thread) reimplement screen capture and template matching inline (main.py:84-101, gui_launcher.py:70-85) rather than calling the dedicated helpers: capture_all_monitors (screenshot.py:17), validate_and_match_template (template_matching.py:17), and select_region (region_selector.py:80) are never invoked by either entry point. These three modules are effectively orphaned/duplicate logic (e.g. screenshot.py returns BGR while main.py needs grayscale).
- **Recommendation:** Either route main.py/gui_launcher.py through these helpers (removing the inline duplicates) or delete the unused modules to avoid divergent matching implementations.

### [LOW] s2c7l2-023 — doc_drift
- **Location:** `scripts/screen_finder_app/template_matching.py:1`
- **Evidence:** Stale path header comments survive the reorganize move: template_matching.py:1 '# rna_predict/scripts/template_matching.py', logger.py:1, region_selector.py:1, screenshot.py:1 all still claim the old rna_predict/scripts/ location, whereas reorganize_scripts.sh:37 moved screen_finder_app to scripts/screen_finder_app/.
- **Recommendation:** Update or remove the path-header comments to reflect scripts/screen_finder_app/.

### [LOW] s2c7l2-022 — bug
- **Location:** `scripts/screen_finder_app/template_matching.py:109-116`
- **Evidence:** validate_and_match_template returns a tuple `(match_location_tuple, correlation_score)` (:79), but the __main__ self-test asserts `location == (200, 100)` (:116) against that 2-tuple-of-(tuple,float). The comparison is always False, so the example's assert raises AssertionError on a successful match. The comment 'Expected location: (200, 100)' (:115) reflects the same misunderstanding of the function's own return shape.
- **Recommendation:** Assert on location[0] (the coordinate tuple), e.g. `assert location[0] == (200, 100)`.

### [LOW] s2c7l2-004 — doc_drift
- **Location:** `scripts/test_utils/hypot_test_gen.py:25`
- **Evidence:** Comment at :24 and PROMPT_TEMPLATE_FILE = Path(__file__).parent / 'prompt_template.md' (:25) assume prompt_template.md sits beside this script, but no such file exists in scripts/test_utils/ (only docs/examples/prompt_template.md exists, verified by find). load_text_prompt_template() therefore logs an error and returns '' (:34-38), so wrap_with_prompt produces an empty ''.format(...) result and the final test_wrapped_*.md is effectively blank.
- **Recommendation:** Ship a prompt_template.md alongside the script, or point PROMPT_TEMPLATE_FILE at the existing docs/examples/prompt_template.md.

### [LOW] s2c7l2-013 — doc_drift
- **Location:** `setup.py:32`
- **Evidence:** setup.py sets python_requires='>=3.8' (:32) while the toolchain targets Python 3.10 (mypy.ini per Stage-1 inventory configures Python 3.10). Mismatched declared interpreter floors between configs.
- **Recommendation:** Align python_requires with the actually supported/tested version (3.10) across setup.py and pyproject.toml.

### [LOW] s2re4-008 — design_defect
- **Location:** `test_download.py:1 and simple_test.py:1 (repo root)`
- **Evidence:** Two pytest files live at the repository root (test_download.py, simple_test.py) outside the collection scope: pytest.ini:4 sets 'testpaths = tests' and .vscode/settings.json points pytestArgs at ['tests'], so these root files are never collected/run. test_download.py is byte-for-byte identical (verified via diff, both 58 lines) to tests/test_download.py, making the root copy a stale orphan duplicate; simple_test.py is an orphan that uses unittest while the suite is pytest-based. Dead/uncollected test artifacts give a false impression of coverage and drift from the canonical copy.

### [LOW] s2c7l2-026 — design_defect
- **Location:** `tests/stageA/integration/conf/default.yaml:2`
- **Evidence:** This integration-test config nests all Stage A params under a top-level `stageA:` key (:2), whereas the canonical conf/model/stageA.yaml is deliberately flat — its comment states 'Direct stageA configuration without double nesting' (rna_predict/conf/model/stageA.yaml:5) with params at top level. The test config also omits several keys present in the canonical file (checkpoint_zip_path, debug_logging, freeze_params, run_example, example_sequence, visualization.output_path), so the test exercises a differently-shaped config than production Stage A.
- **Recommendation:** Mirror the canonical (un-nested) stageA config shape and key set in the test fixture, or document why the test intentionally uses a divergent schema.

### [LOW] s2c7l1-006 — security
- **Location:** `tests/stageA/integration/conf/default.yaml:7-8`
- **Evidence:** checkpoint_url: "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1" points Stage A at a third-party personal Dropbox archive with no integrity verification (no checksum/signature). The downloaded archive is extracted and the resulting .pth is loaded by Stage A (the download/extract/torch.load code lives outside this file set, unverified here). A compromised or swapped link is a supply-chain vector that, combined with pickle-based checkpoint loading, can lead to code execution.
- **Recommendation:** Pin and verify a SHA-256 of the downloaded archive/checkpoint before extraction/loading, and host the artifact on a controlled, versioned location rather than a personal Dropbox share.

### [INFO] s2c0l1-013 — security
- **Location:** `.env.example:1-3`
- **Evidence:** File documents required secrets ANTHROPIC_API_KEY and PERPLEXITY_API_KEY using placeholder values only ('your-api-key-here', 'pplx-abcde') — no real credentials are present. .gitignore:104 also excludes the real `.env`. This is correct handling; recorded as a positive/no-leak observation for the assigned file, not a defect.
- **Recommendation:** No action needed; continue keeping real secrets out of .env.example and ensure .env stays gitignored.

### [INFO] s2c0l0-023 — doc_drift
- **Location:** `.env.example:6`
- **Evidence:** Example env defaults reference `MODEL=claude-3-7-sonnet-20250219` and recommend `claude-3-opus-20240229`. These are stale model identifiers for the (separate) Task Master tooling and do not affect the RNA pipeline; recorded as informational drift, not a pipeline defect.
- **Recommendation:** If Task Master tooling is retained, refresh the example to current model ids; otherwise remove the unrelated env template.

### [INFO] s2c0l2-0043 — design_defect
- **Location:** `.gitignore:182 (.vscode); .idea at :181`
- **Evidence:** .gitignore ignores `.vscode` (line 183, from the Task Master block) and `.idea` (line 181), yet .vscode/settings.json is a tracked, inventoried config file (audit/01-understanding.md:70). As with package.json, the ignore rule contradicts the intent to version-control the VS Code workspace settings.
- **Recommendation:** Negate the tracked file (e.g. add `!.vscode/settings.json`) or remove the broad .vscode ignore.

### [INFO] s2c1l0-rnaconfig-docstring-misplaced — doc_drift
- **Location:** `rna_predict/conf/config_schema.py:1344-1349`
- **Evidence:** In RNAConfig the experiment_name field (lines 1345-1348) is declared before the intended class docstring string literal on line 1349 ('Root configuration for the entire RNA_PREDICT pipeline.'). Because a docstring must be the first statement, RNAConfig.__doc__ is None and the string is a no-op expression, so the class documentation is effectively lost.
- **Recommendation:** Move the docstring to be the first statement in the class body (above experiment_name).

### [INFO] s2c5l1-stageD-debug-stdout-leak-005 — security
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:17`
- **Evidence:** Module import unconditionally prints the absolute source path via print(f"!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED FROM: {__file__} !!!!!!!!!!") at :17 (and another unconditional print at :180,:247). Across the assigned Stage D files there is pervasive unconditional/print-based dumping of full resolved configs, tensor shapes, sys.path and cwd (e.g. run_stageD.py:397-399 prints CWD/SCRIPT DIR/sys.path; diffusion_module.py:70-96 prints full cfg; bridging files print tensor metadata). These leak deployment filesystem layout and configuration to stdout regardless of the debug_logging flag, an information-disclosure / log-hygiene weakness rather than an exploitable vulnerability.
- **Recommendation:** Gate all diagnostic output behind the existing debug_logging flag and the logging module (no bare print), and remove the unconditional module-level path prints so absolute paths, sys.path, and full configs are not emitted in production runs.

### [INFO] s2c5l1-mpnerf-filepath-parse-no-hardening-004 — security
- **Location:** `rna_predict/pipeline/stageC/mp_nerf/utils.py:298-321`
- **Evidence:** get_coords_from_file() dispatches on a caller-supplied file_path suffix to get_coords_from_pdb (:236-256) / get_coords_from_cif (:276-295), which feed the path directly into BioPython PDBParser/MMCIFParser.get_structure(). The path is used as-is with no canonicalization, allow-list, or size/complexity limits, and parse errors are re-wrapped with the full path echoed into the exception message (:242,:282). This is consistent with the intended purpose (loading user structure files), so there is no path-traversal escalation, but malformed/oversized structure files are an untrusted-input parsing surface with no resource bounds (DoS potential) when these helpers are exposed to externally provided files. No code-execution sink is present.
- **Recommendation:** If these loaders ever ingest untrusted/uploaded structures, add input validation (size limits, expected extension/content checks) and avoid echoing full filesystem paths in error strings surfaced to callers. Otherwise document that file_path must be a trusted, locally-controlled path.

### [INFO] s2c6l0-valutils-dead-div — design_defect
- **Location:** `rna_predict/pipeline/stageD/stage_d_utils/validation_utils.py:24-35`
- **Evidence:** validate_run_stageD_inputs computes `_ = n_atoms // n_residues if n_residues else None` (line 31) and discards it, and otherwise only raises when s_trunk is atom-level. z_trunk and s_inputs arguments are accepted but never validated, so the 'validation' is largely a no-op beyond one assertion.
- **Recommendation:** Remove the dead computation and either validate the other tensors (shape/level expectations) or trim the unused parameters.

### [INFO] s2c7l2-025 — other
- **Location:** `scripts/screen_finder_app/py.typed:1`
- **Evidence:** py.typed is an empty PEP 561 inline-types marker (0 bytes, verified via wc -c), but scripts/screen_finder_app/ is vendored tooling under scripts/ and is not a distributed/installed package (no packaging entry references it). A py.typed marker has no effect outside an installed, importable distribution, so it is inert here.
- **Recommendation:** Remove py.typed from this non-packaged script directory, or package screen_finder_app properly if inline-type advertising is intended.


## Machine-checkable object
```json
{
  "meta": {
    "rounds": 5,
    "source_total": 304,
    "examined": 305,
    "survivors": 538,
    "unverified": 0
  },
  "findings": [
    {
      "id": "s2c3l0-001",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:22",
      "class": "bug",
      "severity": "critical",
      "evidence": "FeatureProcessor (line 22), AttentionComponents (attention_components.py:16) and CoordinateProcessor (coordinate_processing.py:13) are plain Python classes (no nn.Module base) but construct nn.Module submodules (LinearNoBias, nn.Sequential small_mlp, LayerNorm). They are assigned as attributes of the nn.Module AtomAttentionEncoder/AtomAttentionDecoder (atom_attention/encoder.py:56,73 ; atom_attention/decoder.py:48,56,63). Because nn.Module.__setattr__ only registers child Modules when the assigned value is itself an nn.Module, none of the parameters inside these wrapper objects are registered. Consequently they are absent from .parameters()/.state_dict(), are NOT moved by .to(device)/.cuda(), are NOT seen by the optimizer, and are NOT saved/loaded in checkpoints.",
      "recommendation": "Make FeatureProcessor/AttentionComponents/CoordinateProcessor subclass nn.Module (call super().__init__()), or register their layers directly on the parent encoder/decoder via setup_* functions as the refactored atom_attention_encoder.py does. Add a unit test asserting len(list(encoder.parameters())) covers these layers.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-020",
      "location": "rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py:4",
      "class": "bug",
      "severity": "critical",
      "evidence": "InputFeatureEmbedder imports `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder` at module top level (line 4) and again inside __init__ (line 36, AtomEncoderConfig). The package rna_predict/models does not exist (verified: `ls rna_predict/models` -> No such file or directory; the sibling atom_encoder.py:6 even comments 'Corrected import path from models.attention to legacy.attention'). Therefore importing this legacy module raises ModuleNotFoundError immediately — the file is unimportable.",
      "recommendation": "Repoint the imports to rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder (the real location), or delete this dead legacy module if superseded by the current/ tree.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-006",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:251-293,418-427",
      "class": "design_defect",
      "severity": "critical",
      "evidence": "apply_tensor_fixes() (invoked in production at rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py:137) calls fix_tensor_add(), which globally replaces torch.Tensor.__add__ (:293) with a wrapper that, on any size-mismatch RuntimeError, silently unsqueezes/interpolates/expands operands to force the addition (lines 258-290). This mutates the semantics of '+' for every tensor in the process, masking genuine shape bugs and yielding silently-incorrect numerical results in a scientific structure-prediction pipeline.",
      "recommendation": "Remove global operator monkey-patching; fix the underlying shape mismatches at their source. If adaptation is truly needed, do it explicitly at the specific call sites with asserts, not by overriding torch.Tensor.__add__.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-tenops-add-silent",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:19-38",
      "class": "bug",
      "severity": "critical",
      "evidence": "fix_tensor_add() monkeypatches torch.Tensor.__add__ globally; on a shape-mismatch RuntimeError ('must match the size'/'at non-singleton dimension') it does NOT add the tensors at all but returns whichever operand has more dims unchanged (`return self`/`return other`, lines 28-32). The addition is silently dropped, producing numerically wrong results everywhere `+` is used after apply, with no error. This corrupts diffusion/embedding arithmetic across the whole process once installed.",
      "recommendation": "Remove this patch. Never globally override Tensor.__add__; fix the real shape mismatch at the call site. If a compatibility shim is unavoidable, raise rather than return a silently-incorrect operand.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-004",
      "location": ".github/workflows/main.yml:39",
      "class": "bug",
      "severity": "high",
      "evidence": "The 'Check for dependency vulnerabilities' step runs `pip freeze > requirements.txt`, overwriting the tracked source-of-truth requirements.txt with the full frozen CI environment. The later 'ruff auto-fix' steps (main.yml:50-70) then run `git add -A`, `git commit`, and `git push` on push-to-main events, so the clobbered requirements.txt can be committed back to the repository, corrupting the curated dependency list.",
      "recommendation": "Write the freeze to a throwaway file (e.g. `pip freeze > /tmp/frozen.txt`) and run pip-audit against that, never overwriting the repo's requirements.txt; or scope `git add` to specific paths instead of `-A`.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0006",
      "location": ".github/workflows/release.yml:38",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "release.yml builds via `python setup.py sdist bdist_wheel`, but setup.py declares `version=\"1.0.0\"` and `python_requires=\">=3.8\"` (setup.py:6,31), whereas pyproject.toml is the real metadata with `version = \"2.0.8\"` and `requires-python >= 3.10` (pyproject.toml:7,11) and rna_predict/VERSION says 2.0.8. The release pipeline would publish wheels labelled 1.0.0 with the wrong dependency set, diverging from the actual 2.0.8 package.",
      "recommendation": "Build with the PEP517 frontend (`python -m build`) so pyproject.toml metadata is used, and delete or reconcile the stale setup.py.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-002",
      "location": "Containerfile:1",
      "class": "bug",
      "severity": "high",
      "evidence": "`FROM python:3.7-slim` but the project requires Python >=3.10 (pyproject.toml:11 `requires-python = \">=3.10\"`, mypy.ini:2 `python_version = 3.10`, CI uses 3.11 in .github/workflows/main.yml:21). `RUN pip install .` (Containerfile:4) will fail dependency resolution / syntax on 3.7 (e.g. lightning>=2.2, torch>=2.0.1 wheels are unavailable for cp37 and the codebase uses 3.10+ syntax).",
      "recommendation": "Bump the base image to python:3.11-slim (or 3.10) to match requires-python and the CI matrix.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-003",
      "location": "Containerfile:5",
      "class": "bug",
      "severity": "high",
      "evidence": "`CMD [\"rna_predict\"]` invokes the console script defined at pyproject.toml:61, whose target module `rna_predict.__main__:main` does not exist anywhere in the tree (verified via find). Even if the image built, the container's default command would crash at startup with an import error.",
      "recommendation": "Fix the console-script target (see s2c0l0-001) or change CMD to a working entry point, e.g. [\"python\", \"-m\", \"rna_predict.predict\"].",
      "status": "survived"
    },
    {
      "id": "s2c0l0-001",
      "location": "pyproject.toml:61",
      "class": "bug",
      "severity": "high",
      "evidence": "[project.scripts] declares `rna_predict = \"rna_predict.__main__:main\"`. A repo-wide `find . -name __main__.py` returns no results (verified across the full tree), so the package's only console-script entry point references a module that does not exist. Installing the wheel creates an `rna_predict` command that fails with ModuleNotFoundError on invocation. The Stage-1 map (audit/01-understanding.md:15) records the same missing target.",
      "recommendation": "Either add rna_predict/__main__.py defining main(), or repoint the console script to an existing Hydra main such as `rna_predict.predict:main` or `rna_predict.interface:main`.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-dimsconfig-reduced-defaults",
      "location": "rna_predict/conf/config_schema.py:46-107",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "DimensionsConfig (and every per-stage dataclass) ships defaults that have been slashed from the documented AlphaFold3-inspired sizes to tiny test values, with inline comments admitting it: c_s default=8 '# CHANGED: was 384', c_z=4 '# was 128', c_s_inputs/c_token=8 '# was 449', c_atom=4 '# was 128', c_noise_embedding=4 '# was 32'. StageAConfig.num_hidden default=8 '# CHANGED: was 128 (to reduce memory usage)'. The Stage-1 provisional intent (audit/01-understanding.md:6,9) is a functional sequence-to-structure inference pipeline; these schema defaults instantiate a non-functional toy model unless a YAML overrides them. The structured schema, which exists to be the validated source of truth, instead encodes throwaway debug dimensions.",
      "recommendation": "Restore production dimensions as the dataclass defaults (c_s=384, c_z=128, c_s_inputs/c_token=449, c_atom=128, etc.) and move the reduced sizes into a dedicated 'test'/'minimal' Hydra override group, so the schema default reflects the real intended architecture.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-schema-yaml-dim-drift-protenix",
      "location": "rna_predict/conf/config_schema.py:604-613 vs rna_predict/conf/model/protenix_integration.yaml:8-12",
      "class": "doc_drift",
      "severity": "high",
      "evidence": "Two sources of truth for the same dimensions disagree. ProtenixIntegrationConfig defaults c_token=8, restype_dim=8, profile_dim=8, c_atom=4, c_pair=4 (all marked 'CHANGED: was 449/32/32/128/32'). The YAML actually composed into default.yaml (default.yaml:14 'model/protenix_integration@model.protenix_integration') sets c_token=449, restype_dim=32, profile_dim=32, c_atom=128, c_pair=32. Whichever wins depends on Hydra merge order, and the schema's documented default is the opposite of what the runtime YAML supplies, so the schema cannot be trusted as documentation.",
      "recommendation": "Pick one authoritative set of production dimensions and make the dataclass default and the YAML agree; or have the YAML interpolate the dimension from the structured schema/shared block rather than hardcoding a conflicting literal.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-predict-yaml-hardcoded-checkpoint",
      "location": "rna_predict/conf/predict.yaml:13",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "checkpoint_path: /Users/tomriddle1/RNA_PREDICT/outputs/checkpoints/last.ckpt is a hardcoded absolute developer path in predict.yaml, which is the config_name for predict.py — the primary, README-recommended inference entry (audit/01-understanding.md:28). On any other machine this path does not exist, so the documented 'Functional' inference flow loads a default/wrong checkpoint or fails. Comment '# Updated for correct test location' confirms it was tuned to one developer's box.",
      "recommendation": "Make checkpoint_path relative to the project (e.g. outputs/checkpoints/last.ckpt) or an env/CLI override (${oc.env:RNA_CKPT,...}); never ship an absolute /Users/... path in the default inference config.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-loader-dummy-shape-mismatch",
      "location": "rna_predict/dataset/loader.py:291-298 vs 300-304",
      "class": "bug",
      "severity": "high",
      "evidence": "_load_atom_features returns inconsistent tensor ranks/dtypes between its two branches. The empty-pdb dummy branch returns coords of shape (L, max_atoms, 3), atom_mask (L, max_atoms) float32, atom_to_tok (L, max_atoms) int32, elem_emb/name_emb (L, max_atoms, C) (lines 291-295). The real branch returns coords (max_atoms, 3), atom_mask (max_atoms,) bool, atom_to_tok (max_atoms,) long, elem_emb/name_emb (max_atoms, C) (lines 300-304). The function docstring (lines 255-261) documents only the 2D shapes. Downstream collate/model code cannot consume both a 2D and a 3D coords tensor, so the missing-file path produces shapes that disagree with every real sample.",
      "recommendation": "Make the dummy branch emit the same rank/dtype as the real branch ((max_atoms,3) coords, bool atom_mask, etc.), and add a shape assertion so the two paths cannot diverge.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-loader-atomfeat-shape-mismatch",
      "location": "rna_predict/dataset/loader.py:291-298 vs 300-326",
      "class": "bug",
      "severity": "high",
      "evidence": "_load_atom_features returns tensors of DIFFERENT rank/dtype depending on whether the structure file is present. Missing-file path (lines 291-298) returns coords shape (L, max_atoms, 3), atom_mask (L, max_atoms) float32, atom_to_tok (L, max_atoms) int32, elem_emb (L, max_atoms, ref_element_size). The real-data path (lines 300-326) returns coords (max_atoms, 3), atom_mask (max_atoms,) bool, atom_to_tok (max_atoms,) long, elem_emb (max_atoms, ref_element_size). __getitem__ (line 129) stores these directly into the sample, so a batch mixing present/absent structure files produces tensors of incompatible shapes/dtypes; rna_collate_fn (collate.py:99 torch.stack) will raise or silently produce wrong batch shapes, and downstream code receives a per-residue-blocked tensor in one case and a flat-atom tensor in the other.",
      "recommendation": "Make the dummy/missing-file branch produce identical rank and dtypes as the real branch: coords (max_atoms,3) coord_dtype, atom_mask (max_atoms,) bool, atom_to_tok (max_atoms,) long, elem_emb/name_emb (max_atoms, size) float32.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-latent-merger-rebuilds-weights-and-ignores-config",
      "location": "rna_predict/pipeline/merger/simple_latent_merger.py:63-73",
      "class": "design_defect",
      "severity": "high",
      "evidence": "SimpleLatentMerger.forward() reconstructs self.mlp with brand-new randomly-initialized nn.Linear layers whenever the runtime input dim differs from the constructed in_features (lines 63-73). At inference this silently discards any loaded/trained weights and emits output from an untrained MLP, with only a '[Debug] Creating MLP' print. Separately, the merger ignores the LatentMergerConfig contract: config_schema LatentMergerConfig defines merge_method ('concat'/'add'/'attention'), attention_heads, use_residual, output_dim=384 (config_schema.py:1148-1176), but this implementation hardcodes concat, has no residual/attention path, and takes only positional dim_* args — none of those config fields are referenced here.",
      "recommendation": "Validate/raise on dimension mismatch instead of rebuilding the network at forward time (or use a lazy module initialized once), and either implement the LatentMergerConfig options (merge_method/use_residual/attention_heads) or remove them from the schema so config and code agree.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-istest-returns-zeros",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:520-528",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "RFoldModel.forward contains `is_test = seqs.shape[0] <= 2 and seqs.shape[1] <= 16` and, when true, returns `torch.zeros((B, L, L))` instead of running the U-Net/Seq2Map. This is gated only on tensor sizes, not on any test flag. rfold_predictor.predict_adjacency pads every sequence to a multiple of 16 via _get_cut_len (rfold_predictor.py:316-329, :401), so any RNA sequence of length <=16 produces padded_len==16 with batch 1, hitting is_test and silently returning an all-zero adjacency matrix. Stage-1 provisional intent (01-understanding.md:6) is that inference (Stage A adjacency) is the functional deliverable; returning zeros for short sequences contradicts that.",
      "recommendation": "Remove the size-based is_test shortcut from production forward. If a fast path is needed for unit tests, gate it behind an explicit test-only constructor flag or move it into the test harness; never infer 'test mode' from real input dimensions.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-001",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:520-528",
      "class": "bug",
      "severity": "high",
      "evidence": "RFoldModel.forward sets is_test = seqs.shape[0] <= 2 and seqs.shape[1] <= 16, and when true returns torch.zeros((B, L, L)) instead of running the U-Net/Seq2Map. This 'test mode' heuristic fires on REAL inference: rfold_predictor.StageARFoldPredictor.predict_adjacency always calls the model with batch=1 (rfold_predictor.py:410 asserts shape[0]==1) and pads the sequence to a multiple of 16 via _get_cut_len, so any RNA sequence of length <=16 yields padded_len==16, triggering is_test and producing an all-zero adjacency matrix for genuine input. The stated intent (predict 2D adjacency/secondary structure) is silently violated for short sequences.",
      "recommendation": "Remove the is_test/zeros shortcut from production forward(); gate any test-only behavior behind an explicit constructor/config flag (e.g. self.test_mode) rather than inferring it from input dimensions, so real short-sequence inputs are processed by the real network.",
      "status": "survived"
    },
    {
      "id": "s2c2l1-rfold-torchload-pickle",
      "location": "rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:279",
      "class": "security",
      "severity": "high",
      "evidence": "StageARFoldPredictor._load_checkpoint calls `ckp = torch.load(checkpoint_path, map_location=self.device)` with no `weights_only=True` and no integrity check. torch.load uses Python pickle, which executes arbitrary code embedded in the file during unpickling. checkpoint_path comes from Hydra config (stage_cfg.checkpoint_path, set at :167) and the code explicitly contemplates the file being absent and fetched from a remote `checkpoint_url` (:191, :259-262) — the companion Stage A runner downloads and unzips it from a URL via urllib (rna_predict/pipeline/stageA/run_stageA.py:70 download_file + extract, referenced at :190). A malicious or MITM-tampered RFold checkpoint .pth therefore yields arbitrary code execution at load time. The provisional intent (Stage 1 map: 'RFold checkpoint download/extraction', README marks Inference as Functional) treats checkpoint loading as a normal trusted step, so the unrestricted pickle load is a defect relative to safe model-weight loading.",
      "recommendation": "Load with `torch.load(checkpoint_path, map_location=self.device, weights_only=True)` (or use a safetensors format) so only tensors/state-dicts are deserialized, never arbitrary objects. Additionally verify the downloaded artifact against a pinned checksum/signature before loading, and serve the checkpoint over HTTPS from a trusted host.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-adaln-runtime-layer-recreation",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:180-192",
      "class": "design_defect",
      "severity": "high",
      "evidence": "AdaptiveLayerNorm.forward, when s.shape[-1] != layernorm_s.normalized_shape, recreates self.layernorm_s, self.linear_s and self.linear_nobias_s as brand-new modules inside the forward pass and mutates self.c_s/self.c_s_layernorm. This discards any trained weights for those layers and replaces them with freshly-initialized parameters mid-run; in training these new params are not in the optimizer, and in inference loaded checkpoint weights are silently thrown away. AF3 Algorithm 26 (which this claims to implement, :24) has fixed dimensions.",
      "recommendation": "Validate/adjust input feature dimensions upstream (or raise) instead of silently rebuilding learnable layers during forward; never re-instantiate parameters inside forward().",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attninternal-test-hardcoded-reshapes",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils_internal.py:170-195,256-259,290-293,341-342",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "The attention output path (wrap_up -> _infer_and_reshape / apply_gating, all reachable from Attention.forward) is littered with test-specific hardcoded reshapes and magic numbers: apply_gating has 'Special case for the test_n_sample_handling test' keyed on `o.numel()==8192 and o.shape[1]==128` with literals 1024/8/128/64 (lines 170-195); _infer_and_reshape repeats 'Special case for the test_n_sample_handling test' returning `o.reshape(64,128)` when numel==8192 (lines 256-259, 290-293); wrap_up has 'Special case for the specific error we're seeing' for `o.shape[-1]==1024 and in_features==128` (lines 341-342). Production tensor reshaping is shaped around specific test fixtures rather than a principled shape contract.",
      "recommendation": "Define and enforce an explicit shape contract for attention outputs and remove all test-named special cases and magic-number reshapes; encode the expected shapes in the tests, not in production reshape logic.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-002",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:164",
      "class": "bug",
      "severity": "high",
      "evidence": "create_pair_embedding treats ref_pos as 2D [N,3]: d=linear_no_bias_d(ref_pos) then p_i=p.unsqueeze(1) (line 182), p_j=p.unsqueeze(0) (line 183), p_ij=p_i+p_j (line 184). For a batched input [B,N,c] this yields p.unsqueeze(1)=[B,1,N,c] and p.unsqueeze(0)=[1,B,N,c] which broadcast to [B,B,N,c] (an incorrect batch-x-batch outer product), not the intended [B,N,N,c]. extract_atom_features however handles batched [B,N,*] via torch.cat, so the two halves of the encoder disagree on rank.",
      "recommendation": "Use dim-relative unsqueeze (e.g. p.unsqueeze(-2) and p.unsqueeze(-3)) so the pair outer product is computed over the atom axis regardless of batch dims, and add a shape assertion for [..., N, N, c_atompair].",
      "status": "survived"
    },
    {
      "id": "s2c3l2-003",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:56-81",
      "class": "bug",
      "severity": "high",
      "evidence": "AtomAttentionEncoder (nn.Module, line 23) and AtomAttentionDecoder (decoder.py:20) assign self.feature_processor/self.coordinate_processor/self.attention_components to PLAIN classes (FeatureProcessor at atom_attention_feature_processing.py:22, AttentionComponents at attention_components.py:16, CoordinateProcessor at coordinate_processing.py:13 — none subclass nn.Module). The LinearNoBias/LayerNorm/AtomTransformer layers they hold are therefore never registered as submodules: they are absent from .parameters(), .state_dict() and are not moved by .to(device). For an encoder claiming to implement AF3 Algorithm 5 (encoder.py:26) this means its core weights are untrainable/unsaveable through the standard nn.Module API.",
      "recommendation": "Make FeatureProcessor/AttentionComponents/CoordinateProcessor subclass nn.Module (or register their layers via nn.ModuleDict/add_module on the parent), so parameters are tracked.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-013",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:622",
      "class": "design_defect",
      "severity": "high",
      "evidence": "AttentionPairBias.forward begins (line 622) with an unconditional `print(f\"[DEBUG][APB] ENTRY: ...\")` and emits five more unconditional `print(\"[DEBUG][APB] ...\")` statements (lines 643,656,659,664,665). Because a statement (the print) precedes the triple-quoted block at lines 623-637, that block is NOT the function docstring (forward.__doc__ is None) and the documented args are lost. This is the core attention block of the transformer; every forward floods stdout.",
      "recommendation": "Remove the debug prints (or use logger.debug guarded by isEnabledFor) and move the docstring to the first statement of forward.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-020",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:251-257",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "In _process_style_embedding, the conditional `if getattr(c_l,'shape',[None])[1] != getattr(x,'shape',[None])[1] and atom_to_token_idx is not None:` (line 251) has a body consisting ONLY of comments ('# ... (no change to the detailed broadcasting logic...)', '# (The rest of the function remains unchanged...)', lines 253-256) — the actual 'Broadcasting c_l from residues to atoms using atom_to_token_idx' operation it claims to perform was removed, leaving a no-op. The same gutting appears at lines 284-288 ('# (rest of fallback unchanged)'). The intended residue->atom broadcast is silently absent.",
      "recommendation": "Restore the removed broadcasting logic or delete the dead conditional and document the real behaviour; do not leave placeholder comments standing in for code.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-019",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:392-393",
      "class": "design_defect",
      "severity": "high",
      "evidence": "process_inputs_with_coords — the main coordinate forward path for the refactored AtomAttentionEncoder (called from atom_attention_encoder.py:167) — starts with two UNCONDITIONAL `print(\"[DEBUG][process_inputs_with_coords] ...\")` statements (lines 392-393), bypassing the file's own config-driven `debug` flag used everywhere else. Every coords forward prints to stdout.",
      "recommendation": "Replace the prints with logger.debug guarded by the existing `debug` flag.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-010",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/pair_embedding.py:39",
      "class": "perf",
      "severity": "high",
      "evidence": "create_pair_embedding builds the atom-pair embedding with nested Python loops over all atom pairs: _process_distances iterates `for query_idx in range(N) for key_idx in range(N)` (lines 39-40) calling encoder.linear_no_bias_d per pair (line 54) and indexed-assigning into pair_embed (line 63); _process_charges does the same O(N^2) Python double loop (lines 110-125). For realistic RNA atom counts (hundreds-to-thousands) this is O(N^2) Python-level iterations with a per-pair nn.Linear call, making the encoder effectively unusable beyond toy inputs and breaking the 'functional inference' intent.",
      "recommendation": "Vectorize: compute all pairwise distance vectors via ref_pos.unsqueeze(-2)-ref_pos.unsqueeze(-3) and apply linear_no_bias_d once over [...,N,N,3]; compute charge products with an outer product, eliminating the Python loops.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-025",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/pair_embedding.py:39-68",
      "class": "perf",
      "severity": "high",
      "evidence": "create_pair_embedding builds the [N_atom,N_atom,c_atompair] pair tensor via _process_distances (lines 39-68) and _process_charges (lines 110-125), each using O(N_atom^2) nested Python for-loops that call encoder.linear_no_bias_d per pair and perform in-place autograd writes `pair_embed[...,q,k,:] += ...`. For realistic atom counts this is orders of magnitude slower (and far more memory/graph-heavy) than a vectorized outer-difference + linear, undermining the 'Functional' inference path.",
      "recommendation": "Vectorize: compute all pairwise distance vectors with broadcasting (ref_pos[...,None,:,:]-ref_pos[...,:,None,:]) and apply the linear once; same for charge products.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-017",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:108-134",
      "class": "bug",
      "severity": "high",
      "evidence": "ConditionedTransitionBlock.forward applies AdaptiveLayerNorm TWICE due to an instrumentation block. Line 110 sets `a_norm = self.adaln(a, s)`; line 119 sets `a = a_norm`; then line 131 recomputes `a = self.adaln(a, s)` on the already-normalized a_norm. The intermediates linear_a1/linear_a2/b computed at lines 112-117 are discarded and recomputed at line 134 on the doubly-normalized tensor. The real output path therefore uses adaLN applied twice — a correctness regression introduced by the instrumentation, affecting every DiffusionTransformerBlock feed-forward (diffusion.py:119).",
      "recommendation": "Delete the instrumentation block (lines 109-119) so adaln is applied once; keep only the lines 130-153 computation.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-005",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:119",
      "class": "bug",
      "severity": "high",
      "evidence": "ConditionedTransitionBlock.forward computes a_norm=self.adaln(a,s) (line 110) and then sets a=a_norm (line 119) before the 'real' forward body. The real body re-applies adaptive layernorm: a=self.adaln(a,s) (line 131), so adaln is applied TWICE to the input (adaln(adaln(input))) and the subsequent gated SiLU (line 134) operates on the doubly-normalized tensor. This is a numerical correctness defect introduced by leftover instrumentation code (lines 108-119) sitting above the docstring/real logic.",
      "recommendation": "Delete the instrumentation block (lines 108-119) including the `a = a_norm` reassignment so adaln is applied exactly once; keep only the documented forward path.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-029",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:173",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "aggregate_atom_to_token branches on test identity in production: it reads `os.environ.get('PYTEST_CURRENT_TEST')` (line 173) and special-cases named tests 'test_run_stageD_basic' (line 183) and 'test_run_stageD_diffusion_inference_original' (lines 234,300), taking different reshape/fallback paths only when those tests are running (including an O(N^2) python double-loop scatter fallback at lines 318-322). A core aggregation utility behaving differently under specific pytest names means tests exercise code paths the real pipeline never takes, and vice-versa.",
      "recommendation": "Remove all PYTEST_CURRENT_TEST/test-name branches; implement one shape-handling path and fix the underlying shape contracts so tests and production share behaviour.",
      "status": "survived"
    },
    {
      "id": "s2c3l1-001",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:112-113",
      "class": "security",
      "severity": "high",
      "evidence": "unzip_file() extracts a downloaded checkpoint archive with `with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extract_dir)` (run_stageA.py:112-113) with NO validation of member names. A crafted zip whose entries contain '../' path components (or absolute paths) writes files outside extract_dir (classic Zip Slip, CWE-22 arbitrary file write). The archive originates from a network download (download_file at run_stageA.py:70-71) whose URL is the Hydra-configurable stage_cfg.checkpoint_url (run_stageA.py:188, default conf/model/stageA.yaml:12). extract_dir is os.path.dirname(checkpoint_dir) (run_stageA.py:191), i.e. an attacker-influenced member like '../../RFold/../../../home/user/.bashrc' could overwrite files. The intended behavior per 01-understanding.md is to fetch the RFold pretrained checkpoint, not to write arbitrary host paths.",
      "recommendation": "Before extracting, validate every member: reject names that are absolute or whose os.path.realpath(os.path.join(extract_dir, name)) does not stay within os.path.realpath(extract_dir); or extract members individually with a sanitized basename. Prefer Python 3.12's ZipFile.extractall(filter='data') / shutil.unpack_archive equivalents.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-035",
      "location": "rna_predict/pipeline/stageB/main.py:171-175",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "run_stageB_combined branches on `pairformer_model.return_value` (lines 171-175) — `return_value` is a unittest.mock.Mock attribute, and the surrounding comments (lines 164-167) explicitly say 'The test is mocking the pairformer_model ... use the mock's return values directly'. Production library code thus inspects test-mock internals to decide its data flow, coupling the shipped Stage B orchestrator to the test harness.",
      "recommendation": "Remove the return_value branch; rely solely on the genuine pairformer_output and assert its type/shape.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-036",
      "location": "rna_predict/pipeline/stageB/main.py:225-245",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "The ProtenixIntegration 's_inputs' branch fabricates synthetic inputs 'for testing purposes ... to speed up tensor generation': it shrinks dims (test_c_atom=min(32,c_atom), test_restype_dim=min(8,...), test_profile_dim=min(8,...) at lines 227-229) and builds constant/hard-coded input_features (ref_pos from a fixed 2-point tensor repeated, ref_charge/ref_element/restype/profile all torch.ones*0.1, lines 232-245). So the returned s_inputs are derived from fabricated dummy features rather than real embeddings, even in the non-test production call path of run_stageB_combined.",
      "recommendation": "Build input_features from the real sequence/embedding data at the configured dimensions; remove the 'for testing' downsizing from the production path.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-003",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:378-415",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "PairformerWrapper.predict() docstring (:381-392) states 'Predict RNA structure using the Pairformer model' and returns single/pair embeddings, but the body returns random tensors: `s_emb = torch.randn(L, self.c_s ...)`, `z_emb = torch.randn(L, L, self.c_z ...)` (:399-400) with the comment 'For now, return dummy tensors' (:397). self.stack (the real PairformerStack) is never invoked. Two unconditional debug prints remain at :413-414. This contradicts the Stage-1 intent that the pairwise branch produces meaningful embeddings.",
      "recommendation": "Implement predict() to run the embeddings through self.stack, or remove the method and document that only forward() is functional; remove the stray print() calls.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-009",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:164-205,885-897",
      "class": "design_defect",
      "severity": "high",
      "evidence": "Production predictor code branches on test identity via os.environ['PYTEST_CURRENT_TEST'] and hardcoded test names: raises for 'test_legacy_config_path_raises' (:164,:173), forces dummy_mode for named tests (:189), raises for 'test_stageb_torsionbert_config_structure_property' (:204), and __call__ contains a 'Special case for tests' block reshaping output to [N,16] when num_angles==16 and angle_mode=='degrees' (:885-897). Test-specific behavior is baked into the runtime model, coupling production output to test names.",
      "recommendation": "Move all test-only behavior into test fixtures/mocks; remove PYTEST_CURRENT_TEST and test-name conditionals and the num_angles==16 reshape from production code paths.",
      "status": "survived"
    },
    {
      "id": "s2c4l1-001",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:294-317",
      "class": "security",
      "severity": "high",
      "evidence": "StageBTorsionBertPredictor loads the TorsionBERT tokenizer and model with trust_remote_code=True in all four branches: AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True, local_files_only=True) (line 294), AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True) (line 301), AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True) (line 307), AutoModel.from_pretrained(..., trust_remote_code=True) (line 314). trust_remote_code=True causes HuggingFace transformers to download and execute arbitrary Python (modeling_*.py / configuration_*.py) from the model repository at load time. self.model_name_or_path defaults to the third-party hub id 'sayby/rna_torsionbert' (DEFAULT_MODEL_PATH at line 20) and is otherwise taken directly from Hydra config (lines 240/266: getattr(torsion_cfg, 'model_name_or_path', ...)). For non-local ids (else branches at 299/312) the repo is fetched from the network with no pinned revision/commit hash, so a compromised or hijacked hub repo, or a config pointing at an attacker-controlled repo, results in remote code execution in the inference/training process. This is the README-documented default Stage B model, so the dangerous path is the normal execution path, not an edge case.",
      "recommendation": "Default trust_remote_code to False and only enable it behind an explicit, documented opt-in flag for a vetted model. Pin the model with a fixed revision/commit hash (revision=...) when calling from_pretrained for non-local ids. Prefer vendoring the model weights/code locally and loading with local_files_only=True from a path under the repo's control, and validate model_name_or_path against an allowlist rather than passing arbitrary config strings straight to from_pretrained.",
      "status": "survived"
    },
    {
      "id": "s2re3-002",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:301-303,314-316",
      "class": "security",
      "severity": "high",
      "evidence": "AutoTokenizer.from_pretrained (:301-304) and AutoModel.from_pretrained (:314-317) are called with trust_remote_code=True on a remote Hugging Face hub id (self.model_name_or_path resolves to 'sayby/rna_torsionbert' per README.md:53,193) whenever is_local_path() is false (:299,:312). trust_remote_code=True executes arbitrary Python shipped in that third-party model repo at load time — a remote-code-execution exposure tied to an external repo the project does not control. Not in the listed findings.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c4l0-torsionbert-getpeftmodel-args",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:381",
      "class": "bug",
      "severity": "high",
      "evidence": "get_peft_model is called as get_peft_model(self.model, lora_config, self.model_name_or_path, self.lora_cfg.r, self.lora_cfg.lora_alpha, self.lora_cfg.target_modules, self.lora_cfg.bias). The PEFT signature is get_peft_model(model, peft_config, adapter_name='default', mixed=False, ...); the extra positional args map model_name_or_path->adapter_name, r(int)->mixed(bool), lora_alpha->autocast/revision, target_modules(list)/bias onto further keyword-only params. This will raise a TypeError or silently mis-bind parameters whenever LoRA is actually applied. The hyperparameters (r, alpha, target_modules, bias) are already carried by lora_config, so they must not be passed again.",
      "recommendation": "Call get_peft_model(self.model, lora_config) (optionally with a string adapter_name) and remove the trailing positional arguments.",
      "status": "survived"
    },
    {
      "id": "s2c4l1-002",
      "location": "rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:202-230",
      "class": "security",
      "severity": "high",
      "evidence": "TorsionBertModel.__init__ loads tokenizer and model with trust_remote_code=True: AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True) (line 202), AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) (line 209), AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True) (line 220), AutoModel.from_pretrained(model_path, trust_remote_code=True) (line 227). model_path is a free-form constructor argument (line 157) and the non-local branches (lines 207/225) pull from the HuggingFace Hub with no pinned revision, so loading executes arbitrary repository-supplied Python. Same remote-code-execution / supply-chain exposure as the StageBTorsionBertPredictor path; both code paths load the RNA TorsionBERT model and share the risk.",
      "recommendation": "Same mitigation as s2c4l1-001: disable trust_remote_code by default, gate it behind explicit opt-in, pin a revision hash for remote loads, validate/allowlist model_path, and prefer locally vendored weights loaded with local_files_only=True.",
      "status": "survived"
    },
    {
      "id": "s2re2-002",
      "location": "rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:204,211,222,229; rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:296-316",
      "class": "security",
      "severity": "high",
      "evidence": "AutoTokenizer.from_pretrained / AutoModel.from_pretrained are called with trust_remote_code=True for the TorsionBERT model. When model_path is not a local dir it resolves the HuggingFace hub id 'sayby/rna_torsionbert' (README.md:53,193) and trust_remote_code=True instructs transformers to download and execute arbitrary Python (modeling_*.py) from that remote repo at import time. A compromised or hijacked hub repo yields remote code execution on any inference/training host. No trust_remote_code finding exists in the current set.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re3-003",
      "location": "rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:209-211,227-229",
      "class": "security",
      "severity": "high",
      "evidence": "Second, independent code path: AutoTokenizer.from_pretrained (:209-212) and AutoModel.from_pretrained (:227-230) also pass trust_remote_code=True for the non-local hub-id branch (else of is_local_path at :207,:225). Same arbitrary-code-execution risk on the downloaded TorsionBERT model repo as s2re3-002, in a distinct file. Not in the listed findings.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c4l2-024",
      "location": "rna_predict/pipeline/stageC/mp_nerf/protein_utils/structure_utils.py:148-162,447",
      "class": "bug",
      "severity": "high",
      "evidence": "Unit mismatch: BB_BUILD_INFO['BONDANGS'] stores angles in RADIANS (sidechain_data.py:17-22: ca-c-n=2.124, c-n-ca=2.035, n-ca-c=1.939, ca-c-o=2.094). structure_utils._get_atom_placement_params reads these as 'bond_angle_deg' (:149-162, e.g. BB_BUILD_INFO[...].get('ca-c-n',116.2)) and AtomPlacementParams.to_mp_nerf_params then applies theta=np.radians(self.bond_angle_deg) (:126-130). So when the dict key is present, np.radians(2.124)=0.037 rad is used instead of 2.124 rad. _place_first_residue does the same (n_ca_c_angle_deg=BB_BUILD_INFO...get('n-ca-c',111.0); np.radians(...) at :446-447). The fallback defaults (116.2/111.0) are degrees, but the actual stored values are radians, so present-key cases yield grossly wrong bond angles.",
      "recommendation": "Treat BB_BUILD_INFO BONDANGS as radians (do not re-apply np.radians) or convert the stored constants to degrees consistently; add a unit assertion/test.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-rna-baseangle-units",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:262-266,307-311,274,331",
      "class": "bug",
      "severity": "high",
      "evidence": "Base-atom bond angles come from final_kb_rna BASE_GEOMETRY['bond_angles_deg'] (DEGREES, e.g. 105.8/120.3) and the fallback default is literally 120.0; they are converted only with torch.tensor(float(bond_angle)) (no deg->rad) and passed as the `theta` argument to calculate_atom_position(). calculate_atom_position treats theta as RADIANS (uses torch.cos(theta)/torch.sin(theta), line 89-91) and even emits '[WARN-RNAPREDICT-ANGLE-RANGE-001] Bond angle theta is outside [-pi, pi]' (rna_atom_positioning.py:47-48) for any value >pi. A 120-degree angle is therefore consumed as ~120 radians, producing geometrically wrong base coordinates.",
      "recommendation": "Convert bond angles to radians (deg_to_rad / math.radians) before passing them as theta to calculate_atom_position, in both the OP1/OP2 branch (line 262-266) and the default-placement branch (line 307-311), and for the 120.0 default.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-001",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:43-48",
      "class": "bug",
      "severity": "high",
      "evidence": "build_rna_chain_from_internal_coords loops over every residue i and unconditionally places the 'P' atom at the origin [0.0,0.0,0.0] (line 48), then builds the remaining backbone atoms only relative to atoms WITHIN the same residue (NeRF refs at lines 62,86,121 all index residue_coords[i, ...]). No inter-residue translation/rotation is ever applied, and the residue-linking point_ref_mask produced by rna_scaffolding (which references the previous residue's C4'/C3'/O3', rna_scaffolding.py:101-108) is never consumed here. Result: all residues are superimposed at the same local frame, so the returned (num_residues, num_atoms, 3) tensor is a physically collapsed structure rather than a connected RNA chain. This contradicts Stage C's stated intent (forward-kinematics reconstruction to atomic coordinates, audit/01-understanding.md:9).",
      "recommendation": "Carry the chain frame forward: place residue i's P relative to residue i-1's O3' (or otherwise chain residues via the previous residue's terminal atoms) instead of resetting P to the origin for every residue, or document/route per-residue local frames to a downstream assembler if that is the true contract.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-007",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:250-262,288-293",
      "class": "bug",
      "severity": "high",
      "evidence": "validate_stageC_config explicitly accepts device 'auto' (line 168), and run_stageC builds a default config with device='auto' when device is None (line 459). run_stageC_rna_mpnerf then sets device = stage_cfg.device (='auto'), only WARNS for unsupported devices (line 260-261 'Proceeding anyway'), and passes device='auto' straight into build_scaffolds_rna_from_torsions and rna_fold, which call torch.zeros(..., device='auto') / torch.device('auto'). PyTorch rejects 'auto' as a device string, raising RuntimeError. So the documented/default 'auto' device crashes instead of resolving to cpu/cuda/mps.",
      "recommendation": "Resolve 'auto' to a concrete device (cuda if available else cpu/mps) before constructing any tensors; do this in validate/normalization rather than warning-and-proceeding.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-009",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:96-112",
      "class": "design_defect",
      "severity": "high",
      "evidence": "StageCReconstruction.__call__ (the 'legacy' method) returns all-zero tensors: coords=torch.zeros((N*3,3)), coords_3d=torch.zeros((N,3,3)), with empty atom_metadata. validate_stageC_config explicitly permits method=='legacy' (line 165), and run_stageC dispatches to this zero-producing path when method != 'mp_nerf' (lines 477-488). So selecting the configured 'legacy' reconstruction silently yields physically meaningless all-zero coordinates rather than reconstructed atoms, contradicting Stage C's intent of producing atomic coordinates.",
      "recommendation": "Either remove 'legacy' from the allowed methods, raise NotImplementedError when selected, or implement a real legacy reconstruction; do not return zero placeholders behind a configurable option.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-016",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py:163-185,369-376",
      "class": "bug",
      "severity": "high",
      "evidence": "Inside forward (_process_pair_features / _process_single_features), when the runtime feature dimension differs from the configured one, the module REPLACES sub-layers with freshly constructed ones: `self.layernorm_z = LayerNorm(actual_z_dim).to(...)` (line 171), `self.linear_no_bias_z = LinearNoBias(in_features=actual_z_dim, ...)` (lines 180-183), and `self.linear_no_bias_s = LinearNoBias(in_features=single_s.shape[-1], ...)` (lines 373-376). This is done in the forward pass, so each mismatching forward creates new randomly-initialized parameters that are not registered with any optimizer and discard previously trained weights, breaking training and yielding nondeterministic inference output.",
      "recommendation": "Fix tensor dimensions to the configured sizes (pad/project deterministically) instead of recreating nn layers inside forward; size layers once in __init__ and treat a runtime mismatch as a hard error.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-027",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:99-132,950-956,1021-1028",
      "class": "design_defect",
      "severity": "high",
      "evidence": "DiffusionModule couples production behavior to the test harness. __init__ checks os.environ['PYTEST_CURRENT_TEST'] and, for 'test_init_with_basic_config', sets a few attrs from kwargs and returns early skipping all real module construction (lines 99-132). forward() and _compute_loss() inspect the caller's frame name via get_caller_frame() and, if it contains 'test_n_sample_handling', return only coordinates / a dummy zero loss (lines 950-956, 1023-1028). Real model output depends on whether a caller function is named like a test.",
      "recommendation": "Eliminate PYTEST_CURRENT_TEST and caller-frame-name branching from the module; express these special cases via test doubles/parameters.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-031",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:317-323,447,509-510",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "multi_step_inference builds its noise schedule via _get_noise_schedule which returns torch.linspace(1.0, 0.0, num_steps+1) for BOTH the 'linear' branch and the default branch (lines 317-323) — schedule_type is effectively ignored, and the NoiseScheduleConfig parameters (s_max=160, s_min=4e-4, p=7, sigma_data) read into noise_schedule_cfg (line 447) are never used. Meanwhile generator.py provides a proper EDM InferenceNoiseScheduler (generator.py:78-141) that is never invoked for inference. The diffusion is run with a trivial 1->0 linear schedule, contradicting the AF3/EDM-inspired schedule the config describes.",
      "recommendation": "Use InferenceNoiseScheduler driven by the noise_schedule config (s_max/s_min/p/sigma_data) instead of the placeholder linspace, and honor schedule_type or remove the dead config.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-020",
      "location": "rna_predict/pipeline/stageD/memory_optimization/run_stageD_memory_efficient.py:10",
      "class": "bug",
      "severity": "high",
      "evidence": "Imports `from rna_predict.pipeline.stageD.memory_fix import run_stageD_with_memory_fixes`, but no such module exists at that path; the file is at rna_predict/pipeline/stageD/memory_optimization/memory_fix.py (verified: `find rna_predict/pipeline/stageD -name memory_fix.py` returns only the memory_optimization/ copy, and stageD/ top level has no memory_fix.py). The argparse main entry (audit/01-understanding.md:26 lists it as a Stage D entry point) therefore fails immediately with ModuleNotFoundError and can never run.",
      "recommendation": "Fix the import to `from rna_predict.pipeline.stageD.memory_optimization.memory_fix import run_stageD_with_memory_fixes` (or relative `from .memory_fix import ...`).",
      "status": "survived"
    },
    {
      "id": "s2c6l0-init-add-broadcast",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:251-293",
      "class": "design_defect",
      "severity": "high",
      "evidence": "A second, different global override of torch.Tensor.__add__ (fix_tensor_add) silently reshapes/interpolates/avg-pools operands to make additions 'succeed' (_expand_tensor_dimension uses adaptive_avg_pool1d / repeat_interleave). This masks genuine shape bugs by inventing data and changing numerics globally. Note this is a distinct implementation from tensor_operations.py's fix_tensor_add, so behavior depends on which apply path runs.",
      "recommendation": "Remove global arithmetic monkeypatching; resolve shape mismatches at their source. At minimum gate behind an explicit opt-in and never silently resample tensor values.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-init-gather-global",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:296-316",
      "class": "bug",
      "severity": "high",
      "evidence": "fix_gather_pair_embedding() replaces torch.gather globally with patched_gather(x, dim_or_idx_q, index_or_idx_k=None). The real torch.gather signature is gather(input, dim, index, *, sparse_grad=False, out=None); the wrapper drops sparse_grad/out and, in the non-int branch, ignores idx_k semantics and hard-codes dim=1 with extra unsqueezes. Any code in the process calling torch.gather with keyword args or non-trivial dims gets wrong results or TypeErrors.",
      "recommendation": "Never replace torch.gather process-wide. Provide a named helper for the pair-embedding case and call it explicitly.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-init-module-forward",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:370-386",
      "class": "design_defect",
      "severity": "high",
      "evidence": "fix_atom_transformer() replaces torch.nn.Module.forward (the base class method for EVERY nn.Module) with patched_forward(self, q, c, p, inplace_safe=False, chunk_size=None). apply_tensor_fixes() (line 424) calls this. The signature is specific to an atom-transformer yet is installed on the universal base class; any module relying on Module.forward (or introspection of it) is affected, and the positional contract is meaningless for general modules.",
      "recommendation": "Do not patch torch.nn.Module.forward. Patch the concrete AtomTransformer class only (as transformer_fixes.py attempts), or pass corrected tensors explicitly.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-attn-sdpa-args",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/attention_fixes.py:19-41",
      "class": "bug",
      "severity": "high",
      "evidence": "patched_attention(q,k,v,attn_bias,dropout_p,scale,dtype) calls original_scaled_dot_product_attention(q,k,v,attn_bias,dropout_p,scale,dtype) positionally. torch.nn.functional.scaled_dot_product_attention's positional order is (query,key,value,attn_mask,dropout_p,is_causal,scale,...); there is no dtype parameter. So `scale` is passed into the is_causal slot and `dtype` into the scale slot, silently producing causal masking / wrong scaling, and the replacement is installed globally (line 64).",
      "recommendation": "Match the real SDPA signature (use keyword args: attn_mask=, dropout_p=, is_causal=, scale=) and drop the non-existent dtype param; avoid global replacement.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-011",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/attention_fixes.py:19-64",
      "class": "bug",
      "severity": "high",
      "evidence": "fix_attention_bias_shape() replaces torch.nn.functional.scaled_dot_product_attention with patched_attention(q,k,v,attn_bias=None,dropout_p=0.0,scale=None,dtype=None) (:19-21,:64). This signature does not match the real F.scaled_dot_product_attention(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,enable_gqa=False): a positional caller passing is_causal would bind it to 'scale', and extra positional args raise TypeError. The bogus 'dtype' param is not accepted by the real function either.",
      "recommendation": "If patching is unavoidable, mirror the exact upstream signature with *args/**kwargs passthrough; better, remove the patch and feed correctly-shaped masks.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-012",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:10-38",
      "class": "bug",
      "severity": "high",
      "evidence": "tensor_operations.fix_tensor_add defines a SECOND torch.Tensor.__add__ override (:38) whose behavior contradicts the one in tensor_fixes/__init__.py: on a 'must match the size...non-singleton dimension' error it returns one operand unchanged (return self / return other, :29-32) WITHOUT performing the addition, silently producing a wrong result.",
      "recommendation": "Delete this contradictory global override; never silently drop an addition. Resolve shape mismatches at the source.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-013",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:41-108",
      "class": "bug",
      "severity": "high",
      "evidence": "fix_matrix_multiplication patches torch.matmul/torch.bmm/torch.nn.functional.linear to truncate inner dimensions to the min and retry (lines 68-72,81-85,98-102), producing silently-wrong numeric results. The retry path calls the now-patched globals (torch.matmul/torch.bmm/F.linear) risking recursion, and the re-patch guard checks attributes (_patch_applied_safe_linear/_safe_matmul/_safe_bmm, :47-53) that are never set on the wrappers, so the guard never actually prevents re-patching.",
      "recommendation": "Remove dimension-truncating math overrides; fix dimension mismatches at the model level. If a guard is needed, actually set the sentinel attribute on the wrapper.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-024",
      "location": "rna_predict/runners/full_pipeline.py:154-171,198-244,533-552",
      "class": "design_defect",
      "severity": "high",
      "evidence": "The orchestrator swallows broad exceptions and returns silently-fabricated outputs: Stage A failures return torch.eye identity adjacency (:151,:158,:171); Stage B init/run failures return all-zero torsion_angles/embeddings (:200-207,:217-223,:238-244); a top-level RuntimeError/AssertionError handler returns empty/dummy tensors for every key (:536-552). A consumer cannot distinguish a real prediction from a zero/identity fallback, contradicting the pipeline's purpose of producing structure predictions.",
      "recommendation": "Fail fast (or return an explicit error/status flag) instead of returning identity/zero placeholders that masquerade as valid predictions.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-022",
      "location": "rna_predict/runners/full_pipeline.py:227-234,377-379",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "run_full_pipeline runs Stage A (run_stage_a, :377) to produce an adjacency matrix, but run_stage_b (:379) ignores it: run_stage_b internally calls run_stageB_combined with adjacency_matrix=torch.eye(len(sequence)) (:229), a hardcoded identity. Stage A's output is only later fed to the optional latent merger (:467), not to Stage B torsion/pairformer. This contradicts the Stage-1 provisional intent of a composed A->B->C->D pipeline where Stage A's secondary structure informs Stage B.",
      "recommendation": "Thread the real adjacency from run_stage_a into run_stage_b/run_stageB_combined, or document that Stage A is not consumed by Stage B and remove the misleading composition.",
      "status": "survived"
    },
    {
      "id": "s2c6l1-lightning-zip-slip",
      "location": "rna_predict/training/rna_lightning_module.py:173",
      "class": "security",
      "severity": "high",
      "evidence": "_unzip_file() does `with zipfile.ZipFile(zip_path,'r') as zip_ref: zip_ref.extractall(extract_dir)` with no validation of archive member names. A zip whose entries contain '../' or absolute paths (Zip Slip) will be written outside extract_dir, enabling arbitrary file overwrite. The zip is the Stage A checkpoint archive fetched from a config-supplied remote URL via _download_file (rna_lightning_module.py:204-208), so a malicious or MITM'd checkpoint_url leads to arbitrary file write on the host.",
      "recommendation": "Before extraction, validate each member: reject names that are absolute or whose normalized path escapes extract_dir (os.path.realpath join check), or use a vetted safe-extract helper. Verify archive integrity (hash/signature) before unzipping.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-035",
      "location": "rna_predict/training/rna_lightning_module.py:211,365,332-336",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "RNALightningModule instantiates self.stageA = StageARFoldPredictor (and downloads/extracts its checkpoint, :191-215) and registers it in the pipeline ModuleDict (:238), but forward() never calls it: adjacency comes from batch['adjacency'] (:363) and a full-file grep for self.stageA(/self.stageA.predict/predict_adjacency in this module returns no matches. Stage A is dead in the training/inference forward pass despite the pipeline claiming an A->B->C->D composition.",
      "recommendation": "Either invoke self.stageA to produce adjacency within forward, or remove Stage A from the module and document that adjacency is supplied externally.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-036",
      "location": "rna_predict/training/rna_lightning_module.py:56-64,519,687,810-817",
      "class": "design_defect",
      "severity": "high",
      "evidence": "Production training logic branches on test identity: __init__ inspects the caller's filename for 'test_partial_checkpoint_full_pipeline.py' to set self._integration_test_mode (:56-62), and training_step repeatedly reads os.environ['PYTEST_CURRENT_TEST'] to alter control flow ('test_noise_and_bridging_runs' :519,:687; 'test_run_stageD_basic' :813). This couples shipped behavior to specific test names and makes runtime behavior depend on the test harness.",
      "recommendation": "Drive test-only behavior via explicit constructor flags/config, not by sniffing caller filenames or PYTEST_CURRENT_TEST in production code paths.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-033",
      "location": "rna_predict/training/train.py:19-20,217",
      "class": "bug",
      "severity": "high",
      "evidence": "Both @hydra.main decorators hardcode config_path='/Users/tomriddle1/RNA_PREDICT/rna_predict/conf' (:20 and :217), an absolute developer-specific path that will not resolve on any other machine, breaking training/CI. The immediately preceding comment '# Use a relative config path instead of absolute' (:19) directly contradicts the code.",
      "recommendation": "Use a path relative to the file (e.g. config_path='../conf') or Hydra search-path/ConfigStore; honor the comment that already states the intent.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-train-abs-config",
      "location": "rna_predict/training/train.py:20,217",
      "class": "bug",
      "severity": "high",
      "evidence": "Both @hydra.main decorators hardcode config_path='/Users/tomriddle1/RNA_PREDICT/rna_predict/conf' (a developer-specific absolute path), despite the comment 'Use a relative config path instead of absolute' (line 19). On any other machine Hydra cannot find the config dir and training fails to launch.",
      "recommendation": "Use config_path relative to the module (e.g. '../conf') or derive from PROJECT_ROOT/importlib.resources.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-034",
      "location": "rna_predict/training/train.py:20,217-219",
      "class": "bug",
      "severity": "high",
      "evidence": "execute_training_run is itself decorated with @hydra.main (:20), and main() (also @hydra.main, :217) calls execute_training_run(cfg) (:219). Invoking a @hydra.main-wrapped function re-enters Hydra initialization and ignores the cfg passed in. The Stage-1 inventory also notes the Kaggle harness calls execute_training_run programmatically, which would trigger the same nested-Hydra problem.",
      "recommendation": "Have a single @hydra.main entry point delegate to an undecorated implementation function (e.g. _execute_training_run(cfg)); expose that plain function for programmatic callers.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-train-nested-hydra",
      "location": "rna_predict/training/train.py:20-22,217-219",
      "class": "bug",
      "severity": "high",
      "evidence": "execute_training_run is itself decorated with @hydra.main (line 20). main (also @hydra.main, line 217) calls execute_training_run(cfg) directly (line 219). Invoking a @hydra.main-wrapped callable re-enters Hydra initialization (re-parses sys.argv / re-instantiates Hydra), which errors ('Hydra is already initialized' / argv reparsing). The Kaggle harness also imports and calls execute_training_run, hitting the same double-initialization.",
      "recommendation": "Split the logic: a plain function body (no decorator) containing the work, wrapped by a single @hydra.main entry. Have both main and external callers invoke the undecorated function.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-046",
      "location": "rna_predict/utils/shape_utils.py:186-209,231-236",
      "class": "design_defect",
      "severity": "high",
      "evidence": "ensure_consistent_sample_dimensions (imported into production by stageD/diffusion/run_stageD_unified.py) reads os.environ['PYTEST_CURRENT_TEST'] and special-cases tensor expansion for named tests 'test_single_sample_shape_expansion'/'test_multi_sample_shape_fix' (:189-190,:200-209,:232-236). Shipped tensor-shape behavior thus differs depending on whether a specific pytest is running.",
      "recommendation": "Remove PYTEST_CURRENT_TEST branches; make sample-dimension handling deterministic and identical in and out of tests.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-analyze-token",
      "location": "scripts/analysis/analyze_code.sh:130-135",
      "class": "security",
      "severity": "high",
      "evidence": "A CodeScene access token is hardcoded and exported in the script (CS_ACCESS_TOKEN default value at line 132). A live credential is committed to the repository in plaintext; anyone with repo access obtains it, and it cannot be rotated without a code change. (Secret referenced by location/category only, not reproduced here.)",
      "recommendation": "Remove the embedded token, require it via environment/secret manager, and rotate the leaked credential.",
      "status": "survived"
    },
    {
      "id": "s2c6l1-analyze-hardcoded-token",
      "location": "scripts/analysis/analyze_code.sh:132",
      "class": "security",
      "severity": "high",
      "evidence": "The script hardcodes and exports a default CodeScene access token (category: third-party API access credential) directly in source when CS_ACCESS_TOKEN is unset (analyze_code.sh:130-135). A committed credential is exposed to anyone with repo read access and remains valid until revoked; it grants CodeScene CLI 'refactor.access'/'cli.access' per the embedded token claims.",
      "recommendation": "Remove the literal token from source, require CS_ACCESS_TOKEN to be supplied via the environment/secret store, and rotate/revoke the leaked token since it is already committed to history.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-003",
      "location": "scripts/dev.js:16",
      "class": "bug",
      "severity": "high",
      "evidence": "`import { runCLI } from './modules/commands.js';` targets scripts/modules/commands.js, but the scripts/modules/ directory does not exist (corroborated by audit/01-understanding.md:14, entry-point table, which notes the missing target). Every npm script that maps to `node scripts/dev.js` (dev/list/generate/parse-prd per package.json) crashes at module resolution time.",
      "recommendation": "Restore/ship the scripts/modules/ package or repoint the import to the actual CLI implementation; otherwise remove the dead npm scripts and dev.js to avoid a broken declared entrypoint.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-004",
      "location": "scripts/partial_checkpoint_full_pipeline_script.py:67",
      "class": "bug",
      "severity": "high",
      "evidence": "`with hydra.initialize(config_path=\"/Users/tomriddle1/RNA_PREDICT/rna_predict/conf\", ...)`. hydra.initialize requires a path RELATIVE to the calling module (absolute paths must use initialize_config_dir), and this absolute path is a developer-specific machine path that does not exist on any other host. The script earlier computes a correct relative `config_path_selected` (lines 42-62) but then ignores it and hardcodes the absolute path, so initialization fails on every non-author machine (caught at lines 71-73 -> sys.exit(1)).",
      "recommendation": "Use the already-computed relative `config_path_selected` with hydra.initialize, or switch to hydra.initialize_config_dir(config_dir=str(config_path.resolve())) for an absolute path.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-009",
      "location": "scripts/run_all_pipeline.py:6-7",
      "class": "bug",
      "severity": "high",
      "evidence": "PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__)))) (:7) with comment 'assuming this script is in <root>/rna_predict/scripts' (:6). But reorganize_scripts.sh moved this file to the top-level scripts/ (line 33), only TWO levels below root. From /home/user/RNA_PREDICT/scripts/run_all_pipeline.py the triple-dirname yields /home/user (one level too high). Consequently every entry in python_files (e.g. os.path.join(PROJECT_ROOT,'rna_predict/pipeline/stageA/run_stageA.py'), :61-66) resolves under /home/user/rna_predict/... which does not exist, so main() skips all files as 'File not found' (:83-84) and cwd=PROJECT_ROOT (:25) is also wrong.",
      "recommendation": "Update the path computation to two levels (PROJECT_ROOT = dirname(dirname(abspath(__file__)))) and fix the stale comment to reflect the scripts/ location.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-005",
      "location": "scripts/run_all_pipeline.py:7",
      "class": "bug",
      "severity": "high",
      "evidence": "`PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` assumes the script lives at <root>/rna_predict/scripts/ (comment at line 6), but the file now resides at top-level scripts/. Verified resolution: for scripts/run_all_pipeline.py PROJECT_ROOT becomes /home/user (one directory ABOVE the repo root /home/user/RNA_PREDICT). All targets built as os.path.join(PROJECT_ROOT, 'rna_predict/pipeline/...') (lines 61-66) then point at /home/user/rna_predict/... which does not exist, so every stage is reported 'File not found' and skipped (lines 83-91).",
      "recommendation": "Compute PROJECT_ROOT with two dirname() calls (scripts/ -> repo root) or anchor on a marker, e.g. `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-001",
      "location": "scripts/test_utils/batch_test_generator.py:4",
      "class": "bug",
      "severity": "high",
      "evidence": "Imports `from rna_predict.scripts.hypot_test_gen import run_test_generation`, but rna_predict/scripts/hypot_test_gen.py only defines `remove_logger_lines` and `fix_leading_zeros` (verified via grep -n 'def '); it has no `run_test_generation`. The real `run_test_generation` lives in the sibling file scripts/test_utils/hypot_test_gen.py:836. The import therefore raises ImportError at module load, so the script (which __main__-guards main() at :60) cannot run at all. Root cause is reorganize_scripts.sh moving batch_test_generator.py to scripts/test_utils/ (line 10) without updating its import target.",
      "recommendation": "Change the import to `from scripts.test_utils.hypot_test_gen import run_test_generation` (or a relative `from .hypot_test_gen import ...`) pointing at the file that actually defines the function.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-007",
      "location": "scripts/test_utils/batch_test_generator.py:4-6",
      "class": "bug",
      "severity": "high",
      "evidence": "Module-level `from rna_predict.scripts.hypot_test_gen import run_test_generation`. Verified that rna_predict/scripts/hypot_test_gen.py defines only remove_logger_lines (line 8) and fix_leading_zeros (line 29) and NOT run_test_generation (grep returned no match). The full run_test_generation/TestGenerator implementation lives instead in scripts/test_utils/hypot_test_gen.py. Therefore this import raises ImportError at load time and the script cannot run at all.",
      "recommendation": "Import from the sibling module that actually defines it, e.g. `from scripts.test_utils.hypot_test_gen import run_test_generation` (or a relative `from .hypot_test_gen import run_test_generation`), and fix the stale rna_predict.scripts package reference.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-005",
      "location": "scripts/test_utils/mark_slow_tests.py:69-72",
      "class": "bug",
      "severity": "high",
      "evidence": "add_slow_marker() guards `if modified:` (:69) BEFORE running the transformer that sets `modified` — `SlowTestMarker().visit(tree)` (:70) and the `nonlocal modified; modified = True` (:65-66) are inside that very block. Since `modified` is initialized False (:33) and nothing flips it beforehand, the block never executes, the visitor never runs, and no file is ever written. The script's sole purpose (inserting @pytest.mark.slow) is dead.",
      "recommendation": "Run the transformer unconditionally first (e.g. `tree = SlowTestMarker().visit(tree)`), then write the file only if `modified` became True.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-006",
      "location": "scripts/test_utils/mark_slow_tests.py:69-73",
      "class": "bug",
      "severity": "high",
      "evidence": "add_slow_marker sets `modified = False` (line 33), defines SlowTestMarker whose visit_FunctionDef sets `nonlocal modified = True` and inserts the marker, but then guards the actual transform with `if modified:` (line 69) BEFORE invoking the visitor (`tree = SlowTestMarker().visit(tree)` is on line 70, inside that block). Since modified is False at line 69, the block is skipped, the visitor never runs, modified is never set, and the file is never written. The script is a permanent no-op and never adds @pytest.mark.slow to any test.",
      "recommendation": "Run the transformer first, then check the flag: `marker = SlowTestMarker(); tree = marker.visit(tree); if modified: ast.fix_missing_locations(tree); write`. Capture `modified` from the visitor instance rather than a closure guarded before the visit.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-007",
      "location": "scripts/test_utils/run_failing_tests.sh:319",
      "class": "bug",
      "severity": "high",
      "evidence": "COVERAGE_GOAL is captured from get_coverage_goal() (which echoes many lines plus a final bc value like '80.50') then re-extracted via `grep -o '[0-9]\\+' | tail -1`. Because grep -o splits '80.50' into separate matches '80' and '50', tail -1 returns the fractional part. Verified: printf '80.50' | grep -o '[0-9]\\+' | tail -1 => 50; for a scale=2 value like '80.00' it returns '00' => 0. The wrong value is then passed to `--cov-fail-under=$COVERAGE_GOAL` (:409), so the coverage gate that is this script's whole purpose is set to the decimal digits (often 0), effectively disabling enforcement.",
      "recommendation": "Have get_coverage_goal emit ONLY the numeric goal on stdout (route diagnostics to stderr), and parse with a decimal-aware pattern, e.g. `grep -oE '[0-9]+(\\.[0-9]+)?' | tail -1`.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-012",
      "location": "setup.py:8-21",
      "class": "design_defect",
      "severity": "high",
      "evidence": "setup.py install_requires omits the project's core runtime deps — no hydra-core, no lightning/pytorch-lightning, no omegaconf — even though Hydra @main and PyTorch Lightning are central (pyproject.toml lists hydra-core==1.3.2 at :32 and lightning>=2.2 at :35). Installing via setup.py would leave `import hydra`/Lightning failing. Conversely it pins GUI/vendored-tool deps as CORE requirements: opencv-python, Pillow, mss, pyautogui, and PySimpleGUI (:16-20), the latter not even used (gui_launcher.py imports dearpygui, which setup.py omits while pyproject.toml lists dearpygui>=1.10 at :31).",
      "recommendation": "Make setup.py match pyproject.toml: add hydra-core/lightning/omegaconf, drop unused PySimpleGUI, and move screen_finder GUI deps to an optional extra rather than core install_requires (or remove setup.py entirely).",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0021",
      "location": ".augement_code_rules:1",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The file is entirely a Task Master AI dev-workflow rulebook centered on `scripts/dev.js` and its supposed modular split into `scripts/modules/` (e.g. lines 312-321), but scripts/modules/ does not exist (verified by ls) so the documented `node scripts/dev.js`/`task-master` workflow is non-functional. The content is unrelated to the RNA prediction pipeline that is the project's stated intent (audit/01-understanding.md:6).",
      "recommendation": "Remove or relocate the Task Master rules; if retained, fix references to the missing scripts/modules/ structure.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0016",
      "location": ".coveragerc:11",
      "class": "design_defect",
      "severity": "medium",
      "evidence": ".coveragerc sets `[report] fail_under = 0`, meaning coverage enforcement is effectively disabled, yet .coverage_config.json declares base_coverage 80, max_coverage 95, current_coverage 89.99 and a phased target schedule (.coverage_config.json:2-39). The governance described in .coverage_config.json is never enforced by the actual coverage tool config.",
      "recommendation": "Either wire .coverage_config.json thresholds into CI/.coveragerc fail_under, or remove the unenforced coverage-config JSON to avoid implying a gate that does not exist.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-009",
      "location": ".coveragerc:11",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "`[report] fail_under = 0` disables any coverage gate, while .coverage_config.json:3-5 declares `base_coverage: 80`, `current_coverage: 89.99`, `max_coverage: 95` and phase targets up to 95/98 (.coverage_config.json:31-38), and .windsurfrules:24 states 'Aim for near 100% test coverage'. The actual enforced threshold (0) contradicts the documented coverage policy, so coverage can drop arbitrarily without CI failing.",
      "recommendation": "Set fail_under to the intended floor (e.g. 80) or wire the phase-based thresholds from .coverage_config.json into the coverage gate; otherwise the coverage policy is unenforced.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0020",
      "location": ".env.example:1-14",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": ".env.example documents only Task Master / LLM-CLI variables (ANTHROPIC_API_KEY, PERPLEXITY_API_KEY, MODEL=claude-3-7-sonnet-20250219, MAX_TOKENS, DEFAULT_SUBTASKS, etc.) — the vendored Node 'Task Master' tool's config — and contains zero variables relevant to the RNA structure-prediction pipeline. A new user copying this file gets no guidance for the actual product.",
      "recommendation": "Replace with env vars the RNA pipeline actually reads (e.g. HF token / cache, device, data paths), or move this file under the Task Master tooling and label it as such.",
      "status": "survived"
    },
    {
      "id": "s2re4-006",
      "location": ".gitattributes:1 (DNA_bert_3.zip)",
      "class": "design_defect",
      "severity": "medium",
      "evidence": ".gitattributes declares '*.zip filter=lfs diff=lfs merge=lfs -text', and DNA_bert_3.zip is committed as a Git-LFS pointer (on-disk content is the 134-byte text 'version https://git-lfs.github.com/spec/v1 ... size 321782167', a 321MB object). Cloning without git-lfs installed yields the pointer stub instead of the model archive, and all *.zip assets (root DNA-BERT model plus the rna_predict/dataset/preprocessing DSSR binary zips) silently fail to materialize. No README/CONTRIBUTING/setup instruction documents the git-lfs requirement, so preprocessing/model loading breaks with confusing errors for fresh clones/CI.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c0l1-001",
      "location": ".github/init.sh:28-63",
      "class": "security",
      "severity": "medium",
      "evidence": "download_template() runs `git clone \"${template_url}\"` where template_url=\"https://github.com/rochacbruno/${template}-project-template\" (:35) with `template` taken from the -t flag or an interactive read (:7,:14). Line 63 then executes the just-downloaded code unconditionally: `./.github/templates/${template}/apply.sh -a ... -d ...`. This is download-and-execute of remote, third-party code over the network with no integrity/pin (no commit SHA, no checksum). The `template` value is also interpolated directly into both the clone URL and the executed path, so a crafted value (e.g. containing path traversal or `;`) flows into the executed path. Invoked via `make init` (Makefile:120-122).",
      "recommendation": "Pin the template source to a specific commit/tag and verify it, restrict `template` to an allow-list (currently only 'flask' is advertised at :13), validate/sanitize the value before using it in URLs or paths, and require explicit confirmation before executing downloaded apply.sh.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-005",
      "location": ".github/workflows/main.yml:50-70",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Two near-identical ruff auto-fix steps ('Run ruff check (auto-fix)' and 'Fix auto-fixable lint issues') run the same `ruff check --fix --unsafe-fixes` then commit and push to the branch. Both run on `pull_request` events (main.yml:10-11); auto-committing/force-style pushing during CI mutates the branch under test, and pushes from fork PRs will fail (masked by `continue-on-error: true`), so failures are silently swallowed and the duplication is wasted work.",
      "recommendation": "Collapse to a single fix step, gate the commit/push on push-to-main (not pull_request), and drop continue-on-error so genuine failures surface.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0007",
      "location": ".github/workflows/rename_project.yml:3",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The 'Rename the project from template' workflow triggers `on: [push]` with `permissions: write-all` and force-pushes a commit on every push (it is gated only by the presence of .github/template.yml). This is leftover python-project-template scaffolding unrelated to the RNA pipeline; if .github/template.yml ever reappears it would sed-rewrite and force-push every tracked file (.github/rename_project.sh:24-33).",
      "recommendation": "Delete the rename_project workflow and rename_project.sh now that the project is no longer a template, or restrict the trigger to workflow_dispatch only.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-007",
      "location": ".github/workflows/rename_project.yml:30",
      "class": "bug",
      "severity": "medium",
      "evidence": "The 'Is this still a template' step uses `echo \"::set-output name=is_template::...\"`. The `::set-output` workflow command was deprecated and disabled by GitHub Actions (mid-2023); it no longer sets step outputs. Consequently `steps.is_template.outputs.is_template` (referenced at rename_project.yml:33) is always empty, so the rename step never fires. The workflow also triggers `on: [push]` (line 3) for every push to this already-renamed repo, and the git-auto-commit-action with `push_options: --force` (lines 38-42) runs on every push regardless.",
      "recommendation": "Replace `::set-output` with `echo \"is_template=...\" >> \"$GITHUB_OUTPUT\"`. Since the project is no longer a template, consider deleting rename_project.yml/.sh entirely to remove the force-push-on-every-push hazard.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-003",
      "location": ".github/workflows/rename_project.yml:5,3,38-43",
      "class": "security",
      "severity": "medium",
      "evidence": "`permissions: write-all` (:5) grants the GITHUB_TOKEN full write scope to all repository resources, and the workflow triggers on every push (`on: [push]`, :3). The final step always runs stefanzweifel/git-auto-commit-action@v5 with `push_options: --force` (:42), force-pushing a commit on every push. The rename body is gated on `.github/template.yml` existing (:33), and that file is absent (verified: `.github/template.yml` does not exist), so the broad-permission + force-push step now fires on every push with no useful work — an over-privileged, surprising mutation surface.",
      "recommendation": "Remove this template-bootstrap workflow now that the project is no longer a template (template.yml is gone), or scope `permissions` to the minimum (`contents: write` only) and drop the unconditional `--force` auto-commit.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0035",
      "location": ".gitignore:220-221",
      "class": "design_defect",
      "severity": "medium",
      "evidence": ".gitignore lists `package-lock.json` and `package.json` as ignored, but package.json is a tracked, inventoried config file (audit/01-understanding.md:175) defining the Task Master npm scripts. Ignoring a file that is intentionally version-controlled is contradictory and will cause newly-introduced copies to be silently skipped.",
      "recommendation": "Remove package.json (and package-lock.json if it is meant to be tracked) from .gitignore, or untrack them deliberately.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0023",
      "location": ".roomodes:6",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "The Test mode roleDefinition instructs running `./rna_predict/scripts/run_failing_tests.sh` 'its way faster', but that path does not exist; the actual script is at scripts/test_utils/run_failing_tests.sh (verified: find run_failing_tests.sh → ./scripts/test_utils/run_failing_tests.sh, and rna_predict/scripts/ contains only __init__.py and hypot_test_gen.py). Following the instruction yields 'No such file or directory'.",
      "recommendation": "Correct the path to scripts/test_utils/run_failing_tests.sh.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0025",
      "location": ".windsurfrules:6-7",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The project rules record a known unresolved pipeline defect: 'inconsistent atom counts between stages, with Stage C producing 21 atoms total while Stage D expects 44 atoms per residue.' This documents a real intent/contract mismatch between Stage C output and Stage D input that, per the project's own notes, breaks the C->D handoff in the full pipeline.",
      "recommendation": "Reconcile the atom-count contract between Stage C reconstruction output and Stage D diffusion input (verify against stageC/stageD source) and update or close out this rule once fixed.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-005",
      "location": "Containerfile:1",
      "class": "security",
      "severity": "medium",
      "evidence": "`FROM python:3.7-slim`. Python 3.7 reached end-of-life in June 2023 and no longer receives security patches; the base image carries unpatched OS/interpreter CVEs. It also contradicts intended runtime: pyproject.toml:11 sets `requires-python = \">=3.10\"`, so `pip install .` (Containerfile:4) would fail on this base anyway.",
      "recommendation": "Use a supported, patched base image consistent with requires-python (e.g. python:3.11-slim) and rebuild regularly to pick up security updates.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-014",
      "location": "Makefile:47-48",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`test: lint` makes the test target depend on the `lint` target, which runs `ruff check --fix --unsafe-fixes rna_predict/ tests/` and `mypy --ignore-missing-imports rna_predict/` (Makefile:33-35). Both return non-zero on unfixable lint issues / type errors, so `make test` aborts before any test runs. CI invokes `make test` (.github/workflows/main.yml:101), so a mypy/ruff finding fails the test job for reasons unrelated to test outcomes, and `--unsafe-fixes` mutates source as a side effect of running tests.",
      "recommendation": "Decouple linting from testing: have `test` run pytest only, and run lint as a separate CI step/target (the linter job already exists in main.yml).",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0013",
      "location": "mypy.ini:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "mypy configuration is split across two files: mypy.ini (full config: python_version, per-module overrides for deepspeed/protenix/scipy/torch/numpy) and pyproject.toml [tool.mypy] (overrides for cv2 and dearpygui at pyproject.toml:94-103). mypy reads mypy.ini in preference to pyproject.toml when both exist, so the cv2/dearpygui ignore_missing_imports overrides in pyproject are silently ignored, allowing missing-import errors for those modules.",
      "recommendation": "Consolidate all mypy config into one file (move the cv2/dearpygui overrides into mypy.ini, or delete mypy.ini and keep everything in pyproject [tool.mypy]).",
      "status": "survived"
    },
    {
      "id": "s2c0l0-018",
      "location": "package.json:2-7",
      "class": "bug",
      "severity": "medium",
      "evidence": "All npm scripts (dev/list/generate/parse-prd) invoke `node scripts/dev.js`, but scripts/dev.js imports from './modules/commands.js' while `scripts/modules/` does not exist (verified: `ls scripts/modules` -> 'modules missing'; corroborated by audit/01-understanding.md:14). Every declared npm script therefore fails at runtime with a module-resolution error.",
      "recommendation": "Restore the missing scripts/modules/ directory or repoint dev.js, or remove the dead npm scripts/Task Master tooling if it is no longer used.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0037",
      "location": "package.json:3-6",
      "class": "bug",
      "severity": "medium",
      "evidence": "npm scripts (dev/list/generate/parse-prd) all shell out to `node scripts/dev.js`, and scripts/dev.js imports from './modules/commands.js' (audit/01-understanding.md:14) but scripts/modules/ does not exist (verified by ls). Every npm script therefore fails at runtime with a module-not-found error. The whole package.json belongs to the vendored Task Master tool, unrelated to the RNA pipeline.",
      "recommendation": "Restore the missing scripts/modules/ tree or remove the Task Master Node tooling (package.json + dev.js) if it is not part of the shipped product.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-010",
      "location": "pyproject.toml:14-38",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Core runtime `dependencies` include unrelated GUI/automation packages for the vendored screen_finder_app (pyautogui>=0.9.54, opencv-python>=4.8.0, mss>=9.0.1, dearpygui>=1.10, Pillow>=10.0.0) and developer tooling (black>=25.1.0, isort>=6.0.1, ruff>=0.11.2, pytest>=8.3.5) as hard install requirements. Per the Stage-1 intent (audit/01-understanding.md:9), screen_finder is 'vendored tooling unrelated to RNA structure prediction'. Every consumer of the rna-predict package is forced to install heavyweight GUI stacks and lint/test tools at runtime.",
      "recommendation": "Move dev tools to the dev optional-dependencies group and the GUI/automation packages to a dedicated optional extra (e.g. [project.optional-dependencies].screenfinder); keep core dependencies to true runtime needs.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0014",
      "location": "pytest.ini:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "pytest configuration is split: pytest.ini provides testpaths/markers/addopts, while pyproject.toml [tool.pytest.ini_options] (pyproject.toml:88-90) sets asyncio_mode=\"strict\" and asyncio_default_fixture_loop_scope. pytest uses pytest.ini in preference to pyproject when both exist, so the asyncio settings are ignored — pytest-asyncio (a declared dependency) will not run in strict mode as intended.",
      "recommendation": "Move the asyncio_mode/loop-scope settings into pytest.ini, or drop pytest.ini and keep all pytest config in pyproject.toml.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-011",
      "location": "requirements.txt:1-21",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "requirements.txt and pyproject.toml [project.dependencies] are divergent, conflicting sources of truth: requirements.txt pins `torch>=2.0.0` (no upper bound) vs pyproject.toml:25 `torch>=2.0.1, <2.6.0`; `transformers>=4.30.0` vs pyproject.toml:26 `transformers>=4.49.0`; `biopython>=1.81` vs pyproject.toml:15 `biopython>=1.83`. requirements.txt also lists packages absent from pyproject (PySimpleGUI, py-cpuinfo, psutil, pandas, tqdm, datasets-less) while pyproject lists datasets/einops/lxml/mdanalysis/tensorboard absent from requirements.txt. A `pip install -r requirements.txt` yields a materially different environment than `pip install .`.",
      "recommendation": "Designate one canonical dependency source (pyproject.toml) and either delete requirements.txt or generate it from the project metadata; reconcile the version bounds.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0027",
      "location": "requirements.txt:14-18",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "requirements.txt lists GUI/automation packages pyautogui, opencv-python, Pillow, mss, PySimpleGUI as runtime dependencies of the RNA pipeline. These belong to the unrelated vendored scripts/screen_finder_app GUI (audit/01-understanding.md:9,36), not to sequence-to-structure prediction, bloating and risking the install (PySimpleGUI now requires a license server).",
      "recommendation": "Move screen-finder GUI deps into an optional extra (e.g. [screen_finder]) and keep the core requirements limited to pipeline needs.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-012",
      "location": "requirements.txt:18",
      "class": "bug",
      "severity": "medium",
      "evidence": "`PySimpleGUI` is listed as an unpinned dependency. PySimpleGUI was relicensed and the open-source releases were removed from PyPI (the package now requires a private server / paid key), so `pip install -r requirements.txt` is liable to fail or pull an incompatible release on a clean machine. It is also a GUI dependency unrelated to the RNA pipeline.",
      "recommendation": "Remove PySimpleGUI from runtime requirements (or pin a known-installable version and move it to an optional extra) since it is only relevant to the unrelated GUI tooling.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0028",
      "location": "requirements.txt:2",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Dependency declarations are fragmented and divergent across four+ sources of truth: requirements.txt (torch>=2.0.0 unbounded, PySimpleGUI, protenix unpinned), setup.py install_requires (torch>=2.0.0,<2.6.0, PySimpleGUI, version 1.0.0), pyproject.toml dependencies (torch>=2.0.1,<2.6.0, dearpygui instead of PySimpleGUI, protenix>=0.4.4), and pyproject [project.optional-dependencies].dev plus a separate [dependency-groups].dev with different contents. The GUI lib even differs (PySimpleGUI vs dearpygui).",
      "recommendation": "Pick pyproject.toml as the single source of dependency truth; delete setup.py/requirements*.txt or generate them from pyproject, and reconcile the GUI library choice.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-validate-metadata-noop",
      "location": "rna_predict/conf/config_schema.py:222-227,435-441,356-362",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Numerous dataclass fields declare metadata={'validate': lambda x: ...} (e.g. StageAConfig.dropout 222-227, PairformerBlockConfig.dropout 435-441, LoRAConfig.dropout 356-362, many others). OmegaConf/dataclasses never invoke metadata['validate'], so these range checks are decorative and never enforced. Only StageAConfig, DeviceConfig and StageCConfig implement real __post_init__ validation; all other 'validate' metadata gives a false impression of input validation (e.g. out-of-range dropout/heads pass silently).",
      "recommendation": "Either remove the misleading 'validate' metadata or implement __post_init__ checks (or a shared validator) that actually call these predicates.",
      "status": "survived"
    },
    {
      "id": "s2c1l1-003",
      "location": "rna_predict/conf/config_schema.py:270-272",
      "class": "security",
      "severity": "medium",
      "evidence": "StageAConfig.checkpoint_url defaults to a hardcoded third-party Dropbox URL ('https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1'), duplicated in conf/model/stageA.yaml:12, with checkpoint_zip_path defaulting to 'RFold/checkpoints.zip' and checkpoint_path to a '.pth' file (lines 266-277). Per the audit map (01-understanding.md:22) Stage A downloads/extracts this checkpoint and loads it; a PyTorch '.pth' is an unpickled artifact, so an unauthenticated download from an externally-controlled file-host (no checksum/signature pinning) is an unsafe-download / supply-chain vector that can lead to arbitrary code execution on torch.load of the downloaded weights.",
      "recommendation": "Pin the checkpoint by content hash and verify it after download; prefer a versioned, integrity-checked source; load weights with weights_only=True (torch>=2.0) and document the trust boundary. Do not ship a mutable third-party share link as the default.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-devmgmt-unregistered-resolver",
      "location": "rna_predict/conf/device_management/default.yaml:2",
      "class": "bug",
      "severity": "medium",
      "evidence": "primary: ${device:cpu} uses OmegaConf custom-resolver syntax (resolver name 'device', argument 'cpu'). No OmegaConf.register_new_resolver call exists anywhere in the repository (grep for register_new_resolver returned no matches), so resolving this node raises UnsupportedInterpolationType. The other YAMLs use plain interpolation ${device} which references the top-level device key and is fine; this colon form is a latent failure for any composition that pulls in device_management.",
      "recommendation": "Change to plain interpolation ${device} (with a default elsewhere) or register a 'device' resolver; otherwise selecting the device_management group crashes config resolution.",
      "status": "survived"
    },
    {
      "id": "s2re1-004",
      "location": "rna_predict/conf/model/stageA.yaml:12",
      "class": "security",
      "severity": "medium",
      "evidence": "checkpoint_url is a personal Dropbox share link: \"https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1\". run_stageA.py:188-190 fetches this via download_file() (urllib.request.urlopen, run_stageA.py:70) and unzips it; the resulting checkpoint is then torch.load-ed (unsafe, see s2re1-002). A mutable, unauthenticated, non-content-addressed third-party URL as the canonical model-weights source is a supply-chain/single-point-of-failure risk: the owner can delete/replace it and there is no hash pinning to detect tampering.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c1l2-stageb-pairformer-internal-dim-inconsistency",
      "location": "rna_predict/conf/model/stageB_pairformer.yaml:8-32",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "stageB_pairformer.yaml mixes production and toy dimensions within one file in mutually inconsistent ways: top-level c_token=384, c_atom=128, c_pair=32 (production) but c_z=2 (toy) and c_s=0; the nested protenix_integration block then overrides c_token=2, restype_dim=2, profile_dim=2, c_atom=2, c_pair=2 — i.e. the same logical embedding (c_token) is 384 at the Pairformer level and 2 in its ProtenixIntegration sub-config. config_schema.py PairformerConfig defaults c_token=8/c_atom=4 again differ. A leftover marker '[UNIQUE-ERR-STAGEB-DEBUGLOGGING-001]' (line 93) further signals ad-hoc debug edits left in the config.",
      "recommendation": "Reconcile the Pairformer dims so token/atom/pair sizes are consistent between the block and its protenix_integration sub-config; remove the debug marker comment; document why c_s=0 is intentional or drop it.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-stagec-angle-representation-semantics",
      "location": "rna_predict/conf/model/stageC.yaml:18-19 vs rna_predict/conf/config_schema.py:767-770",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "The field 'angle_representation' is documented with two incompatible meanings. config_schema.py StageCConfig.angle_representation defaults to 'cartesian' with help 'Angle representation: cartesian or internal'. stageC.yaml sets angle_representation: 'degrees' with comment 'Expected input format (\\'degrees\\' or \\'radians\\')'. The YAML value 'degrees' is not even a member of the schema's documented value set {cartesian, internal}, so the structured-schema documentation and the actual config describe different concepts (coordinate representation vs angle units).",
      "recommendation": "Decide what angle_representation actually controls, unify the help text, and constrain the value set (e.g. an Enum or __post_init__ validator) so 'degrees' vs 'cartesian' cannot silently coexist.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-stageD-arch-layer-count-inconsistency",
      "location": "rna_predict/conf/model/stageD.yaml:34-35,87-88 vs rna_predict/conf/model/stageD_diffusion.yaml:15-23",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "stageD.yaml hardcodes model_architecture.num_layers: 6 and num_heads: 8 and test_residues_per_batch: 25 (production scale) while every embedding dimension it pulls from stageD_diffusion is the toy size (c_token=8, c_s=8, c_z=4, c_atom=4; stageD_diffusion.yaml:16-22) and stageD_diffusion's own transformer is n_blocks: 2 / n_heads: 2 (lines 62-63). The result is an internally contradictory Stage D config: an 8-dim token model asked to run 6 layers / 8 heads in one place but 2 blocks / 2 heads in another, with num_heads=8 not dividing several of the reduced dims cleanly.",
      "recommendation": "Source num_layers/num_heads/test_residues_per_batch from the same stageD_diffusion block via interpolation (as the dims are) rather than hardcoding production-scale literals, so the architecture stays self-consistent.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-predict-yaml-hardcoded-ckpt",
      "location": "rna_predict/conf/predict.yaml:13",
      "class": "bug",
      "severity": "medium",
      "evidence": "predict.yaml sets checkpoint_path: /Users/tomriddle1/RNA_PREDICT/outputs/checkpoints/last.ckpt — an absolute developer-specific path that does not exist on any other machine. The README-recommended inference entry (predict.py) composes this config, so a default run loads a nonexistent checkpoint path.",
      "recommendation": "Use a relative path (e.g. outputs/checkpoints/last.ckpt) or null with an explicit override requirement.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-testdata-embedding-dims-stale",
      "location": "rna_predict/conf/test_data.yaml:25-28",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "test_data.yaml embedding_dims sets s_trunk: 384, z_trunk: 128, s_inputs: 449 with comment 's_inputs ... must match c_s_inputs', but the reduced schema/YAMLs put c_s_inputs at 8 (config_schema.py:820-823; stageD_diffusion.yaml:18). So the asserted invariant 'must match c_s_inputs' is violated (449 vs 8), and these production-size test dims contradict the toy model dims used everywhere else, making the test fixture inconsistent with the model it feeds.",
      "recommendation": "Align test_data.embedding_dims with the actual configured c_s/c_z/c_s_inputs (interpolate from the shared/model config rather than hardcoding 384/128/449), or remove the misleading 'must match' comment.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-atom-lists-incomplete-source-of-truth",
      "location": "rna_predict/dataset/atom_lists.py:1-9",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The module header claims to be the 'Single source of truth for atom ordering and max atoms per residue', but STANDARD_ATOMS lists only 22 atoms (A/G backbone+base, with the comment '# Extend as needed for all bases'), so MAX_ATOMS_PER_RES = 22. The rest of the system assumes ~44 atoms/residue: config_schema RNAConfig.atoms_per_residue=44, TestDataConfig.atoms_per_residue=44 (config_schema.py:1271-1274,1362-1365), default.yaml:27 atoms_per_residue: 44. loader.py:309 iterates STANDARD_ATOMS to fill atom features, so the canonical-atom enumeration is truncated/inconsistent with the declared per-residue atom count.",
      "recommendation": "Complete STANDARD_ATOMS to the full canonical RNA atom set (or document why 22 is intended) and reconcile MAX_ATOMS_PER_RES with the atoms_per_residue=44 used throughout config and the loader.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-collate-atomnames-inconsistent",
      "location": "rna_predict/dataset/collate.py:55-57,107-108",
      "class": "bug",
      "severity": "medium",
      "evidence": "rna_collate_fn batches 'atom_names'/'residue_indices' inconsistently across batch sizes. For a single-item batch each is wrapped as [v] (a list containing one per-sample list) at lines 55-57, preserving per-sample nesting. For a multi-item batch (lines 107-108) any value whose first element is a list is FLATTENED across all samples ([item for sublist in vs for item in sublist]), collapsing per-sample boundaries. A consumer indexing batch['atom_names'][sample_i] gets a per-sample list when batch_size==1 but a single merged flat list when batch_size>1.",
      "recommendation": "Treat 'atom_names'/'residue_indices' explicitly as a list-of-lists in both branches (out[k] = vs) rather than flattening sublists, matching the single-item behavior.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-loader-except-undef-var",
      "location": "rna_predict/dataset/loader.py:469-472",
      "class": "bug",
      "severity": "medium",
      "evidence": "The broad except handler in _load_angles references structure_file and selected_chain_id in the warning string (line 471). Both are local variables assigned only inside the try block (selected_chain_id at lines 402/405/408, structure_file at line 411). If the exception is raised earlier — e.g. during backend resolution at line 361 (getattr on cfg) — these names are unbound, so the except block raises NameError, masking the original error and discarding the intended zeros fallback at line 472.",
      "recommendation": "Initialize structure_file=None and selected_chain_id=None before the try block (or use locals().get) so the fallback path and warning never raise on early failures.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-loader-cuda-numworkers",
      "location": "rna_predict/dataset/loader.py:84,131-166 + rna_predict/conf/data/default.yaml:10",
      "class": "bug",
      "severity": "medium",
      "evidence": "RNADataset.__getitem__ allocates every tensor directly on self.device (torch.device(cfg.device), which may be 'cuda' or 'mps') — e.g. residue_mask (line 131), coords/embeddings via _load_atom_features, angles (line 166). data/default.yaml sets num_workers: 8 (config_schema DataConfig defaults to 0 and explicitly comments 'set to 0 for debugging device mismatch'). Creating CUDA tensors inside forked DataLoader worker processes triggers 'Cannot re-initialize CUDA in forked subprocess' errors / undefined behavior; this couples a config default with __getitem__ device placement in a way that breaks multi-worker GPU loading.",
      "recommendation": "Build tensors on CPU in __getitem__ and move to device in the training loop / collate, or force num_workers=0 whenever cfg.device is non-CPU.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-angles-snoop-decorator",
      "location": "rna_predict/dataset/preprocessing/angles.py:11,14",
      "class": "perf",
      "severity": "medium",
      "evidence": "extract_rna_torsions is decorated with @snoop (line 14, `import snoop` line 11), a line-by-line execution tracer. Left enabled in the production extraction entry point it emits per-line trace output for every residue/structure processed (called from RNADataset._load_angles and compute_ground_truth_angles), drastically slowing dataset loading and flooding logs, and makes `snoop` a hard runtime dependency.",
      "recommendation": "Remove the @snoop decorator (and the import) or gate it behind a debug flag.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-angles-import-side-effect",
      "location": "rna_predict/dataset/preprocessing/angles.py:420-449",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Module import has a heavy, failure-prone side effect: at import time angles.py unzips a bundled DSSR distribution (_DSSR_ZIP) into a 'dssr' directory, selects a platform-specific nested zip, renames/chmods the binary, and raises RuntimeError on any unsupported platform (lines 425-449). Importing this module (e.g. via loader.py:356 'from ...angles import extract_rna_torsions') performs disk extraction and can hard-fail if the zip is missing or the OS is unrecognized, even when the DSSR backend is never used.",
      "recommendation": "Move the DSSR extraction into a lazily-invoked function called only when backend=='dssr' is actually selected, and surface failures as a handled error rather than an import-time exception.",
      "status": "survived"
    },
    {
      "id": "s2c1l1-002",
      "location": "rna_predict/dataset/preprocessing/angles.py:424-449",
      "class": "security",
      "severity": "medium",
      "evidence": "Importing this module triggers, on first use, extraction and `os.chmod(_DSSR_BIN, 0o755)` of a third-party executable (x3dna-dssr) bundled as `dssr-basic-linuxMacWindows-v2.5.3.zip`, which is then executed via `subprocess.run([_DSSR_BIN, ...])` at line 470. There is no integrity/authenticity check on the bundled binary before it is made executable and run. Because this is top-level code (runs at `import rna_predict.dataset.preprocessing.angles`), any code path importing the module — including RNADataset._load_angles (loader.py:356) — silently materializes and prepares an external binary for execution. A compromised or substituted vendored zip yields arbitrary code execution under the user's account.",
      "recommendation": "Gate binary extraction/execution behind an explicit opt-in (config flag), verify a pinned checksum of the binary before chmod/exec, and move the side-effecting setup out of import scope into an explicitly-called function.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-angles-import-side-effect-chmod",
      "location": "rna_predict/dataset/preprocessing/angles.py:424-449",
      "class": "bug",
      "severity": "medium",
      "evidence": "At module import time the code extracts a DSSR zip and then unconditionally calls os.chmod(_DSSR_BIN, 0o755) at line 449. _DSSR_BIN is only created if the rename loop (lines 445-448) finds a top-level file starting with 'x3dna-dssr' that is X_OK. If the nested archive extracts the binary into a subdirectory (the inventory shows the macOS bundle as dssr-basic-macOS-v2.5.3/x3dna-dssr) or the X_OK check fails (e.g. Windows .exe), the loop never renames a file, _DSSR_BIN does not exist, and os.chmod raises FileNotFoundError — crashing on mere import of the module.",
      "recommendation": "Only chmod when a binary was actually located; search recursively (os.walk) for the binary, and wrap the extraction in a function invoked lazily rather than at import.",
      "status": "survived"
    },
    {
      "id": "s2c1l1-001",
      "location": "rna_predict/dataset/preprocessing/angles.py:425-449",
      "class": "security",
      "severity": "medium",
      "evidence": "Module-level (import-time) code unpacks bundled archives with `zipfile.ZipFile(_DSSR_ZIP).extractall(_DSSR_DIR)` (line 430) and `zipfile.ZipFile(nested_path).extractall(_DSSR_DIR)` (line 443) with no member-name sanitization. `extractall` honors absolute paths and `../` traversal entries in the zip ('Zip Slip'), so any maliciously crafted or tampered DSSR archive could write files outside `_DSSR_DIR` (e.g. overwrite arbitrary files in the package tree or user home). The extraction then `os.rename`s the discovered binary and `os.chmod(_DSSR_BIN, 0o755)` (line 449), and the binary is later run via subprocess (line 470). Intent (01-understanding.md:17, audit map) is benign torsion-angle preprocessing, so unsanitized archive extraction is a defect relative to that intent.",
      "recommendation": "Validate each ZipInfo member before extraction (reject absolute paths and any normalized path that escapes the destination dir), or extract members individually with a sanitized join. Verify archive integrity (checksum/signature) before trusting it, and avoid doing extraction/chmod as an import-time side effect.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-ggt-hardcoded-conf-path",
      "location": "rna_predict/dataset/preprocessing/compute_ground_truth_angles.py:53",
      "class": "bug",
      "severity": "medium",
      "evidence": "main() hardcodes the Hydra config path to '/Users/tomriddle1/RNA_PREDICT/rna_predict/conf' (line 53). On any other machine hydra.initialize(config_path=...) fails; the surrounding try/except (line 57) swallows it, silently dropping all config-derived backend/chain selection so the CLI always falls back to argparse defaults rather than honoring the project config.",
      "recommendation": "Derive the conf path relative to the package (e.g. importlib.resources / Path(__file__) parents) instead of an absolute developer path.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-interface-deadcode-after-raise",
      "location": "rna_predict/interface.py:46-54",
      "class": "bug",
      "severity": "medium",
      "evidence": "In main()'s predictor-init except block, line 46 executes `raise ValueError(...) from e` unconditionally, so the diagnostic code at lines 48-54 (printing stageB_torsion/stageB_pairformer config and a second `raise`) is unreachable dead code. The intended debug output on initialization failure never runs.",
      "recommendation": "Remove the redundant early raise (line 46) or move the diagnostic prints before the raise so they actually execute.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-datautils-concat-empty",
      "location": "rna_predict/kaggle/data_utils.py:140-154",
      "class": "bug",
      "severity": "medium",
      "evidence": "process_test_sequences collects per-sequence frames in `frames`, appending only on success and logging.error on failure (lines 146-151). If every sequence raises, `frames` stays empty and `pd.concat(frames, ignore_index=True)` at line 153 raises 'ValueError: No objects to concatenate', crashing the whole submission run instead of producing an empty/partial submission or a clear error.",
      "recommendation": "Guard for empty frames (e.g. if not frames: raise a descriptive error or write an empty submission with the required columns) before pd.concat.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-hardcoded-external-drive-data-path",
      "location": "rna_predict/kaggle/data_utils.py:36 and rna_predict/kaggle/rna_predict.py:77",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The non-Kaggle 'local environment' data root is hardcoded to one developer's removable volume: BASE_INPUT_ROOT_EXTERNAL_DRIVE = pathlib.Path('/Volumes/Totallynotaharddrive/RNA_structure_PREDICT/kaggle/') in both data_utils.py:36 and rna_predict.py:77. For any other contributor running the Kaggle harness locally, load_kaggle_data() raises FileNotFoundError. There is no config/env override path for the local data root.",
      "recommendation": "Source the local data root from config (cfg.data.root_dir / a DATA_ROOT env var) with a repo-relative default such as ./data/kaggle, instead of a hardcoded /Volumes mount.",
      "status": "survived"
    },
    {
      "id": "s2c1l1-004",
      "location": "rna_predict/kaggle/kaggle_env.py:40-63",
      "class": "security",
      "severity": "medium",
      "evidence": "install_wheels() builds `--find-links` from `/kaggle/input` and every subdirectory under it (lines 46-47), then runs `pip install --no-index ... <pkg>` via subprocess (lines 58-61) for a hardcoded package list. On Kaggle, `/kaggle/input` holds attached datasets which can be arbitrary user-supplied content (e.g. when a notebook is forked or a malicious dataset is attached); allowing pip to resolve those package names from attacker-controllable local wheels enables installation of trojaned wheels (arbitrary code execution at install time). setup_kaggle_environment() similarly pip-installs wheels discovered by path/glob from `/kaggle/input` (lines 217-263). Gated by is_kaggle() but still executes whenever running in that environment.",
      "recommendation": "Restrict find-links to a single trusted, integrity-verified directory; pin exact wheel filenames + hashes (pip --require-hashes); avoid globbing untrusted dataset dirs as a package source.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-legacy-modeling-nonrunnable",
      "location": "rna_predict/kaggle/legacy_feature_engineering_and_modeling.py:17-18,189-204",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "This is a flattened notebook (`# %%` cells) executed at import/module top level that references undefined globals: train_sequences, validation_sequences, train_labels, validation_labels (lines 17-18), test_sequences (line 218), X_full (line 257) — none are defined or imported, so the module raises NameError immediately if run/imported. Moreover the modeling half is gone: cells 8-9 are stubs with TODOs 'param_dist removed in cleanup pass 1', 'get_best_xgb removed', 'all related code has now been removed' (lines 189-204), so despite the filename '..._and_modeling', no model is ever trained or used.",
      "recommendation": "Either convert this back into a guarded function/script that takes its inputs as parameters (no top-level free variables) and restore or delete the modeling section, or remove the file and keep it as a notebook artifact outside the importable package.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-kaggle-runner-filename-and-paths",
      "location": "rna_predict/kaggle/rna_predict.py:1-23,314",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "The file is named rna_predict.py (underscore) but its header comment is '# rna-predict.py' (line 1) and every HOW-TO-RUN example invokes a non-existent 'rna_predict/kaggle/rna-predict.py' (hyphen) (lines 12,18,20) plus a trailing example at line 314. The examples also hardcode developer-specific absolute paths '/Users/tomriddle1/.local/bin/uv' and '/Users/tomriddle1/RNA_PREDICT/rna_predict/conf'. A user copy-pasting the documented command runs nothing.",
      "recommendation": "Fix the filename references to rna_predict.py and replace the /Users/tomriddle1 absolute paths with portable relative invocations (e.g. 'uv run rna_predict/kaggle/rna_predict.py --config-path ../conf').",
      "status": "survived"
    },
    {
      "id": "s2c1l2-kaggle-train-cfg-hydra-access",
      "location": "rna_predict/kaggle/rna_predict.py:270",
      "class": "bug",
      "severity": "medium",
      "evidence": "run_training_pipeline reads cfg.hydra.run.dir inside the is_kaggle() branch, but the 'hydra' node is not part of the composed application cfg by default (the runtime cfg keys in combined_pipeline_output.txt:1575 contain no 'hydra'). Accessing cfg.hydra.run.dir will raise ConfigAttributeError/AttributeError, breaking Kaggle 'train' mode at exactly the point it tries to verify the output directory.",
      "recommendation": "Use hydra.core.hydra_config.HydraConfig.get().run.dir (as conf/utils.get_run_dir already does) instead of cfg.hydra, or guard the access with a presence check.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-kaggle-cfg-hydra-rundir",
      "location": "rna_predict/kaggle/rna_predict.py:270-271",
      "class": "bug",
      "severity": "medium",
      "evidence": "Inside @hydra.main main() -> run_training_pipeline, line 270 reads cfg.hydra.run.dir. Under hydra.main the job config does not contain the 'hydra' node (it is stripped from the composed cfg and exposed only via HydraConfig.get()), so cfg.hydra.run.dir raises ConfigAttributeError. This path is reached on Kaggle training (is_kaggle() and mode=train), aborting training during the run-dir check.",
      "recommendation": "Use hydra.core.hydra_config.HydraConfig.get().run.dir instead of cfg.hydra.run.dir.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-submission-validator-sysexit",
      "location": "rna_predict/kaggle/submission_validator.py:32-35",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "run_sanity_checks is imported and called as a library function by the Kaggle harness (rna_predict.py:190), but on a missing input file it calls sys.exit(...) at line 35 (the code comment itself notes 'Or raise an error'). sys.exit terminates the entire process rather than letting the caller handle the condition, so a missing test/submission CSV kills the whole pipeline run instead of being logged/handled.",
      "recommendation": "Raise FileNotFoundError (or return a status) instead of sys.exit so callers can decide how to react.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-merger-ignores-inputs",
      "location": "rna_predict/pipeline/merger/simple_latent_merger.py:14-17,43-76",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "SimpleLatentMerger's class docstring states it merges 'adjacency, angles, single embeddings, pair embeddings, plus partial coords' but forward() only uses inputs.angles, inputs.s_emb and inputs.z_emb (lines 55-58,74); inputs.adjacency and inputs.partial_coords (LatentInputs fields) are never consumed. Additionally, the stored self.expected_dim_angles/_s/_z (lines 30-32) are never used.",
      "recommendation": "Either incorporate adjacency/partial_coords into the merge or remove them from LatentInputs and the docstring to reflect actual behavior; drop the unused expected_dim_* attributes.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-merger-dynamic-mlp-reinit",
      "location": "rna_predict/pipeline/merger/simple_latent_merger.py:63-73",
      "class": "bug",
      "severity": "medium",
      "evidence": "forward() rebuilds self.mlp with freshly-initialized random Linear layers whenever the runtime concatenated input dim differs from self.mlp[0].in_features (lines 63-71). This silently discards any learned/loaded weights on a dimension change, the new submodule's parameters are not registered with any existing optimizer (created mid-forward), and repeated calls with fluctuating dims would re-randomize every step, breaking training/inference reproducibility.",
      "recommendation": "Fix the input dimension at __init__ (from the declared dim_angles/dim_s/dim_z) and assert/validate incoming shapes, rather than reconstructing the MLP inside forward().",
      "status": "survived"
    },
    {
      "id": "s2c2l0-003",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:104-110",
      "class": "bug",
      "severity": "medium",
      "evidence": "process_seqs calls F.one_hot(nseq).float() (line 109) without num_classes. F.one_hot infers the class count from the max index present, so a sequence that lacks the highest-valued base (G=3) yields a one-hot tensor with fewer than 4 channels, breaking the fixed 4-channel assumption used by constraint_matrix (x[:,:,0..3]) and downstream conv input. Shape becomes data-dependent. (Direct caller of process_seqs within this file is unverified — RFoldModel.forward uses Seq2Map instead.)",
      "recommendation": "Pass an explicit num_classes=4: F.one_hot(nseq, num_classes=4).float() so the channel dimension is deterministic regardless of which bases appear.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-seq2dot-hardcoded-test",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:164-167",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "seq2dot() begins with `if len(seq) == 4 and seq[0]==2 and seq[1]==0 and seq[2]==3 and seq[3]==0: return \"(.))\"` — a hardcoded answer for a specific test input baked into production secondary-structure dot-bracket generation. The general logic that follows is bypassed for this input.",
      "recommendation": "Delete the hardcoded special case; if the test expects this output, fix the general algorithm to produce it or assert it in the test fixture rather than in source.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-encoder-decoder-test-identity",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:377-404",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Encoder.__init__ uses `if len(C_lst) <= 3:  # This is likely a test case` to install nn.Identity, and Encoder.forward returns `x, [x]` when `len(self.enc) <= 1` (lines 397-399). Decoder.__init__ has the same `if len(C_lst) <= 3` test branch (:413-415) returning the input unchanged. Production model topology is silently replaced with a no-op based on a heuristic guess that small channel lists 'are likely a test'.",
      "recommendation": "Drop the test-detection branches from the model classes; construct small/identity variants explicitly in tests instead of inferring intent from C_lst length.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-002",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:75-92",
      "class": "bug",
      "severity": "medium",
      "evidence": "constraint_matrix builds au_ua + cg_gc + ug_gu and returns it, but the no-sharp-loop mask produced by base_matrix(...) (defined at line 66) is never applied. The inline comments at lines 89-91 ('Apply the base_matrix constraint' / 'Apply base matrix constraints while preserving the correct pairs') claim a constraint is applied that is not. base_matrix is defined but has no caller within this file, so its intended masking effect is dropped. (Caller set outside this file is unverified.)",
      "recommendation": "Either multiply the combined pair matrix by base_matrix (constraint = (au_ua+cg_gc+ug_gu) * base_matrix(length, device)) as the comments intend, or delete the misleading comments and the unused base_matrix helper to remove the behavior/doc divergence.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfoldpred-docstring-official-claim",
      "location": "rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:2-86",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "Module/class docstrings claim it 'uses the official RFold_Model code from \"RFold/model.py\" so that pretrained checkpoints load successfully without key mismatch' and reference importing 'from RFold.model import RFold_Model'. In reality the predictor imports the local RFoldModel from RFold_code (line 29-31, 172) and loads checkpoints with `load_state_dict(ckp, strict=False)` (line 282-284), which tolerates key mismatches rather than guaranteeing exact matching. No 'RFold/' package is imported.",
      "recommendation": "Update the docstrings to describe the actual local RFoldModel and strict=False loading behaviour, and remove references to a non-existent official RFold module.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfoldpred-silent-random-weights",
      "location": "rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:244-289",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_load_checkpoint logs a warning and returns (continuing with randomly-initialized weights) when checkpoint_path is None, the file is missing, or torch.load/load_state_dict raises. Given Stage-1 intent that Stage A inference is 'Functional' (01-understanding.md:6), silently proceeding with random weights yields meaningless adjacency predictions while appearing to succeed.",
      "recommendation": "For inference mode, fail loudly (raise) when a required checkpoint cannot be loaded, or surface a clear unrecoverable error; reserve random-weight fallback for explicitly opted-in dummy/training-from-scratch paths.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-005",
      "location": "rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:281-289",
      "class": "bug",
      "severity": "medium",
      "evidence": "_load_checkpoint calls self.model.load_state_dict(ckp, strict=False) and then logs '[Load] Checkpoint loaded successfully.' unconditionally. With strict=False, a checkpoint whose keys do not match (e.g. official RFold key naming vs this refactored RFoldModel) loads ZERO parameters yet still reports success, leaving the model on random weights while predict_adjacency proceeds. The returned missing/unexpected keys from load_state_dict are discarded, so this silent failure is undetectable from logs.",
      "recommendation": "Capture the IncompatibleKeys result (missing/unexpected) returned by load_state_dict and warn/raise when the overlap is empty or below a threshold; report counts of matched vs missing keys instead of an unconditional success message.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-embedders-lazy-linear",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/embedders.py:280-294",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "InputFeatureEmbedder.extras_linear is left as None in __init__ (line 127) and constructed lazily inside forward(): `self.extras_linear = LinearNoBias(extras_dim, self.c_token).to(extras_cat.device)`. A submodule created during the first forward is absent from the module's initial state_dict and from any optimizer parameter group built before that forward, so its weights are not checkpointed/restored consistently and are not optimized if the optimizer was created at construction time.",
      "recommendation": "Determine extras_dim at construction (from restype_dim+profile_dim+1) and create extras_linear in __init__, or use nn.LazyLinear, so the parameter is registered before training/serialization.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-019",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/embedders.py:280-294",
      "class": "bug",
      "severity": "medium",
      "evidence": "InputFeatureEmbedder.forward creates self.extras_linear lazily on the first forward pass (LinearNoBias(extras_dim, c_token) at lines 285-287) with random weights, after __init__ has run and after any checkpoint load. Because it is not constructed in __init__, it is absent from the module's state_dict at load time, so a loaded checkpoint cannot populate it; at inference these projection weights remain untrained/random, and the added projection extras_proj = self.extras_linear(extras_cat) (line 294) injects noise into s_inputs. For a 'Functional' inference path this silently degrades the embedding.",
      "recommendation": "Construct extras_linear in __init__ from the known restype_dim+profile_dim+1 input size so it is part of the checkpointed state, or assert/raise if it must be lazily created at inference time.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-007",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives.py:18-33",
      "class": "bug",
      "severity": "medium",
      "evidence": "This module imports `_attention` from .primitives.attention_base (line 18-22), but attention_base.py only defines/exports `attention` (attention_base.py:10-26, __all__ has no '_attention'); and it imports broadcast_token_to_local_atom_pair, gather_pair_embedding_in_dense_trunk, rearrange_qk_to_dense_trunk, rearrange_to_dense_trunk from .primitives.data_transforms (lines 28-33), but data_transforms.py is an empty facade exporting nothing (data_transforms.py:1-16). Either import would raise ImportError. The module survives only because it is shadowed: a `primitives/` package (primitives/__init__.py) exists in the same directory and takes import precedence over `primitives.py`, so this file is never actually imported — it is dead code carrying broken imports.",
      "recommendation": "Delete primitives.py (it is shadowed by the primitives/ package), or fix its imports to source the symbols from the modules that actually define them (attention_core.attention, atom_pair_transforms.*, attention.dense_trunk.*).",
      "status": "survived"
    },
    {
      "id": "s2c2l0-010",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:102,121,126,139,143,152-156,172,176-177,194-195,199",
      "class": "perf",
      "severity": "medium",
      "evidence": "AdaptiveLayerNorm.forward and _apply_conditioning contain ~10 unconditional print(...) statements (not gated by any debug flag) that fire on every forward pass, formatting tensor shapes/devices/grad state each call. In a transformer with many AdaLN invocations per step this floods stdout and adds per-call Python/format overhead on the hot path. Additionally the method docstrings at lines 103-112 and 157-167 are placed AFTER executable print statements, so they are ordinary string expressions, not docstrings.",
      "recommendation": "Remove the print() calls or guard them behind `if self.debug_logging:` using logger.debug; move the docstrings to the first statement of each function.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-adaln-unconditional-print",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:102-199",
      "class": "perf",
      "severity": "medium",
      "evidence": "AdaptiveLayerNorm._apply_conditioning and forward emit numerous unconditional `print(f\"[DEBUG][AdaLN]...\")` statements (lines 102, 121, 126, 139, 143, 152-156, 172, 176-177, 194-195, 199) on every invocation, not gated by any debug flag. Other modules in this tree use logger.debug guarded by debug_logging; here every forward pass writes many lines to stdout, degrading performance and flooding output during inference/training.",
      "recommendation": "Replace these prints with logger.debug guarded by a debug flag, or remove them.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-009",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:182-192",
      "class": "bug",
      "severity": "medium",
      "evidence": "Inside AdaptiveLayerNorm.forward, on a feature-dim mismatch (s.shape[-1] != layernorm_s.normalized_shape) the module re-instantiates self.layernorm_s, self.linear_s, and self.linear_nobias_s with freshly random weights (lines 186-189) at runtime. During inference this silently discards the trained/checkpoint-loaded weights for those layers and replaces them with random ones, producing garbage conditioning instead of failing loudly. It also mutates module structure during forward, which is unsafe under checkpointing/DDP.",
      "recommendation": "Treat a conditioning-dimension mismatch as an error (raise with a clear message), or adapt the input tensor s to c_s instead of rebuilding trained submodules; never reinitialize learned layers inside forward().",
      "status": "survived"
    },
    {
      "id": "s2c2l2-adalnutils-print-and-silent-resize",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm_utils.py:117,342-345",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "interpolate_sequence_dim() emits an unconditional `print(f\"[DEBUG][AdaLN][interpolate_sequence_dim]...\")` (line 117). More substantively, the helpers silently coerce mismatched token dimensions: _handle_more_tokens_in_scale truncates scale/shift to a.shape[-2] (lines 342-343) and _handle_fewer_tokens_in_scale / interpolate_sequence_dim use nearest-neighbour interpolation to resize the token axis. These mask shape bugs in conditioning by altering tensor semantics rather than failing.",
      "recommendation": "Remove the print; treat token-dimension mismatches between scale/shift and the conditioned tensor as errors rather than silently interpolating/truncating, since the result is not a valid AdaLN.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attn-duplicate-divergent-stack",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_bias.py:50-83",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "There are two parallel, divergent implementations of the local-attention utilities. primitives/__init__.py wires in attention_utils.py (imports _local_attention, create_local_attn_bias, optimized_concat_split from .attention_utils, __init__.py:38-42). A second, non-wired stack (attention_bias.py, attention_local.py, attention_tensor.py, attention_types.py) redefines the same names with DIFFERENT behaviour: e.g. create_local_attn_bias here returns shape (1,1,n_queries,n_keys) with simple edge masking (attention_bias.py:47,50-83), whereas attention_utils.create_local_attn_bias returns (1,n_chunks,n_queries,n_keys) with a sliding window (attention_utils.py:108-147). The dataclasses (LocalAttentionInputs, AttentionChunkConfig, etc.) are likewise duplicated in attention_types.py vs attention_utils.py. The second stack appears dead but diverges silently, inviting wrong-import bugs.",
      "recommendation": "Consolidate to a single attention-utility implementation; delete the unused divergent modules (attention_bias/attention_local/attention_tensor/attention_types) or make them thin re-exports of the canonical one.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-012",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_module.py:92-98",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Attention._initialize_parameters zero-initializes the query, key, AND value projection weights (nn.init.zeros_ on self.to_q/to_k/to_v at lines 94-96) in addition to the output projection and gating. With to_q/to_k/to_v all zero, at initialization q=k=v=0, softmax is uniform, the attended value is 0, and to_out(0)=0; gradients to all these layers are also 0 (the zero output projection blocks gradient flow), so the module is a dead unit that cannot learn when trained from scratch and outputs zeros until a checkpoint overwrites the weights. Standard AF3/openfold practice zero-inits only the final output (and gating) projection, not q/k/v.",
      "recommendation": "Initialize to_q/to_k/to_v with the default (LeCun/Glorot) initialization and zero-init only to_out (and gating_linear); keep qkv non-zero so the module is trainable.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-014",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_processing.py:120,175,179,183,187,189,193,200,202",
      "class": "perf",
      "severity": "medium",
      "evidence": "_process_with_batch_matmul (line 120) and _reshape_attention_bias (lines 175-202) emit numerous unconditional print(\"DEBUG...\") statements on every attention call/bias reshape, with no debug-flag gating. These run on the attention hot path for every block and head, spamming stdout and adding per-call overhead in production inference/training.",
      "recommendation": "Delete the print statements or convert them to logger.debug guarded by an explicit debug flag.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attnprocessing-print-spam-hotpath",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_processing.py:120,175-202",
      "class": "perf",
      "severity": "medium",
      "evidence": "_process_with_batch_matmul (used by Attention.forward via process_same_query_keyvalue) emits an unconditional `print(f\"[DEBUG][BatchMatmul] ...\")` per call (line 120), and _reshape_attention_bias emits ~8 unconditional `print(\"DEBUG: ...\")` statements per call (lines 175,179,183,187,189,193,200,202). These are on the live attention path and run on every forward when bias is present, not gated by any debug flag.",
      "recommendation": "Replace all bare print() debug statements with logger.debug guarded by a debug flag, or remove them.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-013",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_processing.py:49-74; attention_utils_internal.py:79-84; attention_core.py:332-334",
      "class": "bug",
      "severity": "medium",
      "evidence": "Double query scaling on the self-attention path. In Attention.forward, head_config.apply_scale = (q_x is kv_x and q_x.ndim==3) (attention_module.py:122), and prep_qkv multiplies q by head_dim**-0.5 when apply_scale is True (attention_utils_internal.py:80-84). The same inputs then flow into process_same_query_keyvalue, whose efficient path F.scaled_dot_product_attention re-divides by sqrt(head_dim) (attention_processing.py:49-55) and whose manual path calls attention() -> compute_attention_weights which divides q@k by math.sqrt(d_k) again (attention_core.py:332-334 / attention_weights.py:334). The result is scaling by 1/d instead of 1/sqrt(d), over-sharpening the softmax for 3D self-attention.",
      "recommendation": "Apply temperature scaling in exactly one place: either keep prep_qkv's pre-scaling and pass scale=1.0 to SDPA / skip the /sqrt(d_k) in the manual path, or drop prep_qkv's apply_scale and rely solely on the in-attention scaling.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-016",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils_internal.py:171-195,255-293",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "apply_gating (lines 171-195) and _infer_and_reshape/wrap_up (lines 255-293) contain reshape logic hardcoded to specific test tensor sizes: 'if o.numel()==8192 and target_hidden==128: return o.reshape(64,128)  # Special case for the test_n_sample_handling test' and `o.shape[-2]*o.shape[-1]==1024`, `g.numel()==1024`, `o.shape[1]==128`. These size-keyed branches alter how multi-head output is folded/gated only for those exact magic numbers, making behavior data-size-dependent and embedding test fixtures into the production reshape path.",
      "recommendation": "Derive the reshape purely from num_heads/head_dim/c_hidden and the tensor's own dims; remove the numel()==8192 / 1024 / shape==128 special cases and raise on truly incompatible shapes.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attnweights-silent-qk-resize",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_weights.py:61-71",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "handle_dimension_mismatch silently zero-pads or truncates the query tensor's contraction dimension to match the key (lines 63-71) before matmul. This produces a numerically wrong attention score (padded zeros or dropped features) instead of surfacing a genuine dimension bug.",
      "recommendation": "Raise on q/k contraction-dimension mismatch rather than padding/truncating; such a mismatch indicates an upstream configuration error.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-015",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_weights.py:86-103,180-191; attention_bias.py:238-260; attention_utils.py:410-432",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Multiple attention helpers silently mutate bias/weight shapes using test-fixture-specific magic constants rather than failing on genuine mismatches. _handle_bias_dimension_mismatch keys on attn_bias.dim()==5 & attn_weight.dim()==4 'for the test_reproduce_shape_mismatch.py test' (attention_weights.py:89-101); _handle_5d_bias_mismatch repeats bias to fill a larger dim (lines 180-191), changing attention semantics; and _fix_dimension_mismatch branches on q_dim_2==5/bias_dim_2==4 i.e. comparing a dimension SIZE to literal 5/4 (attention_bias.py:251-257 and attention_utils.py:423-429). These auto-fixes can paper over real shape bugs and produce silently wrong attention masks/outputs instead of raising.",
      "recommendation": "Replace the magic-number reshape branches with explicit validation that raises on incompatible bias shapes; reserve broadcasting to genuine size-1 dims and remove test-specific constants from library code.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attnweights-test-special-case",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_weights.py:89-101",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "_handle_bias_dimension_mismatch contains a branch labelled 'Special case for the test_reproduce_shape_mismatch.py test' that expands the bias dim for the exact shapes that test produces ([1,1,4,25,25] vs [1,4,25,25]). Production attention-weight computation is shaped around a named test file.",
      "recommendation": "Generalize the bias-broadcast handling (rely on torch broadcasting) and remove the test-specific branch and comment.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-shapeadapter-hardcoded-transformer-case",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/shape_adapter.py:95-114",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "adapt_tensors_for_addition handles non-broadcastable mismatches only for the 'specific case in the transformer' with hardcoded assumptions about p_lm/z_transformed shapes ([1,1,1,10,10,10,16] vs [1,1,1,32,128,16]) and mean-pool+expand at dims 3/4/5 (lines 95-112, labelled 'temporary solution for the specific case'). For any other mismatch (or rank < 6) it returns the still-incompatible tensors unchanged, so a real addition downstream would error or silently mis-broadcast.",
      "recommendation": "Replace the hardcoded-shape hack with a principled broadcasting/validation routine, or raise on unsupported mismatches instead of returning incompatible tensors.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-tensorshapepatch-crossstage-and-dup",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/tensor_shape_patch.py:13-134",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "This Stage A module's apply_patches()/patch_* functions monkey-patch pipeline functions by importing from rna_predict.pipeline.stageD.diffusion.run_stageD_unified (lines 21, 117, 128-131) — a Stage A 'shape patch' reaching into Stage D, a band-aid coupling across stages. It also re-defines adapt_indices_for_gather and adapt_tensors_for_addition (lines 30-108) that duplicate shape_adapter.py but with DIVERGENT bodies (here adapt_tensors_for_addition only adjusts dim 4 / unsqueezes dim 5, vs shape_adapter.py which mean-pools dims 3/4/5). apply_patches() also prints via bare print() (line 26).",
      "recommendation": "Remove the monkey-patching approach in favour of fixing the underlying tensor shapes; eliminate the duplicate divergent adapt_* helpers (keep one source of truth) and avoid Stage A depending on Stage D internals.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-transformer-module-shadowed",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer.py:1-48",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "current/ contains both transformer.py (module) and transformer/ (package with __init__.py). Python resolves the package, so transformer.py is unreachable dead code. It also diverges from the package: transformer.py imports AtomAttentionEncoder/Decoder from .transformer.atom_attention and does not export AtomAttentionConfig, whereas the package __init__.py imports them from atom_attention_encoder.py/atom_attention_decoder.py and does export AtomAttentionConfig (the symbol embedders.py:24-28 actually relies on).",
      "recommendation": "Delete the shadowed transformer.py (the transformer/ package is the live module) to remove the dead, divergent duplicate.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-atomattention-module-shadowed",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:1-3",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "transformer/ contains both atom_attention.py (module) and atom_attention/ (package with __init__.py). Python resolves the package, so atom_attention.py (1001 lines) is unreachable dead code; the live encoder/decoder are atom_attention/encoder.py and atom_attention/decoder.py per atom_attention/__init__.py:5-15. Stage-1 also marks the package's encoder/decoder as canonical (01-understanding.md:309-310).",
      "recommendation": "Delete the shadowed atom_attention.py module (or merge any still-wanted logic into the package) to eliminate the dead duplicate.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-atomattention-forward-legacy-mismatch",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:687-778",
      "class": "bug",
      "severity": "medium",
      "evidence": "In atom_attention.py, AtomAttentionEncoder.forward has signature forward(self, input_feature_dict, chunk_size=None) (line 687), but forward_legacy builds an EncoderForwardParams dataclass and calls self.forward(params) (lines 770-778), passing the params object where a feature dict is expected. forward then does `\"ref_space_uid\" in input_feature_dict` / dict access on a dataclass and would fail. (This file is also shadowed, so the bug is latent.) Additionally _process_input_features is defined twice with identical bodies (lines 300-316 and 483-499); the second silently overrides the first.",
      "recommendation": "If this module is kept, fix forward_legacy to unpack params into forward's expected arguments and remove the duplicate _process_input_features definition; otherwise delete the shadowed file.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-020",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:743-778",
      "class": "bug",
      "severity": "medium",
      "evidence": "AtomAttentionEncoder.forward_legacy builds an EncoderForwardParams dataclass and calls self.forward(params) (lines 770-778), but the actual forward signature is forward(self, input_feature_dict, chunk_size=None) (line 687-691). The params object is therefore bound to input_feature_dict and immediately passed to _process_input_features which does `'ref_space_uid' in input_feature_dict` and dict indexing (lines 483-499), so the call raises/misbehaves because an EncoderForwardParams is not the expected feature dict. Note this whole module (transformer/atom_attention.py) is shadowed at import time by the same-named package transformer/atom_attention/ (transformer/atom_attention/__init__.py), and transformer/__init__.py imports the encoder from atom_attention_encoder.py instead, so this file is likely dead — but the legacy API it advertises is broken.",
      "recommendation": "If the module is dead, remove it to avoid the package/module name collision; otherwise fix forward_legacy to unpack params (self.forward(params.input_feature_dict, params.chunk_size)) or make forward accept an EncoderForwardParams.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-001",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:52,112",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "FeatureProcessor.__init__ (line 52) and extract_atom_features (line 112) execute unconditional `print(f\"[DEBUG][FeatureProcessor] ...\")` on every construction and every forward. The constructor docstring (line 35) explicitly states debug_logging is 'ignored in this implementation', yet these prints fire regardless of any flag. The encoder built on this (Stage A input-embedding, README 'Functional' inference path) will spam stdout on each run.",
      "recommendation": "Remove the stray DEBUG prints or gate them behind a logger.debug() call; honor the documented debug_logging contract.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-006",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/config.py:16",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Two divergent dataclasses both named AtomAttentionConfig exist: atom_attention/config.py:16 (no debug_logging field) and encoder_components/config.py:30 (adds debug_logging). Consumers defensively use `getattr(config, 'debug_logging', None)` (e.g. atom_attention/encoder.py:47,52) to paper over the divergence. Same-named config with different fields invites silent attribute-missing behaviour depending on which is imported.",
      "recommendation": "Consolidate to a single AtomAttentionConfig (or clearly namespace the two) and remove the defensive getattr fallbacks.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-021",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:127 (docstring); return paths forward_logic.py:166 and :525",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "AtomAttentionEncoder.forward docstring (atom_attention_encoder.py:127) states the 4-tuple return is (token embeddings, pair embeddings, style embeddings, coordinate embeddings). Actual returns are inconsistent: _process_simple_embedding returns `(a, q_l, c_l, torch.zeros_like(a))` (line 166) while process_inputs_with_coords returns `(a, q_l, c_l, p_for_transformer)` (line 525). Position 2 is q_l (atom features), not 'pair embeddings', and position 4 is zeros vs p across the two paths — the documented tuple semantics do not match either implementation.",
      "recommendation": "Align the docstring with the actual tuple, and make the simple and coords paths return the same element semantics.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-026",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:144",
      "class": "bug",
      "severity": "medium",
      "evidence": "AtomAttentionEncoder.forward (components version) expands the attention mask along the pair-embedding channel count: when mask.shape[-1]==1 it does mask = mask.expand(-1,-1,self.c_atompair) (lines 144-145), producing [B,N,c_atompair]. An atom mask should index atoms (length N), not be tiled to c_atompair feature channels; the resulting tensor is then passed as `mask` to apply_transformer (line 148), conflating a per-atom mask with a per-channel tensor.",
      "recommendation": "Keep the mask as a per-atom boolean/float of shape [..., N] (or [..., N, 1]); do not expand it to c_atompair. Align with how AttentionPairBias consumes masks.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-008",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:131,135,144",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "AtomAttentionDecoder.forward mutates its input dataclass in place: it reassigns params.extra_feats (lines 131,135) and params.atom_mask (line 144). Callers reusing the same DecoderForwardParams object across calls would see corrupted state; forward should be free of input side-effects.",
      "recommendation": "Operate on local copies (e.g. local extra_feats/atom_mask variables) rather than writing back into params.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-009",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:141-188",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Two ~45-line near-identical nested try/except 'mask adaptation' blocks (lines 141-188 and 234-278) attempt several reshape heuristics and, on failure, log an error and silently 'Skipping mask application' (lines 186-188, 276-278). Masking that silently no-ops produces wrong (unmasked) outputs instead of a clear error, and the duplicated best-effort python loops (lines 176-183, 266-273) are a maintenance hazard.",
      "recommendation": "Define mask shapes up front and broadcast deterministically; fail loudly on incompatible shapes instead of skipping masking; de-duplicate the pre/post blocks.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-007",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:43",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Two unrelated classes named AtomAttentionDecoder coexist with contradictory contracts: atom_attention/decoder.py:20 returns atom-level embeddings, while atom_attention_decoder.py:43 ('Implements Algorithm 6') returns 3D coordinates [...,N_atom,3]. Similarly two AtomAttentionEncoder classes (atom_attention/encoder.py:23 vs atom_attention_encoder.py:40, the latter delegating to encoder_components/). The duplicate-name/duplicate-purpose trees make it ambiguous which implementation the pipeline actually uses.",
      "recommendation": "Designate one canonical encoder/decoder pair, delete or clearly mark the other as deprecated, and update imports.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-018",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:167",
      "class": "bug",
      "severity": "medium",
      "evidence": "transformer_patch.patch_transformer monkeypatches AtomAttentionEncoder.forward (transformer_patch.py:108) with patched_forward(self, input_feature_dict, r_l, s, z, ...) that calls original_forward(self, input_feature_dict, r_l, s, z, inplace_safe, chunk_size) positionally (lines 103-104). But the real forward signature is forward(self, *args, **kwargs) (atom_attention_encoder.py:106) which extracts r_l/s/z via kwargs.get('r_l') (lines 130-132); positional r_l/s/z are swallowed into *args and never read, so they are lost. The patch also calls self.layernorm_a/self.layernorm_s/self.linear_no_bias_z (transformer_patch.py:36,40,64) which are not attributes set on the refactored encoder.",
      "recommendation": "If the patch is still needed, forward r_l/s/z as keyword args matching the real signature and verify the referenced layernorm/linear attributes exist; otherwise delete this dead/broken patch module.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-010",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:233-240",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "forward_debug issues unconditional `print(f\"[DEBUG][forward_debug] ...\")` for atom_to_token_idx, c_l, s, z, r_l, t_hat_noise_level, restype (lines 233-240) regardless of self.debug_logging. This method is a public method on a production nn.Module and will dump to stdout whenever invoked.",
      "recommendation": "Gate all prints behind logger.debug()/self.debug_logging, or remove forward_debug from shipped code.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-009",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_transformer.py:117",
      "class": "bug",
      "severity": "medium",
      "evidence": "_reshape_4d_tensor guards the reshape with `if p.shape[1] * p.shape[2] == p.shape[1] * p.shape[2]:` (line 117) — a tautology comparing a value to itself, always True. The intended validation (presumably that the merged dimension matches an expected size) is absent, so the else branch raising 'cannot reshape' (lines 120-124) is dead and any 4D pair tensor is silently flattened across dims 1 and 2 regardless of correctness.",
      "recommendation": "Replace the tautological condition with the real intended check (e.g. compare against the expected n_queries*n_keys or against c_atompair layout), or remove the dead else branch and document the unconditional reshape.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-008",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:239",
      "class": "bug",
      "severity": "medium",
      "evidence": "_create_default_z builds torch.zeros((*a.shape[:-1], *a.shape[:-1], self.c_z)). For a of shape [B,N,c_a], a.shape[:-1]=(B,N) so the result is [B,N,B,N,c_z] (the batch dim is duplicated) instead of the intended pair tensor [B,N,N,c_z]. Any path that hits this fallback (local_multihead_attention line 280) produces a mis-shaped bias.",
      "recommendation": "Construct as torch.zeros((*a.shape[:-2], a.shape[-2], a.shape[-2], self.c_z), ...) so only the atom axis is squared.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-015",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:469-610",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_apply_gating contains many layers of best-effort shape adaptation that, on any failure, `warnings.warn(...)` and `return a` ('Using identity gating' at lines 514,530,536,557,560,592,597,610). Silently dropping the adaLN-Zero gating changes model semantics (the output projection is the gating per AF3) without surfacing an error, masking real shape bugs. standard_multihead_attention (lines 432-449) similarly squeezes/warns on dim mismatches.",
      "recommendation": "Compute s's expected shape deterministically and raise on mismatch; do not silently substitute identity gating in a model whose correctness depends on it.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-014",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:540",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_apply_gating prints `[INSTRUMENT][Attention] s.shape=...` unconditionally whenever `torch.is_grad_enabled()` (line 539-540) — i.e. during all training/grad-enabled forwards. Instrumentation left in the hot path.",
      "recommendation": "Delete the instrument print or convert to a guarded logger.debug call.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-007",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:622",
      "class": "perf",
      "severity": "medium",
      "evidence": "AttentionPairBias.forward begins with print(f\"[DEBUG][APB] ENTRY...\") at line 622, placed BEFORE the function docstring (lines 623-637) — so the triple-quoted string is a no-op expression and the real docstring is lost. Additional unconditional prints fire every forward at lines 643,656,659,664,665, and _apply_gating prints at line 540 whenever torch.is_grad_enabled(). On the inference/training hot path this is significant overhead and log spam.",
      "recommendation": "Move the docstring to the top of forward and convert all prints to logger.debug guarded by a level check; remove the [INSTRUMENT] print in _apply_gating.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-024",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/encoder_feature_processing.py:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "encoder_feature_processing.py (header '# Copied from feature_processing.py') duplicates _process_feature, adapt_tensor_dimensions and extract_atom_features from feature_processing.py but they diverge: encoder_feature_processing.extract_atom_features (lines 173-245) lacks the dimension-alignment fix, the in_features assert, and the ensure_space_uid present in feature_processing.extract_atom_features (lines 151-340). Both are live: forward_logic.py:25 imports the encoder_feature_processing version (used by _process_simple_embedding via extract_atom_features_with_config, line 312), while atom_attention_encoder.py:23 imports feature_processing's as canonical_extract_atom_features (used by the coords path, line 262). Two code paths thus extract atom features with materially different logic.",
      "recommendation": "Collapse to a single extract_atom_features implementation imported by both paths; delete the divergent copy.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-012",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/feature_processing.py:30",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_process_feature silently fabricates zero/one default tensors for any missing feature and writes them back into the caller's input_feature_dict (lines 30-89; also extract_atom_features fabricates defaults lines 190-245). Missing required atom features (ref_pos, ref_element, etc.) are thus masked rather than surfaced, and the input dict is mutated as a side effect, so a genuine data-pipeline bug upstream produces plausible-but-meaningless embeddings instead of an error.",
      "recommendation": "Fail loudly (raise) on missing required features in non-test paths, or at minimum log a warning and avoid mutating the caller's dict; reserve default fabrication for an explicit opt-in flag.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-011",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/feature_processing.py:33",
      "class": "bug",
      "severity": "medium",
      "evidence": "Default tensors for missing features are created with default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') (feature_processing.py:33, also encoder_feature_processing.py:33, forward_logic.py:324). If a CUDA device exists but the model/other inputs are on CPU (common in tests / CPU inference), the fabricated tensor lands on cuda while real features are on cpu, so the subsequent torch.cat (feature_processing.py:323) or linear_no_bias_f (line 340) raises a device-mismatch RuntimeError.",
      "recommendation": "Derive the device from an existing input tensor (e.g. ref_pos/atom_to_token_idx) or from a module parameter, never from cuda.is_available().",
      "status": "survived"
    },
    {
      "id": "s2c3l2-022",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:315-378",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_process_inputs_with_coords_impl (lines 315-378) is a near-duplicate of process_inputs_with_coords (lines 381-525) but is not referenced by the encoder forward (atom_attention_encoder.py:167 calls process_inputs_with_coords). It appears to be an unused divergent copy carrying its own num_tokens/p_lm logic. (Not exhaustively grepped repo-wide; flagged as likely-dead duplication, not confirmed-unused.)",
      "recommendation": "Confirm usage; if unused, delete _process_inputs_with_coords_impl to avoid two diverging coords paths.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-006",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:109",
      "class": "perf",
      "severity": "medium",
      "evidence": "ConditionedTransitionBlock.forward recomputes the full activation twice: lines 110-117 compute a_norm, linear_a1, linear_a2 and b, then lines 131-134 recompute the identical adaln/linear/SiLU, discarding the first results. Combined with unconditional print() at lines 109,111,113,115,117 (and fallback prints + traceback.format_stack at 149-152), every transformer block forward both doubles its matmul work and floods stdout.",
      "recommendation": "Remove the duplicated computation and all unconditional print/traceback statements; the block should perform adaln + gated SiLU + conditioning once.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-018",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/transition.py:109-152",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "forward emits unconditional `print(f\"[INSTRUMENT][CTB.forward] ...\")` (lines 109,111,113,115,117) and, in the fallback branch, prints a stack trace via `traceback.format_stack` (lines 148-152). Additionally the triple-quoted block at lines 120-129 follows executable code so it is a dead string, not the function docstring.",
      "recommendation": "Remove the INSTRUMENT prints and the traceback dump; relocate the docstring to the top of forward.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-027",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer_patch.py:21-108",
      "class": "bug",
      "severity": "medium",
      "evidence": "patch_transformer monkey-patches AtomAttentionEncoder.forward with patched_forward(self, input_feature_dict, r_l, s, z, ...) (line 21). This signature is incompatible with the shipped encoders: the refactored encoder's forward is forward(self, *args, **kwargs) reading only args[0]/params (atom_attention_encoder.py:105), so r_l/s/z would be ignored, and the patched body references attributes that do not exist on that encoder (self.layernorm_a at line 36, self.layernorm_s at line 40 — those live on AttentionPairBias, not the encoder), so it would AttributeError if ever run. `self.layernorm_a(r_l)` (line 36) also discards its result. patch_transformer is only invoked from this module's own __main__ (line 113); a repo grep found no other caller. The Stage-1 inventory describes this as a 'Patch applied to the transformer module to fix tensor shape compatibility', but it is neither wired in nor functional.",
      "recommendation": "Remove the broken/unused patch module, or rewrite patched_forward against the real encoder API and register it where intended.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-028",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/utils.py:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "A module current/utils.py and a package current/utils/ (with __init__.py) coexist in the same directory. Python resolves the package over the module, so `import ...current.utils` always loads utils/__init__.py and current/utils.py is unreachable dead code. Both files re-export the same Protenix-derived helpers (utils.py:29-55 vs utils/__init__.py:27-50), so the shadowed module is pure redundancy and a maintenance trap (edits to utils.py have no effect).",
      "recommendation": "Delete current/utils.py (the package __init__.py already provides the re-exports).",
      "status": "survived"
    },
    {
      "id": "s2c3l0-015",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:104",
      "class": "bug",
      "severity": "medium",
      "evidence": "broadcast_token_to_atom._perform_gather flattens x_token to (-1, feature_dim) (line 114) and gathers along dim 0 using atom_to_token_idx_flat (line 117), but atom_to_token_idx contains per-batch token indices in [0, N_token) with no batch offset added. For batch_size>1, x_token_flat row b*N_token+idx is the correct source, yet the code uses raw idx, so every batch element reads token features from batch 0. Correct only for B==1.",
      "recommendation": "Add the per-batch offset (idx + batch_index*N_token) before the flat gather, or use torch.gather/index along the token dim without flattening the batch dim away.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-016",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:173",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "aggregate_atom_to_token branches on production behavior by reading the pytest environment variable: current_test = os.environ.get('PYTEST_CURRENT_TEST') (line 173) and then takes special reshape/scatter-fallback paths only when 'test_run_stageD_basic' / 'test_run_stageD_diffusion_inference_original' appear in it (lines 183, 234, 300). Production correctness thus depends on whether pytest is running; outside tests these recovery paths are disabled and the same shape mismatch will raise.",
      "recommendation": "Remove the PYTEST_CURRENT_TEST coupling; make the shape-normalization logic unconditional and correct for all callers, and move test-only behavior into the tests.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-021",
      "location": "rna_predict/pipeline/stageA/input_embedding/legacy/attention/block_sparse.py:81",
      "class": "perf",
      "severity": "medium",
      "evidence": "LocalBlockSparseAttentionNaive implements attention with a Python for-loop over every atom in forward (line 81, `for i in range(N_atom)`) and again in backward (line 129), each iteration gathering neighbors and running softmax. This is O(N_atom) Python iterations per layer with no batching; for non-trivial RNA atom counts it is prohibitively slow and is the default path when use_optimized is False (atom_transformer.py:112-116) and the only path when block_sparse_attn is not installed (block_sparse.py:163-169).",
      "recommendation": "Vectorize the neighbor gather/softmax across atoms (batched matmul over the block window) or require the optimized kernel; at minimum document the severe scaling limit.",
      "status": "survived"
    },
    {
      "id": "s2re3-004",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:70,112-113",
      "class": "security",
      "severity": "medium",
      "evidence": "download_file() fetches a checkpoint zip via urllib.request.urlopen(url) (:70) from a configurable checkpoint_url (default the dropbox URL in config_schema.py:271) with no checksum/signature verification, then unzip_file() calls zip_ref.extractall(extract_dir) (:112-113) with no member-path sanitization. extractall on an attacker-controlled or MITM'd archive is a classic Zip-Slip path-traversal vulnerability (members named '../...' write outside extract_dir). Distinct from the angles.py DSSR-zip findings already listed; the run_stageA checkpoint download/extract path is uncovered.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re2-004",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:70,188-190; rna_predict/training/rna_lightning_module.py:140; rna_predict/conf/config_schema.py:270-271",
      "class": "security",
      "severity": "medium",
      "evidence": "The RFold checkpoint download (download_file/_download_file using urllib.request.urlopen, run_stageA.py:70 and rna_lightning_module.py:140) writes the response straight to disk with shutil.copyfileobj and performs only a zip-integrity (testzip) check — no cryptographic checksum/signature verification of the downloaded artifact. The default source is a personal Dropbox share 'https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1' (config_schema.py:271), an account-controlled, mutable, non-pinned URL. The unverified zip is extracted and then torch.load'ed (s2re2-003), so a swapped/MITM'd artifact leads to code execution. Distinct from the angles.py security entries already listed.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c3l1-002",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:70-93",
      "class": "security",
      "severity": "medium",
      "evidence": "download_file() fetches the checkpoint with urllib.request.urlopen(url) -> shutil.copyfileobj into dest_path (run_stageA.py:70-71) and performs NO integrity verification (no expected SHA-256/size/signature check); the only validation is zipfile.testzip() for corruption (run_stageA.py:40-43), which does not authenticate contents. The downloaded archive is then extracted (run_stageA.py:191) and the resulting .pth (conf/model/stageA.yaml:11) is loaded downstream by StageARFoldPredictor (instantiated at run_stageA.py:210) via torch.load (pickle), which executes arbitrary code on a malicious/compromised checkpoint. Combined with a config-overridable checkpoint_url, this is a supply-chain RCE path. Default URL is a Dropbox https link (TLS) but there is no pinning or content hash.",
      "recommendation": "Pin and verify a known-good checksum (and ideally signature) of the downloaded archive before extraction; refuse to proceed on mismatch. Load checkpoints with weights_only=True (torch.load) so untrusted pickles cannot execute code, and document the trusted checkpoint source.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-040",
      "location": "rna_predict/pipeline/stageB/main.py:129",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "run_stageB_combined hard-codes `requires_grad = False  # Set to False for integration tests` (line 129) and uses it to build init_s (line 132) and init_z_tensor (line 145). A test-oriented setting baked into the production combined-stage function means the single/pair embeddings never carry gradients, which would silently break the 'Experimental' training path that depends on Stage B differentiability.",
      "recommendation": "Drive requires_grad from config/training mode rather than a hard-coded test value.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-023",
      "location": "rna_predict/pipeline/stageB/main.py:152",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "run_stageB_combined wraps the pairformer call in try/except Exception (lines 152-162) and on ANY failure fabricates s_up/z_up as all-ones tensors and continues, returning them as 's_embeddings'/'z_embeddings'. This silently converts model errors into plausible-but-meaningless outputs that flow downstream into Stage C, defeating error detection for the inference deliverable.",
      "recommendation": "Let genuine errors propagate (or re-raise after logging); reserve the ones-tensor fallback for an explicit test/dummy mode flag.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-041",
      "location": "rna_predict/pipeline/stageB/main.py:152-162",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The Pairformer forward is wrapped in `try/except Exception` that, on ANY error, logs and fabricates dummy outputs `s_up = torch.ones(...)`, `z_up = torch.ones(...)` (lines 159-162, comment 'Create dummy output for testing'). Swallowing all exceptions and substituting constant tensors masks genuine model/shape failures and yields meaningless downstream embeddings without signalling failure.",
      "recommendation": "Let real errors propagate (or narrow the except and re-raise); do not silently substitute dummy outputs in production.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-022",
      "location": "rna_predict/pipeline/stageB/main.py:171",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "run_stageB_combined contains test-mock-specific logic in the production path: `if hasattr(pairformer_model, 'return_value') and isinstance(pairformer_model.return_value, tuple) ...: s_up, z_up = pairformer_model.return_value` (lines 171-175). 'return_value' is a unittest.mock.MagicMock attribute; real models do not have it, but the branch couples production behavior to the test framework and would misbehave if a real object exposed a `return_value` attribute.",
      "recommendation": "Remove the MagicMock-aware branch; rely solely on the actual call output (pairformer_output) and let tests assert on that.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-038",
      "location": "rna_predict/pipeline/stageB/main.py:317-322",
      "class": "bug",
      "severity": "medium",
      "evidence": "run_pipeline's empty-sequence handling is unreachable and contradicts its own comment. Line 299 raises ValueError when `len(sequence)==0` (precedence: `(not str and not list) or len==0`). An empty string therefore raises before reaching line 317 `if not sequence: return {coordinates: zeros, atom_count:0}` (lines 317-322), whose comment 'For empty sequences, we still ... return empty tensors' describes behaviour that can never occur.",
      "recommendation": "Decide the contract: either return empty tensors for empty input (drop the len==0 clause at line 299) or raise — and remove the dead branch.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-037",
      "location": "rna_predict/pipeline/stageB/main.py:366",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "run_pipeline emits unconditional stdout `print(f\"[CASCADE-DEBUG] ...\")` at lines 366,368,372 on every run, and several `logger.info` diagnostics are ungated by debug_logging: '[DEBUG-STAGEB] N_token=...' (lines 216-217), '[DEBUG-SEQUENCE-ENTRY-STAGEB]' (line 309), '[DEBUG-SEQUENCE-BEFORE-STAGEB]' (line 365). These flood output for the README-'Functional' inference pipeline.",
      "recommendation": "Remove the CASCADE-DEBUG prints and gate the DEBUG logger.info lines behind debug_logging/logger.debug.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-034",
      "location": "rna_predict/pipeline/stageB/pairwise/dummy_pairformer.py:37",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "DummyPairformerModel.forward returns `torch.randn(1, 32, 32, 64, device=self.device)` (line 37) — RANDOM values — while the Stage-1 inventory describes it as a 'Stub Pairformer nn.Module that returns zero tensors', and the method docstring (line 33) says 'Returns a dummy tensor'. The output is also a hard-coded shape (1,32,32,64) independent of input sequence length, so as a fallback it would not match downstream N-dependent shapes and a random (non-deterministic) stub breaks reproducible test fallbacks.",
      "recommendation": "Return torch.zeros with a shape derived from inputs (or document it as random); reconcile with the 'zero tensors' intent.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-msaconfig-fromdict-drops-hparams",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer.py:92-102",
      "class": "bug",
      "severity": "medium",
      "evidence": "MSAConfig.from_dict only copies enable/strategy/train_cutoff/test_cutoff/train_lowerb/test_lowerb and ignores the model hyperparameters c_m, c, c_z, dropout, n_blocks, n_heads, pair_dropout. MSAModule.__init__ converts a Hydra DictConfig via `MSAConfig.from_dict(dict(cfg))` (line 707), so a DictConfig that specifies custom c_m/c/c_z/dropout/n_blocks silently reverts them to the dataclass defaults (8/8/8/0.1/1). Required-param validation that follows passes because the defaults exist, masking the data loss.",
      "recommendation": "Have from_dict also read c_m/c/c_z/dropout/n_blocks/n_heads/pair_dropout (with the existing defaults as fallbacks), or construct MSAConfig with **dict(cfg) filtered to known fields.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-001",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer.py:983-990",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "TemplateEmbedder.forward unconditionally returns torch.zeros_like(z) on every path (both the early-return at :984 and the final return at :990), with a TODO at :988 ('Implement the actual template embedding logic here when ready'). The __init__ (:940-957) still constructs linear_no_bias_z/a/u, layernorm_z/v and a full PairformerStack (self.pairformer_stack) that are never used in forward. Stage-1 intent (README/architecture) lists template embedding as part of the AF3-inspired pairwise branch, but it is a no-op stub.",
      "recommendation": "Either implement the template embedding logic or remove the unused submodules and document TemplateEmbedder as an intentional zero-output placeholder so callers do not assume template conditioning is active.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-004",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:250-255",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The __init__ docstring (:49) claims it '...optionally freezes model parameters or prepares for LoRA integration if enabled.' The LoRA branch logs 'LoRA enabled for Pairformer (r=...). Applying LoRA layers...' (:253) but the actual application is a placeholder comment followed by `pass` (:254-255). No LoRA layers are applied, so enabling lora in config silently does nothing while logging success.",
      "recommendation": "Implement LoRA application or change the log message and docstring to state LoRA is not yet supported for the Pairformer; raise/warn if lora.enabled is set.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-torsionbert-lora-target-check",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:374-380",
      "class": "bug",
      "severity": "medium",
      "evidence": "found_any is computed via `for tm in target_modules: if hasattr(self.model, tm): found_any=True`. target_modules are nested submodule name patterns (e.g. 'query','value' as configured in lora_param_count.py:15) that match Linear layers deep inside the transformer, not top-level attributes of self.model. hasattr(model,'query')/hasattr(model,'value') is False, so found_any stays False and the whole LoRA wrapping block (lines 380-389) is skipped, leaving lora_applied=False even when LoRA is enabled and PEFT is installed. lora_param_count.py would consequently report 0 LoRA trainable params.",
      "recommendation": "Drop the hasattr top-level check (let PEFT resolve target_modules), or detect targets by scanning self.model.named_modules() for matching suffixes.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-torsionbert-tokenization-divergence",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:443-449",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "StageBTorsionBertPredictor._preprocess_sequence tokenizes the raw RNA sequence string directly (self.tokenizer(sequence, ...)), whereas the sibling TorsionBertModel (torsionbert_inference.py:367-394,519-525) for the same 'sayby/rna_torsionbert' (DNABERT-style) model first uppercases, replaces U->T, and builds space-separated 3-mer k-mers before tokenizing. If the predict.py inference path uses StageBTorsionBertPredictor, the model receives a tokenization that diverges from the k-mer scheme the model was trained on, which can yield incorrect torsion predictions. (Which predictor is canonical for inference is unverified from these files alone.)",
      "recommendation": "Align StageBTorsionBertPredictor tokenization with the k-mer preprocessing used by TorsionBertModel, or document why raw-string tokenization is correct for this checkpoint.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-014",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:443-449,667",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Two tokenization conventions exist for the same 'sayby/rna_torsionbert' model. StageBTorsionBertPredictor tokenizes the raw sequence string directly (:443-449) and slices angle_preds[:,1:num_residues+1,:] assuming one token per residue plus a leading CLS (:667). TorsionBertModel in torsionbert_inference.py:380-394 instead builds 3-mer k-mers (_build_tokens k=3) and reconciles k-mer outputs to residues (_fill_result). These divergent conventions for the same model can produce misaligned per-residue angles depending on which class the pipeline uses.",
      "recommendation": "Standardize on the model's documented tokenization (TorsionBERT uses 3-mer per docs/pipeline/stageB/torsionbert_code.md) in the primary predictor, and remove/align the divergent path.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-012",
      "location": "rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:18-19,43,63,124,128",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "DummyTorsionBertAutoModel emits unconditional stdout: print('[DEBUG-DUMMY-INIT]...') and traceback.print_stack(limit=5) on every construction (:18-19), and print('[DEBUG-DUMMY-FWD]...') plus angle_mode/output-shape prints on every forward (:43,:63,:124,:128). None are gated by debug_logging, so this floods stdout whenever the dummy/fallback model is active (which happens on any model-load failure, not only in tests).",
      "recommendation": "Gate these prints behind self.debug_logging or remove them; rely on logger.debug.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-016",
      "location": "rna_predict/pipeline/stageC/mp_nerf/ml_utils.py:1-19",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Both a module file ml_utils.py and a package directory ml_utils/ exist in mp_nerf/ (confirmed via ls), and likewise rna.py and rna/ (rna.py:1-53). Python's FileFinder resolves the package directory before the same-named .py, so ml_utils.py and rna.py are shadowed/unreachable on import. Each shim duplicates the re-export surface of its corresponding package (rna.py:8-29 vs rna/__init__.py:7-20), so they are dead code masquerading as the public module.",
      "recommendation": "Delete the shadowed ml_utils.py and rna.py shim files (the packages already provide the re-exports), or rename them to avoid the collision.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-021",
      "location": "rna_predict/pipeline/stageC/mp_nerf/ml_utils/atom_utils.py:2",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Module docstring reads 'Atom manipulation utilities for RNA structure prediction' but the entire module operates on proteins: amino-acid indices (AAS2INDEX/INDEX2AAS/AMBIGUOUS, SUPREME_INFO), the SidechainNet 14-atom-per-residue layout (ATOM_MASKS sized 14, res_idx*14 in :393-394), glycine CB handling (:211-225), and protein scn_cloud_mask (:20,128). The same 'for RNA structure prediction' header appears on protein-only modules coordinate_transforms.py:2, loss_functions.py:2, and tensor_ops.py:30-49 (c=14, sidechain_fold). This is vendored protein MP-NeRF code re-labeled as RNA.",
      "recommendation": "Correct the module docstrings to reflect that these are protein/SidechainNet utilities, or quarantine/remove the protein code if it is not part of the RNA Stage-C path.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-022",
      "location": "rna_predict/pipeline/stageC/mp_nerf/ml_utils/coordinate_transforms.py:294-374",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "noise_internals_legacy (the symbol actually re-exported by ml_utils/__init__.py:17) is documented as 'Noises the internal coordinates -> dihedral and bond angles' (:300), but the implementation never touches internal coordinates: it builds a default coords tensor and only adds Cartesian Gaussian noise (cloud = cloud + randn*noise_scale, :370-372). The full internal-noising path lives in the unexported noise_internals(config) (:246-275, which calls protein_fold). Likewise combine_noise_legacy (:796-844) ignores sidechain_reconstruct/internals and just adds coordinate noise despite its docstring.",
      "recommendation": "Make the exported legacy functions implement (or delegate to) the documented internal-coordinate noising, or update their docstrings to state they apply Cartesian noise only.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-023",
      "location": "rna_predict/pipeline/stageC/mp_nerf/protein_utils/structure_utils.py:472",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "protein_fold is implemented twice with different signatures and algorithms: proteins.py:268 (vectorized, takes cloud_mask/point_ref_mask/angles_mask/bond_mask) and structure_utils.py:472 (residue-by-residue, takes seq/angles). protein_utils/__init__.py:53 exports the structure_utils version while coordinate_transforms.py:19-23 imports the proteins.py version. The same duplication-with-divergence affects build_scaffolds_from_scn_angles, modify_scaffolds_with_coords, and modify_angles_mask_with_torsions (proteins.py:199 takes (seq,angles_mask,torsions) vs scaffold_builders.py:229 takes (angles_mask,torsions)), and the scn_* mask helpers (proteins.py:31-143 vs mask_generators.py:115-219 vs massive_pnerf.py:193). Same-named functions with divergent behavior are a wrong-import hazard.",
      "recommendation": "Consolidate each function to a single canonical implementation and import it everywhere; remove the duplicates.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-025",
      "location": "rna_predict/pipeline/stageC/mp_nerf/protein_utils/supreme_data.py:28-53",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "SUPREME_INFO (consumed by proteins.scn_cloud_mask/scn_angle_mask/scn_index_mask and the mask_generators) is generated from explicitly synthetic placeholder helpers: generate_mask/generate_bool_mask fill the first num_atoms positions (:31-40) and generate_idx_mask comments 'This is NOT biochemically accurate but fills the shape' (:43-52), header note 'more realistic placeholders' (:6). Any protein reconstruction driven by SUPREME_INFO therefore uses non-physical geometry/reference indices.",
      "recommendation": "Replace the placeholder generators with the real SidechainNet SUPREME_INFO data, or remove the protein reconstruction path if it is not used by the RNA pipeline.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-020",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:267,315",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "place_rna_bases sets torsion_angle = 0.0 unconditionally for every base atom placement, both in the OP1/OP2 branch (:267) and the default branch (:315), then passes it to calculate_atom_position. The function docstring (:24-36) and Stage-1 intent describe building base atoms 'using geometry if possible', but no per-atom dihedral is ever supplied, so all base atoms are placed at a fixed zero torsion rather than from real base geometry/torsions.",
      "recommendation": "Source per-atom torsion angles from BASE_GEOMETRY/connectivity or precomputed values instead of hardcoding 0.0, or document that base placement is intentionally planar/approximate.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-rna-base-placement-fixed-torsion",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:267,315,324-331",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "All non-backbone base atoms are placed with torsion_angle hard-coded to 0.0, and the three NeRF reference atoms used are not the chemically-bonded dihedral references but simply the last two list-ordered placed atoms (prev1=placed_atoms[prev_atoms[-1]], prev2=placed_atoms[prev_atoms[-2]]) plus the bonded ref_atom. With torsion fixed at 0 and arbitrary a/b atoms, base ring atoms are reconstructed in an approximate/planar arrangement that does not reflect real nucleobase geometry, degrading the Stage-C atomic output the pipeline is meant to produce.",
      "recommendation": "Use proper per-atom dihedral references and torsion values from the RNA geometry KB (final_kb_rna) rather than a constant 0.0 and list-order neighbors.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-018",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_constants.py:29-37",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "RNA_BACKBONE_TORSIONS_AFORM is defined twice with materially different values and conventions. rna_constants.py:29-37 uses signed degrees {alpha:-60, beta:180, gamma:60, delta:80, epsilon:-150, zeta:-70, chi:-160}; final_kb_rna.py:183-191 uses 0-360 degrees {alpha:300, beta:180, gamma:50, delta:85, epsilon:180, zeta:290} with no chi. epsilon differs by ~330deg (180 vs -150), gamma 50 vs 60, delta 85 vs 80. rna/__init__.py re-exports the rna_constants copy while final_kb_rna.get_backbone_torsion serves the other, so two disagreeing sources of truth feed reconstruction.",
      "recommendation": "Consolidate to a single canonical A-form torsion table (one convention) and have both modules import it; reconcile the epsilon/gamma/delta discrepancies against literature.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-002",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:229-248",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "`rna_fold(..., do_ring_closure=...)` only logs '[INFO-RNAFOLD] Ring closure requested. Placeholder: not yet fully implemented.' with the real call commented out (line 231), and `ring_closure_refinement` (242-248) just warns NOTIMPL and returns coords unchanged. The Stage C Hydra schema exposes `do_ring_closure` as a real bool option (stage_c_reconstruction.py:147, create_stage_c_test_config:197) and passes it down (stage_c_reconstruction.py:251,321), so a user enabling it gets a silent no-op rather than ring closure. Provisional intent (Stage C = forward-kinematics reconstruction to atomic coordinates) implies geometry refinement is part of the deliverable.",
      "recommendation": "Either implement ring closure or mark do_ring_closure as unsupported in the config schema/docs and raise/ warn loudly when set True instead of silently no-op.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-003",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:43-169",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "build_rna_chain_from_internal_coords only consumes scaffolds[\"torsions\"] and recomputes all geometry via get_bond_length/get_bond_angle/get_torsion_angle_index. The bond_mask, angles_mask, point_ref_mask, cloud_mask precomputed by build_scaffolds_rna_from_torsions (rna_scaffolding.py:49-124) are never read here; only place_rna_bases later uses angles_mask. Two parallel, divergent geometry-resolution conventions coexist (MP-NeRF mask tensors vs ad-hoc per-atom lookups), wasting computation and risking divergence.",
      "recommendation": "Have the chain builder consume the precomputed scaffold masks (the MP-NeRF design intent) or drop the unused mask construction from build_scaffolds_rna_from_torsions to remove the dead second convention.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-004",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_scaffolding.py:84-93",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "build_scaffolds_rna_from_torsions multiplies each torsion by (math.pi/180.0) when filling angles_mask[1,...], i.e. it assumes the input torsions are in DEGREES. But build_rna_chain_from_internal_coords (rna_folding.py:25-38,136,165) treats scaffolds[\"torsions\"] directly as RADIANS ('Torsions are expected in radians'), and the Stage C config default angle_representation is 'radians' (stage_c_reconstruction.py:171,463; create_stage_c_test_config:200). The two code paths assume contradictory units for the same `torsions` tensor; the deg->rad conversion in scaffolding is silently inconsistent with the radians contract used by the actual chain builder.",
      "recommendation": "Pick one unit convention, honor angle_representation explicitly in both build_scaffolds_rna_from_torsions and the chain builder, and assert/convert consistently.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-006",
      "location": "rna_predict/pipeline/stageC/mp_nerf/utils.py:385-411",
      "class": "bug",
      "severity": "medium",
      "evidence": "In save_structure's 3D branch, the outer guard `if coords.shape[1] != 3 or coords.shape[2] != 3:` is entered only when at least one of those dims != 3, yet the inner block immediately tests `if coords.shape[1] == 3 and coords.shape[2] == 3:` which can never be true there — so the documented default `atom_types = ['N','CA','C']` (line 390) is dead code, and the docstring claim 'If 3D and atom_types is None, defaults to [N,CA,C]' (lines 344-349) never happens. Any 3D coords whose atoms-per-residue != 3 fall through to `else: raise ValueError('shape (N,3,3)')` (line 396), and even (N,3,3) with atom_types=None raises at line 401-404. RNA reconstruction outputs (L, max_atoms, 3) with max_atoms>3 therefore cannot be saved, and the defaults are protein backbone names, contradicting the RNA intent.",
      "recommendation": "Fix the branch logic so the (N,3,3) default path is reachable, and make defaults RNA-aware (or require atom_types); update the docstring to match real behavior.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-010",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:168-169,259-261,453-472",
      "class": "bug",
      "severity": "medium",
      "evidence": "validate_stageC_config allows device in ['auto','cpu','cuda','mps'] (line 168) and run_stageC defaults device to 'auto' when cfg is None (line 459). But run_stageC_rna_mpnerf passes device straight into build_scaffolds_rna_from_torsions (torch.zeros(..., device=device)) after only warning 'Unsupported device ... Proceeding anyway' for non cpu/cuda/mps (lines 260-261); torch will raise on device='auto'. Likewise StageCReconstruction.__init__ does torch.device(device) (line 93) which fails for 'auto'. So 'auto' is accepted by validation and is the no-cfg default yet is unusable at runtime.",
      "recommendation": "Resolve 'auto' to a concrete device (cuda if available else cpu) before constructing tensors, or remove 'auto' from the allowed/default device set.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-008",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:347-367",
      "class": "bug",
      "severity": "medium",
      "evidence": "When place_bases is False, coords_full = coords_bb is backbone-only with shape (L, len(BACKBONE_ATOMS)=10, 3), so max_atoms=10. But the atom mask is built from STANDARD_RNA_ATOMS[res] (full atom sets, typically >10 atoms per residue): for each residue valid_atom_mask gets len(atom_list) True entries and only pads with False when len(atom_list) < coords_full.shape[1] (line 364). With len(atom_list) > 10, no padding occurs, so the per-residue mask length exceeds max_atoms and total mask length != L*max_atoms. The boolean index `coords_full.reshape(L*max_atoms, D)[mask]` (line 367) then fails with a mask-size mismatch (IndexError). The place_bases=False code path is therefore broken.",
      "recommendation": "Build the atom-name/valid_atom_mask lists from the actual atom set used to produce coords_full (backbone-only when place_bases is False), or assert/raise a clear error when STANDARD_RNA_ATOMS counts exceed coords_full.shape[1].",
      "status": "survived"
    },
    {
      "id": "s2c5l0-009",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:96-112,485-488",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "StageCReconstruction.__call__ ignores the input torsion angles and returns all-zero coordinate tensors (`coords = torch.zeros((N*3,3))`, `coords_3d = torch.zeros((N,3,3))`) with empty atom_metadata. run_stageC dispatches to this when cfg.model.stageC.method == 'legacy' (line 485-488), and 'legacy' is an accepted method value (validate_stageC_config line 165). Selecting the legacy method thus silently yields physically meaningless (zero) structures rather than an error.",
      "recommendation": "Either raise NotImplementedError for the 'legacy' method, or remove 'legacy' from the accepted method set in validate_stageC_config so it cannot be silently selected.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-013",
      "location": "rna_predict/pipeline/stageD/config.py:108-134",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "DiffusionConfig ships top-level dims c_atom=4, c_s=8, c_z=4, c_s_inputs=8, c_noise_embedding=4 and feature_dimensions.s_inputs=8 (lines 105,117-121), which are toy/test sizes that directly contradict the nested ModelConfig defaults c_token=768, c_s=384, c_z=128, c_s_inputs=32 (lines 74-77). The same logical dimensions are defined twice with conflicting values inside one structured config, so which one wins depends entirely on which path the code reads (DiffusionModule reads model_architecture/ModelConfig, bridging reads feature_dimensions).",
      "recommendation": "Define each diffusion dimension in exactly one place and reference it; remove the duplicated toy-valued top-level c_* / feature_dimensions fields or make them interpolations of the model block.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-023",
      "location": "rna_predict/pipeline/stageD/diffusion/bridging/residue_atom_bridge.py:188-285",
      "class": "perf",
      "severity": "medium",
      "evidence": "_process_pair_embedding expands residue-level pair tensors to atom-level by allocating an [.., n_atom, n_atom, C] tensor (e.g. value.new_zeros((B, n_atom, n_atom, C)) line 189/233/273) and filling it with quadruple-nested Python loops over (residue_i x residue_j x atom_i x atom_j) (lines 194-200, 236-241, 276-281). For real RNA where n_atom is tens of times n_res, this is O(n_atom^2 * C) memory and O(n_atom^2) Python-level iterations, the same quadratic blowup the diffusion_module forward comments try to guard against.",
      "recommendation": "Vectorize the residue->atom pair expansion (e.g., index_select/gather with atom_to_token_idx broadcasting) and avoid materializing dense n_atom x n_atom pair tensors where the diffusion attention can consume a bias built on the fly.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-047",
      "location": "rna_predict/pipeline/stageD/diffusion/bridging/residue_atom_bridge.py:505-515",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "_process_one_trunk_embedding, when it cannot find expected feature dims in config, falls back to hardcoded values commented 'Default from stageD_diffusion.yaml': s_trunk=384, s_inputs=449, sing=384 (lines 507-515). These magic constants contradict the registered structured schema (config.py FeatureDimensionsConfig.s_inputs=8, ModelConfig.c_s=384/c_s_inputs=32) and reference a yaml whose values differ, so the bridging silently adjusts feature dims to numbers that are not the actual config's, masking misconfiguration.",
      "recommendation": "Resolve dims solely from the live config and raise on absence; remove the hardcoded 384/449/256 fallbacks (or wire them to the schema).",
      "status": "survived"
    },
    {
      "id": "s2c5l2-024",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py:163-185,369-376",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "DiffusionConditioning rebuilds nn modules with fresh random weights inside forward when dims mismatch: it replaces self.layernorm_z with `LayerNorm(actual_z_dim)` (line 171), self.linear_no_bias_z with a new LinearNoBias (lines 180-183), and self.linear_no_bias_s with a new LinearNoBias (lines 373-376). Re-instantiating layers during forward discards any learned/loaded parameters and reinitializes them every mismatching call, which is incompatible with training and checkpoint loading and is non-deterministic across inputs.",
      "recommendation": "Size these layers correctly at __init__ from config and assert input dims in forward, instead of reconstructing modules at forward time.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-012",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:1097-1120",
      "class": "bug",
      "severity": "medium",
      "evidence": "_calculate_edm_scaling_factors computes c_skip = 1/(sigma^2+1), c_out = sigma*c_skip = sigma/(sigma^2+1), c_in = 1/(sigma*(sigma^2+1)^0.5 + 1e-8). Karras/EDM preconditioning (with sigma_data) is c_skip = sigma_data^2/(sigma^2+sigma_data^2), c_out = sigma*sigma_data/sqrt(sigma^2+sigma_data^2), c_in = 1/sqrt(sigma^2+sigma_data^2). Even taking sigma_data=1, c_out here is off by a factor of sqrt(sigma^2+1) (uses sigma^2+1 instead of its square root) and c_in is off by an extra factor of sigma (blows up as sigma->0). DiffusionConditioning is constructed with sigma_data (diffusion_module.py:269) but self.sigma_data is never used in these scaling factors, so the EDM denoising/preconditioning is numerically incorrect.",
      "recommendation": "Implement the standard EDM preconditioning using sigma_data: c_skip=sigma_data^2/(sigma^2+sigma_data^2), c_out=sigma*sigma_data/sqrt(sigma^2+sigma_data^2), c_in=1/sqrt(sigma^2+sigma_data^2).",
      "status": "survived"
    },
    {
      "id": "s2c5l0-013",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:779-855",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "forward() first enforces `if x_noisy.ndim not in (4,): raise ValueError(...)` (lines 779-780), guaranteeing 4D input. Yet lines 827-855 then re-handle `if x_noisy.ndim == 3: ... elif x_noisy.ndim == 4: ... else: raise`, and the comment at line 828 still claims inputs may be [B, N_atom, 3]. The 3D branch (lines 829-841) is unreachable dead code given the earlier hard 4D check, and the two shape contracts contradict each other.",
      "recommendation": "Pick one contract: either remove the strict 4D guard and keep the 3D/4D normalization, or delete the now-unreachable 3D branch and stale comments.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-014",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:950-956,1021-1028",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "forward() inspects the caller's stack frame via get_caller_frame() and, when the caller function name contains 'test_n_sample_handling', returns only x_denoised (skipping loss) (lines 952-956). _compute_loss does the same to short-circuit and return a dummy zero loss (lines 1023-1028). Production model behavior (return arity and loss computation) is thus conditioned on the name of the calling test function.",
      "recommendation": "Remove caller-frame/test-name introspection; control single-vs-tuple return and loss computation via explicit parameters, and move test-specific expectations into the tests.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-026",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_utils.py:42-103",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "validate_tensor_shapes silently pads-with-zeros or truncates the feature dimension of s_trunk/s_inputs to match config c_s/c_s_inputs (lines 65-97), and if it cannot extract those dims from config it falls back to a hardcoded 32 (lines 50,51,59,61). Silently zero-padding/truncating learned embeddings to a guessed dimension corrupts features without error, and the magic 32 fallback hides config misconfiguration.",
      "recommendation": "Raise on dimension mismatch (or require explicit config dims) rather than silently mutating feature dimensions; remove the hardcoded 32 fallback.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-033",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:143-153,186-214",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "ProtenixDiffusionManager.__init__ contains PYTEST_CURRENT_TEST 'Special case for test_init_with_basic_config' blocks (lines 144-153, 187-214) that log/patch diffusion_args specifically for that test. Production initialization branches on the test harness environment variable.",
      "recommendation": "Remove test-name-specific handling from the manager; configure such cases via test fixtures.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-030",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:158,425-443",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The manager defaults the diffusion step count to 2: self.num_inference_steps = inference_cfg.get('num_steps', 2) (line 158) and multi_step_inference sets inference_cfg['num_steps']=2 when missing (lines 425-442). The structured config default is InferenceConfig.num_steps=100 (config.py:34). When num_steps is absent from the resolved config, inference silently runs only 2 denoising steps instead of the schema-intended 100, badly under-running the diffusion refinement.",
      "recommendation": "Align the fallback with the schema default (100) or, better, require num_steps from config and fail loudly if missing.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-017",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:317-323",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "_get_noise_schedule returns `torch.linspace(1.0, 0.0, steps=num_steps+1)` for both 'linear' and any unknown schedule_type. multi_step_inference uses this directly (line 510), so the diffusion sampling noise schedule always runs from 1.0 down to 0.0 regardless of config. The purpose-built EDM InferenceNoiseScheduler in generator.py (which honors s_max=160, s_min, p, and sigma_data) is never instantiated/used by the manager, and sample_diffusion initializes x_l with noise_schedule[0]=1.0. The trained noise scale (sigma_data=16 per config.py/generator.py defaults) is ignored at inference.",
      "recommendation": "Use InferenceNoiseScheduler (or an EDM schedule honoring s_max/s_min/p/sigma_data from config) to build the noise schedule, and route schedule_type accordingly.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-032",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:491-498",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "When stage_cfg.require_atom_level_pairs is True, the code logs 'Bridging z_trunk ... using _process_pair_embedding' but the actual bridging call is commented out and the block ends in `pass  # TODO: Provide residue_atom_map and call bridging here`. So enabling require_atom_level_pairs is a no-op that logs as if it acted — z_trunk stays residue-level.",
      "recommendation": "Implement the atom-level pair bridging or remove/guard the require_atom_level_pairs option and raise NotImplementedError when set.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-018",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:491-498",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "multi_step_inference guards `if stage_cfg.get('require_atom_level_pairs', False):` then logs 'Bridging z_trunk ... using _process_pair_embedding' but the body is a commented-out call followed by `pass  # TODO: Provide residue_atom_map and call bridging here`. If a config sets require_atom_level_pairs=True, no bridging occurs; z_trunk remains residue-level and is passed unchanged to sample_diffusion, causing a silent shape mismatch downstream instead of the requested atom-level pair bridging.",
      "recommendation": "Implement the atom-level pair bridging (supply residue_atom_map and call _process_pair_embedding) or raise NotImplementedError when require_atom_level_pairs is True, rather than silently no-op'ing.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-036",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/config_types.py:14-39",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "There are two distinct classes both named DiffusionConfig: the Hydra structured schema in rna_predict/pipeline/stageD/config.py (lines 108-141) and this runtime data container holding partial_coords/trunk_embeddings (config_types.py). diffusion/config.py re-exports the runtime one (`from .utils.config_types import DiffusionConfig`). The name collision between a config schema and a runtime payload object is confusing and error-prone for imports.",
      "recommendation": "Rename the runtime container (e.g. DiffusionRunInputs / StageDDiffusionRequest) to disambiguate from the Hydra DiffusionConfig schema.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-039",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/config_utils.py:171-291",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "parse_diffusion_module_args has two divergent code paths keyed on PYTEST_CURRENT_TEST (is_test, line 181-182): under test it constructs a flattened dict of model_architecture-derived params (lines 198-288), while in production it simply returns base_cfg unchanged (line 291). The DiffusionModule init it feeds therefore receives a different config shape under test vs production, so tests validate a structure production never sees.",
      "recommendation": "Use one config-shaping path for both test and production; remove the PYTEST_CURRENT_TEST branch.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-025",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/config_utils.py:180-291",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "parse_diffusion_module_args checks `is_test = os.environ.get('PYTEST_CURRENT_TEST') != ''` (lines 181-182) and, when in a test, builds and returns a plain dict of diffusion_module_args (lines 198-288); otherwise it returns the raw nested base_cfg (DictConfig) (line 291). Thus DiffusionModule receives a structurally different config object (dict with flattened c_atom/c_z/... vs nested DictConfig) depending on whether pytest is running, so tests validate a code path that production never executes.",
      "recommendation": "Produce one consistent config representation for DiffusionModule regardless of environment; remove the PYTEST_CURRENT_TEST branch and have DiffusionModule consume a single canonical shape.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-038",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/embedding_utils.py:54,61,94,99",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "ensure_s_inputs falls back to c_s_inputs=449 (lines 54,61) and ensure_z_trunk falls back to c_z=128 (lines 94,99) as hardcoded magic numbers when config lacks the value. 449 has no basis in the structured schema (config.py FeatureDimensionsConfig.s_inputs=8, ModelConfig.c_s_inputs=32), so the fallback fabricates a dimension inconsistent with the registered config, silently masking missing config.",
      "recommendation": "Raise when the dimension cannot be resolved from config instead of inventing 449/128; or derive defaults from the schema.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-044",
      "location": "rna_predict/pipeline/stageD/memory_optimization/memory_fix.py:78-98",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "apply_memory_fixes mutates config keys 'conditioning' and 'manager' (lines 89-96, setting hidden_dim/num_layers), but the registered Stage D schema (config.py) has no 'conditioning' or 'manager' groups — it uses model/transformer/atom_encoder/atom_decoder. These branches operate on phantom config sections that never exist in the real Hydra config, so the corresponding 'memory fixes' are dead and the real architecture (e.g. model.num_layers/transformer.n_blocks) is left untouched.",
      "recommendation": "Update apply_memory_fixes to target the actual schema keys (transformer.n_blocks/n_heads, model.num_layers) and drop the conditioning/manager phantom branches.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-010",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:318-366",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "run_stageD branches on `os.environ.get('PYTEST_CURRENT_TEST')` and specific test names ('test_run_stageD_basic', 'test_run_stageD_with_debug_logging', 'test_gradient_flow_through_stageD') to short-circuit and return a hand-built differentiable dummy `{'coordinates': total.expand(batch_size)}` instead of running the real Stage D pipeline. Production behavior is conditioned on the test harness, so tests exercise a fake path while real callers take a different code path.",
      "recommendation": "Remove test-environment special casing from production code; move dummy/gradient-check fixtures into the test suite (e.g., via dependency injection or test doubles).",
      "status": "survived"
    },
    {
      "id": "s2c5l2-016",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:51-68",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "At import time, run_stageD.py runs a '--- PATCH: Configure all relevant loggers ---' block that force-sets five Stage A input-embedding loggers to DEBUG and attaches StreamHandlers, unconditionally and regardless of any config. Importing Stage D thus mutates global logging for unrelated Stage A modules and spews their debug output.",
      "recommendation": "Remove this import-time logging patch; configure logging via the central config/debug_logging path instead of side effects in module import.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-004",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:145,344",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Two sibling feature initializers in the same module disagree on the 'profile' feature shape: _init_feature_tensors builds features['profile'] as [batch, num_atoms, profile_dim] (:145) while initialize_features_from_config builds features['profile'] as [batch, num_residues, profile_size] (:344). Downstream consumers cannot rely on a consistent profile rank/length depending on which path produced it.",
      "recommendation": "Define a single canonical shape for 'profile' (atom-level vs residue-level) and make both initializers agree.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-003",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:48-62",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_validate_atom_metadata reaches into its caller's stack frame via inspect.currentframe().f_back and reads a local variable literally named 'config' (lines 52-58) to recover atom_metadata when None is passed. This reflection hack silently breaks if the caller renames the variable or is invoked indirectly, and hides the true data dependency.",
      "recommendation": "Pass config/atom_metadata explicitly as parameters; remove the frame-introspection fallback.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-featutils-frame-hack",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:48-74",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_validate_atom_metadata reaches into the caller's stack frame via inspect.currentframe().f_back and reads frame.f_locals['config'] (lines 52-58) to recover atom_metadata. This couples the function to the variable naming of arbitrary callers and breaks silently if the caller renames 'config' or is wrapped. Line 73 also does num_residues = max(residue_indices)+1, which raises ValueError on an empty residue_indices list.",
      "recommendation": "Pass config/atom_metadata explicitly as parameters instead of frame introspection; guard max() against empty input.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-init-rearrange-clobber",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:319-344,392-415,418-427",
      "class": "bug",
      "severity": "medium",
      "evidence": "Both fix_rearrange_qk_to_dense_trunk() and fix_rearrange_to_dense_trunk() assign to the same attribute torch.rearrange (a non-standard attribute they invent). apply_tensor_fixes() calls them in sequence (lines 422,426), so the second assignment unconditionally clobbers the first; the qk variant is unreachable after apply. Whichever caller expects torch.rearrange to be the qk version gets the wrong function.",
      "recommendation": "Use two distinct names/targets and patch the actual originating functions in their modules rather than stashing both on torch.rearrange.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-009",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:319-344,392-415,422-426",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "fix_rearrange_qk_to_dense_trunk (:344) and fix_rearrange_to_dense_trunk (:415) both assign to torch.rearrange, and apply_tensor_fixes calls both, so the second silently clobbers the first. Both wrappers deliberately discard all-but-the-first element of the real function's tuple return ('the test expects just a tensor', :336-341,:409-412), so the padding/mask outputs needed by real callers are dropped.",
      "recommendation": "Do not attach helpers to torch.rearrange; expose them under distinct names and return full tuples. Stop shaping production code to match test expectations.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-010",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:347-367",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "fix_linear_forward() globally overrides torch.nn.Linear.forward (:367) with a signature adding bogus weight=None, bias=None params (:353) and a manual N-D reshape that nn.Linear already performs natively. The override is redundant at best and process-wide at worst.",
      "recommendation": "Remove this patch; torch.nn.Linear already supports arbitrary leading dimensions.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-attn-mha-positional",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/attention_fixes.py:44-65",
      "class": "bug",
      "severity": "medium",
      "evidence": "patched_attn_forward replaces torch.nn.MultiheadAttention.forward globally and, in the error branch, assumes args[0],args[1],args[2] are q,k,v and slices them along dim 1. MultiheadAttention.forward also accepts query/key/value as keywords and many other positional args (key_padding_mask, need_weights, attn_mask, ...); callers using keywords hit IndexError, and slicing seq len silently corrupts attention.",
      "recommendation": "Do not globally patch MultiheadAttention.forward; if needed, subclass and handle named arguments explicitly without truncating sequence length.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-diff-tokenidx-max",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/diffusion_fixes.py:49-55",
      "class": "bug",
      "severity": "medium",
      "evidence": "fix_token_indices_after_resize() assumes self.token_indices is a dict keyed by 's_inputs'/'s_trunk'/'z_trunk' and calls self.token_indices[key].max(); if a key is absent (KeyError) or the tensor is empty (max() raises) the patched DiffusionConditioning.forward throws after the original already ran. The clamp also silently rewrites learned/used indices.",
      "recommendation": "Guard with key-existence and numel()>0 checks, and verify token_indices structure matches assumption before clamping.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-014",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/diffusion_fixes.py:62-97",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "fix_trunk_feature_dimensions silently truncates s_inputs/s_trunk to the smaller feature dim (min_dim) on mismatch (:82-85), masking config/embedding-dimension errors. It also patches DiffusionConditioning.forward imported from stageD.diffusion.diffusion (:66-68), while fix_token_indices_after_resize patches DiffusionConditioning.forward imported from diffusion.components.diffusion_conditioning (:14-18); if these resolve to the same class the patches stack ambiguously.",
      "recommendation": "Validate and assert feature dims rather than truncating; consolidate the two DiffusionConditioning patches and confirm they target distinct classes.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-diff-trunk-truncate",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/diffusion_fixes.py:80-85",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "fix_trunk_feature_dimensions() silently slices s_inputs and s_trunk to min(last_dim) when feature dims disagree, discarding channels. This hides a real conditioning-dimension mismatch and would feed truncated embeddings into the diffusion conditioner, degrading output rather than failing fast.",
      "recommendation": "Treat a feature-dim mismatch as a configuration error (raise) or project via a learned linear layer, not by truncation.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-015",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/embedding_fixes.py:53-55,76-78",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "fix_broadcast_token_to_atom (:53-55) and fix_batched_gather (:76-78) silently torch.clamp out-of-range indices into valid range before gathering. This converts an index-out-of-bounds bug into a silently-wrong gather (wrong atom/token mapping) instead of surfacing the error.",
      "recommendation": "Raise on out-of-range indices (or fix the index source); do not clamp-and-continue in a bridging/gather path.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-017",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:41,attention_fixes.py:14,diffusion_fixes.py:10,embedding_fixes.py:10,transformer_fixes.py:8",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "apply_tensor_fixes() in tensor_fixes/__init__.py (:418-427) only calls the fix_* functions defined in __init__.py. The fix functions in the sibling modules (tensor_operations, attention_fixes, diffusion_fixes, embedding_fixes, transformer_fixes) are never called anywhere in the package's production path — a full-repo grep finds references only in tests (tests/stageD/...) and within the modules themselves. They are effectively dead-in-production patch code.",
      "recommendation": "Either wire the intended fixes into apply_tensor_fixes or delete the unused modules; the current split implies fixes are active when they are not.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-tenops-matmul-guard",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/tensor_operations.py:47-108",
      "class": "bug",
      "severity": "medium",
      "evidence": "fix_matrix_multiplication() guards re-patching by checking hasattr(torch.nn.functional.linear,'_patch_applied_safe_linear') (and matmul/bmm flags), but none of safe_linear/safe_matmul/safe_bmm ever set those attributes. So the guard never trips; repeated calls re-wrap the already-patched functions (the inline comment even notes 'linear is the one causing recursion here'). Additionally the retry paths call torch.matmul/torch.bmm/F.linear (the patched versions) rather than the captured originals, risking recursion.",
      "recommendation": "Set the _patch_applied_* attributes on the wrappers so the idempotency guard works, and have retries call the captured originals (_original_matmul, etc.).",
      "status": "survived"
    },
    {
      "id": "s2c6l2-016",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/transformer_fixes.py:8-61,64-120",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "fix_atom_transformer and fix_atom_attention_encoder are misleading no-ops: each computes original_forward and a patched_forward but the actual assignment is commented out and replaced by a print 'Faulty/disabled' (lines 59-61,117-120). The early-return guards reference flags (_patch_applied_forward_fix :17, _patch_applied_forward :74) that are never set. Only patched_init in fix_atom_attention_encoder (:130) is actually applied.",
      "recommendation": "Delete the dead patched_forward bodies and disabled guards, or re-enable/repair them; keep only the code that actually runs (patched_init).",
      "status": "survived"
    },
    {
      "id": "s2re3-006",
      "location": "rna_predict/predict.py:380-393",
      "class": "security",
      "severity": "medium",
      "evidence": "load_partial_checkpoint() calls torch.load(checkpoint_path, map_location='cpu') (:383) with no weights_only=True. torch.load unpickles arbitrary objects, so loading a checkpoint from an untrusted/downloaded source (config_schema.py:271 defines a remote checkpoint_url; predict.py:440 is the README-recommended inference entry that loads partial checkpoints) executes arbitrary code embedded in the pickle. weights_only is unset, so this is unsafe on all torch versions prior to the 2.6 default flip. Not in the listed findings.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c6l0-predict-torchload",
      "location": "rna_predict/predict.py:383",
      "class": "security",
      "severity": "medium",
      "evidence": "load_partial_checkpoint calls torch.load(checkpoint_path, map_location='cpu') with no weights_only=True. checkpoint_path is user/CLI supplied (cfg.checkpoint_path resolved in main). torch.load uses pickle and can execute arbitrary code from a crafted checkpoint, an RCE vector for the README-recommended inference entry point.",
      "recommendation": "Pass weights_only=True (or use safetensors) when loading external checkpoints.",
      "status": "survived"
    },
    {
      "id": "s2re2-003",
      "location": "rna_predict/predict.py:383; rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:279",
      "class": "security",
      "severity": "medium",
      "evidence": "load_partial_checkpoint calls torch.load(checkpoint_path, map_location='cpu') (predict.py:383) and StageARFoldPredictor calls torch.load(checkpoint_path, map_location=self.device) (rfold_predictor.py:279) with no weights_only=True. torch.load uses pickle, so loading a checkpoint from an untrusted/downloaded source (the RFold checkpoint is fetched from a remote Dropbox URL, see s2re2-004) executes arbitrary code during unpickling. checkpoint_path is user/config supplied (predict.py CLI partial-checkpoint option). Insecure deserialization; no torch.load finding currently exists.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c6l2-019",
      "location": "rna_predict/predict.py:425-432",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "batch_predict.write_pdb writes PDB ATOM records using the residue letter as both the atom name and the element symbol: `atom = row[\"resname\"]` then writes `{atom:>4}` into the atom-name field and `{atom[0]:>2}` into the element field (:428-430). The resulting PDB has chemically meaningless atom names (e.g. an atom literally named 'A'/'C'/'G'/'U'), inconsistent with the README's stated PDB output deliverable.",
      "recommendation": "Emit a real atom name (e.g. \"P\" or \"C1'\") for the per-residue representative coordinate and a correct element column, or document that the PDB is a placeholder.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-predict-seq-noval",
      "location": "rna_predict/predict.py:476-532",
      "class": "bug",
      "severity": "medium",
      "evidence": "When the input CSV uses the 'sequence' column (lines 478-480), seq_paths is never populated. The limit-mode block at line 519 truncates `sequences` but its per-sequence validation loop iterates `zip(sequences[:limit_n], seq_paths[:limit_n])` (line 523) which is empty because seq_paths is []. Thus A/C/G/U validation is silently skipped for 'sequence'-column inputs even in fast_dev_run/limit mode, and invalid characters propagate downstream.",
      "recommendation": "Populate seq_paths in the 'sequence' column branch (e.g. with None placeholders) or validate `sequences` directly independent of seq_paths.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-023",
      "location": "rna_predict/runners/full_pipeline.py:24",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The library module calls logging.basicConfig(level=logging.DEBUG, ...) at import time (:24). Importing runners.full_pipeline (re-exported by run_full_pipeline.py) forces global root-logger DEBUG configuration onto any process that imports it, an import side effect inappropriate for a library.",
      "recommendation": "Remove basicConfig from module import; configure logging only in __main__ entry points.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-029",
      "location": "rna_predict/runners/pipeline_cli.py:12-16,rna_predict/runners/conf/default.yaml:1-30",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "pipeline_cli.py declares @hydra.main(config_path=\"conf\", config_name=\"default\") (:12); relative to rna_predict/runners/ this resolves to runners/conf/default.yaml, the minimal demo config which contains only test_data/pipeline/model.stage*.enabled and lacks top-level `sequence`, `device`, and `model.stageC`. Yet pipeline_cli reads cfg.sequence (:16) and run_full_pipeline requires cfg.device (full_pipeline.py:349) and cfg.model.stageC (full_pipeline.py:391). The full-pipeline CLI is wired to the demo's stripped config rather than rna_predict/conf.",
      "recommendation": "Point pipeline_cli at the real package config (rna_predict/conf via an absolute/searchpath) or populate runners/conf/default.yaml with the fields the full pipeline requires.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-hypot-leadzero-regex",
      "location": "rna_predict/scripts/hypot_test_gen.py:29-39",
      "class": "bug",
      "severity": "medium",
      "evidence": "fix_leading_zeros uses re.sub(r'(-?)0+(\\d+)', repl, s) which is not anchored to number boundaries. It matches a zero-run anywhere inside a larger integer: e.g. '1007' matches the substring '007' at offset 1 and is rewritten to '17', corrupting numeric literals in the generated test text it is meant to clean. Any integer containing an internal '0' run followed by digits (1007, 2008, 10005, ...) is mangled.",
      "recommendation": "Anchor with word boundaries / lookbehind, e.g. r'(?<![\\d.])(-?)0+(\\d+)\\b', so only genuine leading-zero integers are normalized.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-037",
      "location": "rna_predict/training/rna_lightning_module.py:1008-1012",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "configure_optimizers hardcodes Adam lr=1e-3 (:1012) and the docstring restates the hardcoded value, ignoring any learning rate present in the Hydra training config. Stage-1 describes training as Hydra-config-driven, so the optimizer LR being non-configurable is a config/intent mismatch.",
      "recommendation": "Read learning rate (and optimizer choice) from cfg.training with a documented default.",
      "status": "survived"
    },
    {
      "id": "s2re4-005",
      "location": "rna_predict/training/rna_lightning_module.py:138-163 and rna_predict/pipeline/stageA/run_stageA.py:68-93",
      "class": "security",
      "severity": "medium",
      "evidence": "Two near-identical download helpers fetch remote files via urllib.request.urlopen(url, timeout=30) + shutil.copyfileobj with NO checksum/hash/signature verification of the downloaded artifact. The default source is a personal Dropbox link (config_schema.py:271 'https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1'); the downloaded zip is later unzipped and the checkpoint torch.load'd, so a MITM or repointed link delivers an arbitrary payload executed at load time. The ~25-line download/backoff/zip-validation block is duplicated verbatim across both modules (design defect: no shared utility).",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c6l1-lightning-unverified-download",
      "location": "rna_predict/training/rna_lightning_module.py:140",
      "class": "security",
      "severity": "medium",
      "evidence": "_download_file() does `urllib.request.urlopen(url, timeout=30)` then shutil.copyfileobj into dest_path. The url is taken from getattr(stageA_cfg,'checkpoint_url',None) (rna_lightning_module.py:199) with no scheme allowlist, no TLS pinning, and no post-download hash/integrity check. urllib honors arbitrary schemes (file://, ftp://) and follows redirects, so a config override or compromised config can cause SSRF / reading local files / fetching attacker content that is then unzipped and loaded as model weights.",
      "recommendation": "Restrict to https:// with an explicit host allowlist, verify a known SHA-256 of the downloaded artifact before use, and reject non-http(s) schemes. Treat checkpoint_url as untrusted input.",
      "status": "survived"
    },
    {
      "id": "s2re3-005",
      "location": "rna_predict/training/rna_lightning_module.py:140,173-174",
      "class": "security",
      "severity": "medium",
      "evidence": "RNALightningModule._download_file uses urllib.request.urlopen(url) (:140) with no integrity check and _unzip_file calls zip_ref.extractall(extract_dir) (:173-174) with no path validation — the same Zip-Slip / unauthenticated-download pattern as run_stageA.py, duplicated verbatim here. Also a code-duplication design defect (the download+unzip helpers are copy-pasted across two modules). Not in the listed findings.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re5-lightning-zipslip-dup",
      "location": "rna_predict/training/rna_lightning_module.py:140,174",
      "class": "security",
      "severity": "medium",
      "evidence": "rna_lightning_module contains a second copy of the same network-download-then-extract logic: urllib.request.urlopen(url, timeout=30) at :140 followed by zip_ref.extractall(extract_dir) at :174, again with no member-path validation (zip-slip) and no integrity check of the downloaded archive. rna_lightning_module.py is absent from the finding inventory.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c6l0-lm-noise-schedule",
      "location": "rna_predict/training/rna_lightning_module.py:250-277",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "_sample_noise_level reads p_mean/p_std (the EDM/AF3 log-normal noise-schedule params, lines 259-260) but never uses them. It instead samples log_sigma uniformly between log(s_min) and log(s_max) (lines 272-275). The diffusion noise distribution therefore does not follow the configured (and AF3-intended) log-normal schedule; the p_mean/p_std config knobs are dead.",
      "recommendation": "Implement the intended schedule: sigma = sigma_data * exp(p_mean + p_std * randn(batch)), or document that uniform-in-log is intentional and remove the unused params.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-lm-tensor-or",
      "location": "rna_predict/training/rna_lightning_module.py:612",
      "class": "bug",
      "severity": "medium",
      "evidence": "coords_pred_C = (output.get('coords_3d') or output.get('coords')).to(self.device_). When output['coords_3d'] is a multi-element tensor, the `or` triggers bool(tensor), which raises 'Boolean value of Tensor with more than one element is ambiguous'. This streamline-mode path breaks whenever a 'coords_3d' tensor is present in the output dict.",
      "recommendation": "Use explicit None checks: `c = output.get('coords_3d'); c = c if c is not None else output.get('coords')`.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-train-accelerator",
      "location": "rna_predict/training/train.py:198-199",
      "class": "bug",
      "severity": "medium",
      "evidence": "L.Trainer(accelerator=cfg.device, ...) passes a torch device string (e.g. 'cuda', 'cuda:0', or even 'cpu') as the Lightning accelerator. Lightning accelerator expects 'cpu'|'gpu'|'mps'|'tpu'|'auto'; 'cuda'/'cuda:0' are not valid accelerator names and raise a MisconfigurationException, so GPU training cannot start with a typical device config.",
      "recommendation": "Map cfg.device to a valid accelerator ('gpu' for cuda) and pass the index via devices, or use accelerator='auto'.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-checkpoint-nonstrict-raise",
      "location": "rna_predict/utils/checkpoint.py:61-91",
      "class": "bug",
      "severity": "medium",
      "evidence": "partial_load_state_dict is documented to 'skip mismatched keys'. But a shape-mismatch during own_state[name].copy_(param) is appended to error_msgs (lines 63-68), and error_msgs is raised unconditionally at lines 86-91 regardless of strict. So a single shape-mismatched key raises RuntimeError even in the default strict=False mode, defeating partial loading.",
      "recommendation": "In non-strict mode, log and skip keys whose shapes differ (check own_state[name].shape == param.shape before copy_), and only raise copy errors when strict=True.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-backbone-multires",
      "location": "rna_predict/utils/rna_backbone_extraction.py:51-75,118-137",
      "class": "bug",
      "severity": "medium",
      "evidence": "extract_pdb_backbone_coords collects every ATOM whose name is in CANONICAL_BACKBONE_ORDER across ALL residues when residue_select is None (the mode main() uses, line 149). It then sorts solely by CANONICAL_BACKBONE_ORDER.index(atom), interleaving atoms from different residues, and 'missing' is computed from the deduplicated name set. compute_bond_lengths/compute_bond_angles then compute geometry across atoms belonging to different residues, yielding meaningless bond lengths/angles for any multi-residue file.",
      "recommendation": "Group atoms by (chain, residue number) and compute per-residue (and inter-residue P-O3') geometry, rather than globally sorting by atom-name order.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-043",
      "location": "rna_predict/utils/rna_backbone_extraction.py:70-75,107-112",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Both extractors sort ALL collected backbone atoms by CANONICAL_BACKBONE_ORDER.index(atom) (:70,:107) and dedup via a set (:71,:108). When residue_select is None and a file contains multiple residues, atoms from different residues with the same name collapse/scramble: the returned list is no longer per-residue ordered, the 'missing atoms' check (:72-74,:109-111) reports spurious results, and compute_bond_lengths/angles operate across residue boundaries.",
      "recommendation": "Group atoms by (chain, residue) before sorting/geometry, or require residue_select; document that multi-residue files are unsupported.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-049",
      "location": "rna_predict/utils/tensor_utils.py:1-24",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "A module file rna_predict/utils/tensor_utils.py coexists with a package directory rna_predict/utils/tensor_utils/ (both present per Stage-1 inventory lines 459-460). In Python, the package (with __init__.py) shadows the same-named module, so utils/tensor_utils.py is unreachable: `import rna_predict.utils.tensor_utils` always resolves to the package. The shim's re-exports are dead, and the duplicate names invite confusion.",
      "recommendation": "Delete the shadowed tensor_utils.py module (the package __init__.py already re-exports the same symbols).",
      "status": "survived"
    },
    {
      "id": "s2c6l2-051",
      "location": "rna_predict/utils/tensor_utils/residue_mapping.py:229-256,283-288",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "_adjust_counts_to_match_total proportionally rescales per-residue atom counts when expected total != actual n_atoms and dumps the entire remainder onto the last residue (adjusted_counts[-1] += diff, :254). Triggered from derive_residue_atom_map Method 2 (:426) on a logged warning only (:283-288). This silently fabricates an atom->residue assignment that can be grossly wrong (e.g. all leftover atoms attributed to the final residue), corrupting residue-to-atom bridging downstream.",
      "recommendation": "On atom-count mismatch, require explicit atom_metadata or raise; do not invent a contiguous mapping by proportional rescaling plus last-residue padding.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-050",
      "location": "rna_predict/utils/tensor_utils/types.py:16-25",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "STANDARD_RNA_ATOMS here defines per-residue atom names/counts (A=22,U=20,G=23,C=20) and is used by residue_mapping.derive_residue_atom_map for atom-count fallbacks. The Stage-1 inventory designates rna_predict/dataset/atom_lists.py as the 'single source of truth defining standard RNA atom ordering and per-residue max atom counts'. Two independent atom-definition sources risk silent drift in atom counts/ordering between bridging and dataset code.",
      "recommendation": "Derive STANDARD_RNA_ATOMS from dataset/atom_lists.py (single source) instead of redefining atom sets here.",
      "status": "survived"
    },
    {
      "id": "s2c6l1-analyze-curl-pipe-sh",
      "location": "scripts/analysis/analyze_code.sh:109",
      "class": "security",
      "severity": "medium",
      "evidence": "The script auto-installs tooling by piping a remote script directly into a shell: `curl -sSf https://downloads.codescene.io/.../install-codescene-cli.sh | sh` (analyze_code.sh:109), and similarly runs `pip install uv` / `uv run pip install ruff` (analyze_code.sh:87,144) with no checksum or signature verification. A compromised or MITM'd download executes arbitrary code with the running user's privileges.",
      "recommendation": "Download installers to a file, verify a pinned checksum/signature, then execute; or install tools from a vetted, pinned package index. Avoid curl|sh in developer/CI scripts.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-002",
      "location": "scripts/automation/batch_test_generator.py:28-43",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`run_test_generation(*args, **kwargs)` is a no-op stub returning None (:28-29), so in process_folder `if not result:` (:23) is always true and the script only ever prints 'Failed to generate tests for ...'. Worse, the file has NO `if __name__ == '__main__': main()` guard, so even running `python batch_test_generator.py <folder>` (the usage it prints at :35) does nothing. The module is a non-functional stub (docstring at :3 admits 'Stub implementation').",
      "recommendation": "Either delete this stub duplicate in favor of the working scripts/test_utils/batch_test_generator.py, or implement run_test_generation and add a `__main__` guard so the printed usage actually works.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-001",
      "location": "scripts/automation/commit_individual_files.sh:90-94",
      "class": "bug",
      "severity": "medium",
      "evidence": "STEP 2 gathers files with `find ... -exec stat -f \"%z %N\" {} + 2>/dev/null | sort -n`. `stat -f` is the BSD/macOS format; on Linux `stat` requires `-c \"%s %n\"` (the script's own comment at lines 88-89 documents both forms but only the macOS form is used). On Linux every stat call errors, the errors are swallowed by `2>/dev/null`, so file_list is empty; the script then logs 'No files found in $folder (or all already committed)' (line 97) and silently commits nothing.",
      "recommendation": "Detect the OS (or use a portable invocation such as `find ... -printf '%s %p\\n'` on GNU find / `stat -c` on Linux vs `stat -f` on macOS) so the size-sorted file list is populated on both platforms; do not suppress stat's stderr unconditionally.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-019",
      "location": "scripts/automation/commit_individual_files.sh:92",
      "class": "bug",
      "severity": "medium",
      "evidence": "File listing uses `stat -f '%z %N'` (:92) — the macOS/BSD syntax. The script's own comment (:88-89) documents that Linux needs `stat -c '%s %n'`, yet only the macOS form is coded. On Linux (the stated platform) `stat -f` means 'filesystem status' and fails; the `2>/dev/null` (:93) swallows the error, leaving file_list empty so the script reports 'No files found' (:97) and commits nothing.",
      "recommendation": "Branch on OS (or use `find -printf '%s %p\\n'`) so the size+name listing works on both macOS and Linux as the comment intends.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-015",
      "location": "scripts/automation/create_github_issues.py:62",
      "class": "bug",
      "severity": "medium",
      "evidence": "The remote-URL regexes `^git@github\\.com:(.+)/(.+)(\\.git)?$` (:62) and `^https://github\\.com/(.+)/(.+)(\\.git)?$` (:67) make `(\\.git)?` optional after a greedy `(.+)`, so for 'git@github.com:OWNER/REPO.git' group(2) greedily captures 'REPO.git' and the optional group matches empty. The auto-detected repo therefore keeps the '.git' suffix, producing an API URL repos/OWNER/REPO.git/issues that 404s. (The sibling shell script github_automation.sh:56 explicitly strips '.git' with sed, showing the intended behavior.)",
      "recommendation": "Strip the suffix or make the group mandatory/non-greedy, e.g. capture `(.+?)(?:\\.git)?$` for the repo and rstrip('.git').",
      "status": "survived"
    },
    {
      "id": "s2c7l0-002",
      "location": "scripts/automation/create_github_issues.py:62-69",
      "class": "bug",
      "severity": "medium",
      "evidence": "get_repo_from_git uses `re.match(r\"^git@github\\.com:(.+)/(.+)(\\.git)?$\", ...)` (and the analogous HTTPS pattern). The second `(.+)` is greedy and the `(\\.git)?` group is optional, so for `git@github.com:OWNER/REPO.git` group(2) captures `REPO.git` (the `.git` is never stripped). The returned repo name therefore includes the `.git` suffix and the GitHub API URL `repos/{owner}/{repo}` is wrong. (The sibling shell script github_automation.sh:51-57 explicitly strips `.git`, confirming intent.)",
      "recommendation": "Anchor the `.git` non-optionally for SSH or strip it explicitly, e.g. `re.match(r'^git@github\\.com:(.+?)/(.+?)(?:\\.git)?$', url)` with non-greedy groups, or `repo = group(2).removesuffix('.git')`.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-017",
      "location": "scripts/automation/github_automation/analyze_commit_log.py:112",
      "class": "bug",
      "severity": "medium",
      "evidence": "COMMIT_RE (:11) parses logs in the `%h - %an, %ar : %s` shape produced by github_automation.sh (`--pretty=format:'%h - %an, %ar : %s'` at github_automation.sh:119,130), where the time field is %ar — a RELATIVE string like '2 days ago'. But line 112 parses it with `pd.to_datetime(df['time_ago'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce')` (absolute ISO+tz, comment :111 'exact ISO format'). All relative strings coerce to NaT, dropna (:113) empties the frame, and the time-distribution plot (:116-124) is built on no data.",
      "recommendation": "Either change the log generator to emit `%ad`/`%aI` (absolute ISO date), or parse the relative %ar format instead of an ISO format string.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-010",
      "location": "scripts/automation/github_automation/analyze_commit_log.py:112",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Time parsing uses `pd.to_datetime(df['time_ago'], format='%Y-%m-%d %H:%M:%S %z', errors='coerce')`, but the companion log generators emit git's relative `%ar` format ('2 days ago') — see scripts/automation/github_automation.sh:119,127,130 (`--pretty=format:\"%h - %an, %ar : %s\"`), which is also the shape the COMMIT_RE/time group at line 11 captures. Relative strings never match the ISO format, so every value coerces to NaT and `df.dropna(subset=['time_ago'])` (line 113) empties the frame, leaving the time-distribution plot (lines 116-124) blank/meaningless.",
      "recommendation": "Generate the source log with an absolute timestamp (`%ai`/`%aI`) to match the parser, or parse relative times appropriately; assert non-empty rows after dropna and warn instead of silently producing an empty plot.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-001",
      "location": "scripts/inspect_checkpoint.py:10",
      "class": "security",
      "severity": "medium",
      "evidence": "main() calls torch.load(ckpt_path, map_location='cpu') on an arbitrary path supplied as argv[1] with no weights_only=True. torch.load uses Python's pickle, which executes arbitrary code embedded in the file during unpickling. This script is explicitly a tool to inspect *any* .pt/.ckpt file (including third-party/downloaded checkpoints), so a maliciously crafted checkpoint achieves arbitrary code execution when inspected.",
      "recommendation": "Pass weights_only=True to torch.load (PyTorch>=2.0) for inspection, or load with a safe unpickler / pickletools-based metadata reader. Document that only trusted checkpoints should be loaded.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-002",
      "location": "scripts/inspect_pt_file.py:10",
      "class": "security",
      "severity": "medium",
      "evidence": "torch.load(pt_path, map_location='cpu') is called on argv[1] (any user-supplied .pt path) without weights_only=True. Pickle deserialization of an untrusted .pt file allows arbitrary code execution. The script's stated purpose is inspecting arbitrary .pt files, maximizing the chance of pointing it at an untrusted artifact.",
      "recommendation": "Use torch.load(..., weights_only=True) or a restricted unpickler when the goal is only to enumerate keys/preview content; never unpickle untrusted files with default settings.",
      "status": "survived"
    },
    {
      "id": "s2re4-002",
      "location": "scripts/partial_checkpoint_full_pipeline_script.py:60-67",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Lines 56-62 carefully detect a relative config path ('rna_predict/conf' or 'conf') under PROJECT_ROOT and store it in config_path_selected, exiting with [UNIQUE-ERR-HYDRA-CONF-PATH-NOT-FOUND] on failure. Line 67 then ignores config_path_selected entirely and hardcodes hydra.initialize(config_path=\"/Users/tomriddle1/RNA_PREDICT/rna_predict/conf\"). The portability logic is dead code; the script only runs on the original author's machine ('/Users/tomriddle1') and crashes everywhere else despite the [HYDRA-PROJECT-RULE] comment.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re2-006",
      "location": "scripts/partial_checkpoint_full_pipeline_script.py:67; rna_predict/pipeline/stageD/memory_optimization/test_memory.py:90",
      "class": "bug",
      "severity": "medium",
      "evidence": "Two shipped (non-tests/ tree) scripts hardcode hydra config_path to the developer-only absolute path \"/Users/tomriddle1/RNA_PREDICT/rna_predict/conf\": scripts/partial_checkpoint_full_pipeline_script.py:67 (hydra.initialize) and rna_predict/pipeline/stageD/memory_optimization/test_memory.py:90 (get_config). The latter lives inside the installed package (rna_predict/pipeline/...), not under tests/, so it ships in the wheel yet is unrunnable on any other machine. Distinct location set from train.py (s2re2-001) and compute_ground_truth_angles.py:53.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c7l0-009",
      "location": "scripts/run_mutation_tests.sh:44",
      "class": "bug",
      "severity": "medium",
      "evidence": "`if ! $CMD 2>&1 | tee \"$LOG_FILE\"; then` evaluates the exit status of the last pipeline element (`tee`), not of `mutatest`. The script has no `set -o pipefail`, so a failing mutatest run (nonzero exit) is masked whenever tee succeeds (almost always), and the entire error-classification branch (lines 45-66) is effectively unreachable, while the final 'completed successfully' path (lines 70-74) runs on failures.",
      "recommendation": "Add `set -o pipefail` at the top, or capture mutatest's status via PIPESTATUS, e.g. `$CMD 2>&1 | tee \"$LOG_FILE\"; status=${PIPESTATUS[0]}; if [ \"$status\" -ne 0 ]; then ...`.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-004",
      "location": "scripts/test_utils/hypot_test_gen.py:327-336",
      "class": "security",
      "severity": "medium",
      "evidence": "run_hypothesis_write builds full_cmd = f\"hypothesis write {command}\" and executes subprocess.run(full_cmd, shell=True, ...). 'command' embeds method_path/module_path derived from the scanned file's filesystem path/stem (construct_module_path/generate_*_variants). A Python file whose name or directory contains shell metacharacters (e.g. ';', '$(...)', backticks) yields command injection executed via the shell when generating tests for an attacker-supplied folder.",
      "recommendation": "Invoke subprocess with an argument list (shell=False) and pass the hypothesis arguments as discrete tokens; validate/whitelist module path characters before interpolation.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-005",
      "location": "scripts/test_utils/hypot_test_gen.py:785-794",
      "class": "security",
      "severity": "medium",
      "evidence": "combine_and_cleanup_tests constructs ruff commands as f-strings interpolating combined_filepath (derived from the input file stem) and runs subprocess.run(cmd, shell=True, ...). A controlled file stem containing shell metacharacters or spaces breaks out of the intended command, enabling command injection / argument splitting when processing an untrusted target folder.",
      "recommendation": "Replace shell=True f-string commands with list-form subprocess calls (e.g. ['ruff','check',str(combined_filepath)]) so the path is passed as a single, non-interpreted argument.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-006",
      "location": "scripts/test_utils/mark_slow_tests.py:6",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "SLOW_TEST_THRESHOLD = 1.0 (:6) and the implied intent (mark tests slower than 1s) are never realized: the code contains no timing/measurement logic. SlowTestMarker.visit_FunctionDef marks EVERY function named test_* (is_test_function at :25-27) regardless of runtime, so the threshold constant is dead/misleading and the marker would (if it ran) flag all tests, not slow ones.",
      "recommendation": "Either feed the marker from actual pytest --durations data keyed on SLOW_TEST_THRESHOLD, or drop the threshold constant and rename the tool to reflect that it marks all tests.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-008",
      "location": "scripts/test_utils/run_failing_tests.sh:1-6,328",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The filename and the Stage-1 inventory summary ('Shell script to run and report on failing tests') promise running failing tests, but the script runs the ENTIRE suite: `test_files=$(find tests -type f -name 'test_*.py')` (:328) then `pytest $test_files ...` (:404). Its actual purpose is progressive-coverage gating across a Kaggle timeline (header :3-6). Nothing selects or re-runs only failing tests.",
      "recommendation": "Rename to reflect its real role (e.g. run_progressive_coverage.sh) or add `--lf/--last-failed` selection if running only failures is the intent.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-008",
      "location": "scripts/test_utils/run_failing_tests.sh:317-319",
      "class": "bug",
      "severity": "medium",
      "evidence": "get_coverage_goal() prints several human-readable status lines (e.g. 'Days since last run: 1') and finally `echo $COVERAGE_GOAL` such as '85.00'. The caller captures the ENTIRE multi-line output into COVERAGE_GOAL, then extracts with `grep -o '[0-9]\\+' | tail -1`. For a fractional goal like 85.00 the last digit-run is the decimal fraction '00', so COVERAGE_GOAL becomes '00' and the pytest invocation runs `--cov-fail-under=00`, effectively setting a 0% gate and disabling the coverage threshold the script exists to enforce.",
      "recommendation": "Return only the numeric goal from get_coverage_goal (write status to stderr, value to stdout) and parse the integer part explicitly, e.g. `printf '%.0f' \"$RAW_GOAL\"`, instead of `grep -o '[0-9]\\+' | tail -1`.",
      "status": "survived"
    },
    {
      "id": "s2re1-003",
      "location": "setup.py:1-32 vs pyproject.toml:5-38",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "A second, conflicting build manifest exists. setup.py declares name=\"rna_predict\", version=\"1.0.0\", python_requires=\">=3.8\" and a totally different dependency set (numpy, scipy, pandas, tqdm, PySimpleGUI). pyproject.toml declares name=\"rna-predict\", version=\"2.0.8\", requires-python=\">=3.10\" with build-backend=setuptools.build_meta. Because PEP 621 [project] metadata in pyproject takes precedence under setuptools.build_meta, setup.py is dead/misleading config: its version, python floor and deps (e.g. PySimpleGUI) are never applied, yet it advertises a different package identity to any reader/tool that parses it.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c7l2-011",
      "location": "setup.py:6",
      "class": "doc_drift",
      "severity": "medium",
      "evidence": "setup.py declares version='1.0.0' (:6) while pyproject.toml:7 declares version='2.0.8' and rna_predict/VERSION contains 2.0.8. Two coexisting build configs with contradictory version metadata; whichever build path is used yields an inconsistent package version.",
      "recommendation": "Pick one source of truth (pyproject.toml is canonical per the setuptools backend) and either delete setup.py or sync its version/metadata to 2.0.8.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-011",
      "location": "setup.py:6-21",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "setup.py declares `version=\"1.0.0\"` and an install_requires that lists GUI/screen-finder deps (opencv-python, mss, PySimpleGUI, pyautogui) as CORE requirements while omitting the pipeline's real runtime deps (no hydra-core, pytorch-lightning, omegaconf). The authoritative build config is pyproject.toml (version 2.0.8, verified). Two divergent build-metadata sources for the same package cause version/dependency drift and ambiguous installs depending on which path the toolchain uses.",
      "recommendation": "Consolidate on pyproject.toml (delete or reduce setup.py to a shim), or sync setup.py's version and dependency list with pyproject.toml; move GUI-only deps to an optional extra.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0022",
      "location": ".augement_code_rules",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The rules filename is misspelled `.augement_code_rules` (should be 'augment'), and the file's own internal references call it `.augment coderules` / `AUGMENT CODE_RULES` (lines 337-338, 357), none of which match the actual filename. Augment Code looks for a specific rules filename, so the misspelled name likely means the tool never loads it.",
      "recommendation": "Rename to the filename Augment Code expects and make the in-file self-references match.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0019",
      "location": ".coverage_config.json.bak:1",
      "class": "other",
      "severity": "low",
      "evidence": ".coverage_config.json.bak is byte-identical to .coverage_config.json (both 56 lines, same content) — a committed editor backup file. Both are also listed in .gitignore (.gitignore:195,198,199) yet are tracked in the repo, an internal contradiction.",
      "recommendation": "Delete the .bak backup from version control and either untrack or stop gitignoring the canonical .coverage_config.json.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0017",
      "location": ".coverage_config.json:31-38",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The phased coverage schedule ends at final_submission end_date 2025-05-29 and last_updated is 2025-05-10; all phase windows are in the past relative to the current date (2026-06-17). The phase-driven targets are stale and no longer actionable, and current_coverage 89.99 is a hard-coded snapshot with no link to actual measured coverage.",
      "recommendation": "Update or retire the phase schedule; derive current_coverage from a live coverage run rather than a static value.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0018",
      "location": ".coverage_config.json:51-53",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "module_categories.utility_modules lists `rna_predict.scripts` for coverage tracking, but rna_predict/scripts/ contains only __init__.py and hypot_test_gen.py (verified by ls), and .coveragerc explicitly omits `*/scripts/*` from coverage (.coveragerc:8). Tracking a scripts package that coverage is told to omit is contradictory.",
      "recommendation": "Remove rna_predict.scripts from the tracked categories or stop omitting scripts in .coveragerc — pick one consistent policy.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-014",
      "location": ".coveragerc:11; .coverage_config.json:3",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Coverage enforcement is contradictory: .coveragerc sets `fail_under = 0` (no gate), while .coverage_config.json declares base_coverage 80 / current 89.99 / phase targets up to 95 (lines 3-5,31-38). The effective CI gate (`make test` -> pytest --cov-config .coveragerc, Makefile:48) enforces nothing, so the documented coverage policy is unenforced.",
      "recommendation": "Either wire .coverage_config.json thresholds into the actual gate or set .coveragerc fail_under to the intended floor; reconcile the two sources of truth.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0039",
      "location": ".github/FUNDING.yml:12",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The custom funding list includes the placeholder URL `'https://www.example.com/sponsor2'` alongside the real GitHub sponsors link. example.com is a non-functional placeholder that would render as a broken sponsor link on the repository.",
      "recommendation": "Remove the example.com placeholder entry.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0038",
      "location": ".github/dependabot.yml:3",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Dependabot is configured only for the `github-actions` ecosystem. The project has substantial Python dependencies (requirements*.txt, pyproject.toml) and a Node dependency set (package.json), none of which Dependabot is configured to monitor, so security/version updates for the actual application dependencies are never proposed.",
      "recommendation": "Add `pip` (and optionally `npm`) update entries to dependabot.yml to cover the real dependency surfaces.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-008",
      "location": ".github/init.sh:62",
      "class": "bug",
      "severity": "low",
      "evidence": "Line reads `echo \"Applying ${template} template to this project\"}` — a stray closing brace `}` is appended outside the quoted string. Bash prints it literally (output ends with `...project}`), indicating a copy/paste corruption; the following line then unconditionally execs `./.github/templates/${template}/apply.sh`.",
      "recommendation": "Remove the trailing `}` on line 62.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-012",
      "location": ".github/rename_project.sh:24-31",
      "class": "security",
      "severity": "low",
      "evidence": "The loop `for filename in $(git ls-files)` is unquoted (word-splits on whitespace) and runs `sed -i \"s/$original_author/$author/g\" $filename` with both the replacement values ($author/$name/$urlname/$description, from -a/-n/-u/-d args) and $filename unquoted and unescaped. Filenames with spaces break, and replacement values containing `/` or other sed metacharacters corrupt the substitution or could alter unintended files. In CI these values come from the repo owner/name (rename_project.yml:36), limiting external attacker control.",
      "recommendation": "Quote $filename, iterate safely (e.g. git ls-files -z | while IFS= read -r -d ''), and escape sed metacharacters in substitution values.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-009",
      "location": ".github/workflows/main.yml:104,38-47; .github/workflows/release.yml:40; .github/workflows/rename_project.yml:38",
      "class": "security",
      "severity": "low",
      "evidence": "Third-party GitHub Actions are pinned only to mutable major-version tags rather than immutable commit SHAs: codecov/codecov-action@v5 (main.yml:104), actions/upload-artifact@v4 (main.yml:44), softprops/action-gh-release@v2 (release.yml:40), stefanzweifel/git-auto-commit-action@v5 (rename_project.yml:38). A moved/compromised tag could inject code into CI, which here holds write tokens and (release.yml) the PyPI token.",
      "recommendation": "Pin third-party actions to full commit SHAs and use Dependabot (already configured for github-actions in .github/dependabot.yml) to bump them.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-006",
      "location": ".github/workflows/main.yml:40",
      "class": "bug",
      "severity": "low",
      "evidence": "`pip-audit -r requirements.txt --severity high --exit-code 1 ...` passes a `--severity` flag. pip-audit's CLI does not expose a `--severity` option (it has no built-in severity filtering), which would make the step error out on argument parsing. UNVERIFIED against the exact pip-audit version installed in CI (no version pin), so flagged as a likely-invalid flag rather than confirmed.",
      "recommendation": "Confirm the installed pip-audit version's supported flags; remove `--severity high` (filter severities downstream from the JSON output) or pin a pip-audit version that supports the intended behavior.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-007",
      "location": ".github/workflows/main.yml:50-70",
      "class": "security",
      "severity": "low",
      "evidence": "The linter job auto-applies `ruff check --fix --unsafe-fixes` (:53,:64) then configures a bot identity and runs `git add -A`/`git commit`/`git push` (:57-59,:67-70) inside CI, twice, under `continue-on-error: true`. This performs automated writes/pushes to the repository using GITHUB_TOKEN as a side effect of CI, masks failures, and applies *unsafe* (potentially behavior-changing) auto-fixes without human review before pushing.",
      "recommendation": "Move auto-fix to a gated, reviewable flow (open a PR rather than direct push), drop `--unsafe-fixes` from CI auto-commit, and avoid `continue-on-error` hiding push/commit failures.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0040",
      "location": ".github/workflows/mkdocs.yml:23",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The docs-deploy workflow pins Python 3.10 and installs only `mkdocs mkdocs-material`, while the main CI uses Python 3.11 (.github/workflows/main.yml:21) and mkdocs.yml relies on pymdownx extensions / arithmatex (mkdocs.yml:16-28). pymdown-extensions is declared in pyproject dev (pyproject.toml:53) but is not explicitly installed in this workflow (it ships transitively with mkdocs-material, so the build is fragile to that transitive dependency and the Python version is inconsistent with the rest of CI).",
      "recommendation": "Explicitly install pymdown-extensions and align the Python version with the main CI matrix to make doc builds reproducible.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-022",
      "location": ".github/workflows/release.yml:38",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Release builds use `python setup.py sdist bdist_wheel` (lines 38 and 63). setup.py does exist (verified `ls setup.py`), so this runs, but `setup.py` invocations are deprecated by setuptools/PyPA in favor of `python -m build` given the project already declares a PEP517 build-backend (pyproject.toml:1-3). The direct setup.py path risks breakage on newer setuptools.",
      "recommendation": "Switch to `python -m build` (add `build` to the install step) for a PEP 517-compliant release flow consistent with the pyproject build-backend.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-015",
      "location": ".gitignore:195-199,220-221,226-227",
      "class": "bug",
      "severity": "low",
      "evidence": "Files that are committed and tracked are also listed in .gitignore: `.coverage_config.json` (:195,:198), `.coverage_config.json.bak` (:199), `package.json` (:221), and even `.gitignore` itself (:226-227). Because they are already tracked the ignore has no effect now, but it is misleading and risks accidental loss/non-tracking of regenerated copies and confuses contributors about which files are canonical.",
      "recommendation": "Remove ignore entries for files intended to be tracked (package.json, .coverage_config.json*, .gitignore), or untrack them deliberately if they are meant to be local-only.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-017",
      "location": ".gitignore:220-227",
      "class": "design_defect",
      "severity": "low",
      "evidence": ".gitignore lists already-tracked files: `package-lock.json` (220), `package.json` (221) — both are committed config files (package.json is an assigned tracked file) — and `.gitignore` itself appears twice (226-227). gitignore has no effect on tracked files, so these entries are inert and the self-ignore of .gitignore is nonsensical, signaling accidental edits. .coverage_config.json is likewise ignored (195,198) yet committed.",
      "recommendation": "Remove ignore entries for files that are intentionally tracked, or `git rm --cached` them if they should be untracked; delete the duplicate/self `.gitignore` lines.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0036",
      "location": ".gitignore:226-227",
      "class": "design_defect",
      "severity": "low",
      "evidence": ".gitignore lists `.gitignore` itself (twice, lines 226-227). A file cannot meaningfully ignore itself once tracked; this is dead/nonsensical configuration. The file also contains many duplicated entries (e.g. `.DS_Store` repeated ~7 times across lines 141-152, `rna_predict/.DS_Store` twice, `.coverage_config.json` twice).",
      "recommendation": "Remove the self-referential .gitignore entries and de-duplicate the file.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0044",
      "location": ".gitmodules:1-3",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The RooFlow submodule is declared (path RooFlow, url https://github.com/ImmortalDemonGod/RooFlow.git) but is uninitialized — the working tree RooFlow/ is an empty directory (verified by ls; contains no files). A fresh clone without `--recurse-submodules` leaves RooFlow empty, and nothing in the build/runtime documents that this external dependency is required, so its purpose and necessity are unclear.",
      "recommendation": "Either initialize/commit the submodule pointer with documentation of why it is needed, or remove the RooFlow submodule if it is unused tooling.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0024",
      "location": ".windsurfrules:2",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The debugging rule points to `docs/comprehensive_debugging_guide.md`, but the file actually lives at docs/guides/best_practices/debugging/comprehensive_debugging_guide.md (per the inventory at audit/01-understanding.md and mkdocs.yml:46). The referenced top-level path does not exist.",
      "recommendation": "Update the reference to the real docs path.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0026",
      "location": ".windsurfrules:38",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Version-control rule states 'Do not commit changes to version control.', which directly contradicts the CI workflow that auto-commits and pushes ruff fixes on every push/PR (.github/workflows/main.yml:56-59,68-70). The guidance and the automation disagree.",
      "recommendation": "Reconcile the policy: either stop CI auto-commits or relax the rule for the CI bot.",
      "status": "survived"
    },
    {
      "id": "s2re2-005",
      "location": "MANIFEST.in:4-5",
      "class": "design_defect",
      "severity": "low",
      "evidence": "MANIFEST.in does `graft tests` and `graft rna_predict`, which recursively include the entire test suite and every binary blob under the package tree in the source distribution. rna_predict/ contains large vendored binaries — rna_predict/dataset/preprocessing/dssr-basic-linuxMacWindows-v2.5.3.zip plus rna_predict/dataset/preprocessing/dssr/*.zip (linux/macOS/windows DSSR archives) and example .cif/.pt/.pdb fixtures — so the built sdist bundles multi-MB third-party DSSR archives (whose redistribution licensing is also questionable) and the full test tree. No MANIFEST.in finding exists in the current set.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re3-007",
      "location": "MANIFEST.in:5",
      "class": "design_defect",
      "severity": "low",
      "evidence": "MANIFEST.in contains 'graft tests' (:5), which bundles the entire tests/ tree into the built sdist/wheel. That ships ~223 test files plus large fixtures and the developer-only absolute-path tests (e.g. tests carrying '/Users/tomriddle1/...') into the published package, bloating the distribution and leaking dev-environment paths to consumers. The package distribution config has no existing finding.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c0l0-019",
      "location": "Makefile:107",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The `switch-to-poetry` target runs `poetry init --no-interaction --name=a_flask_test --author=ImmortalDemonGod`, hardcoding the unrelated template name 'a_flask_test' as the project name. If executed it would mislabel the package.",
      "recommendation": "Use the actual project name (rna-predict) in the poetry init invocation, or remove the obsolete switch-to-poetry target.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0010",
      "location": "Makefile:107-110",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "The switch-to-poetry target runs `poetry init --no-interaction --name=a_flask_test --author=ImmortalDemonGod` (line 107) — a leftover flask-template placeholder name — and appends a poetry script `rna_predict = 'rna_predict.__main__:main'` (line 110) targeting the same non-existent __main__ module (see s2c0l2-0003).",
      "recommendation": "If poetry support is wanted, set name to rna-predict and a valid entry point; otherwise remove the switch-to-poetry target.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0012",
      "location": "Makefile:2-3",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`USING_POETRY=$(shell grep \"tool.poetry\" pyproject.toml && echo \"yes\")` runs unconditional grep whose stdout (the matched line, if any) leaks into every `make` invocation; combined with `.ONESHELL`, this prints grep output as noise. pyproject.toml has no [tool.poetry] section, so USING_POETRY is always empty — the poetry branches in install/show/virtualenv are dead.",
      "recommendation": "Suppress grep output with `grep -q ... && echo yes` and remove the unreachable poetry branches.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0011",
      "location": "Makefile:28-30",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The `fmt` target's help text is `## Format code using black & isort.` but its recipe only runs `$(ENV_PREFIX)isort rna_predict/` — black is never invoked. Separately the project uses ruff for formatting/import-sorting elsewhere (Makefile:43-44, .github/workflows/main.yml:73-79), so black/isort here are inconsistent with the actual toolchain.",
      "recommendation": "Update the help text to match the recipe, or standardize formatting on ruff and remove the redundant black/isort target.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-010",
      "location": "mkdocs.yml:30-31",
      "class": "security",
      "severity": "low",
      "evidence": "`extra_javascript: - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js` loads remote JavaScript into the published docs site from a third-party CDN with no Subresource Integrity hash and only a major-version range. A compromised/poisoned CDN asset would execute arbitrary JS for every docs visitor.",
      "recommendation": "Pin to an exact version and add an SRI integrity hash, or vendor MathJax locally under docs/.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-020",
      "location": "mkdocs.yml:81",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Nav entry 'MP-NeRF Integration: pipeline/stageC/Unified, Comprehensive Plan for Integrating MP-NeRF into Stage C.md' references a filename containing a literal comma and spaces. The Stage-1 inventory flags this exact file as UNRESOLVED (audit/01-understanding.md:48), and the on-disk name uses a non-breaking space variant; a mismatch will trigger mkdocs 'doc not found in nav'/missing-file warnings during `mkdocs build` (Makefile:99) and gh-deploy (.github/workflows/mkdocs.yml).",
      "recommendation": "Rename the doc to an ASCII-safe filename without commas/special spaces and update the nav entry to match exactly.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-013",
      "location": "mutatest.ini:3",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Comment states 'This configuration requires coverage==5.5 and pytest-cov==2.12.1', but requirements-test.txt:2,9 pin `coverage>=7.6.12` and `pytest-cov>=6.0.0`, and pyproject.toml:44,51 do the same. The stated mutatest prerequisite is years out of date relative to the actual pinned versions.",
      "recommendation": "Update or remove the stale version comment in mutatest.ini to reflect the coverage/pytest-cov versions actually used.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0033",
      "location": "pyproject.toml:41-76",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Two distinct dev dependency mechanisms coexist with different contents: [project.optional-dependencies].dev (lines 42-54: black, coverage, flake8, gitchangelog, isort, mkdocs, mypy, pytest, pytest-cov, pytest-xdist, pymdown-extensions) and [dependency-groups].dev (lines 63-76: cosmic-ray, mkdocs, mutatest, pydeps, pytest-asyncio, pytest-cov, pytest-faulthandler, pytest-memprof, pytest-timeout, snoop, types-requests). The two 'dev' sets barely overlap, so which tools you get depends on installer (pip extras vs uv groups).",
      "recommendation": "Merge into a single dev dependency declaration to avoid installer-dependent dev environments.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-015",
      "location": "pyproject.toml:42-76",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Two parallel and inconsistent dev dependency declarations exist: `[project.optional-dependencies].dev` (lines 42-54: black, coverage, flake8, gitchangelog, isort, mkdocs, mypy, pytest, pytest-cov, pytest-xdist, pymdown-extensions) and `[dependency-groups].dev` (lines 63-76: cosmic-ray, mkdocs, mutatest, pydeps, pytest-asyncio, pytest-cov, pytest-faulthandler, pytest-memprof, pytest-timeout, snoop, types-requests). The two sets barely overlap, so `pip install .[dev]` (used by Makefile:26 and CI) and the uv dependency-group give different environments.",
      "recommendation": "Consolidate dev dependencies into a single canonical list to avoid environment skew between pip and uv installs.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-016",
      "location": "pyproject.toml:8",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "`description = \"Add your description here\"` is the unedited template placeholder. This string is published as the package summary on any wheel/sdist build (release.yml).",
      "recommendation": "Replace with a real one-line project description.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0015",
      "location": "pytest.ini:15",
      "class": "design_defect",
      "severity": "low",
      "evidence": "addopts includes `-p no:warnings` (line 15) which disables the pytest warnings plugin entirely, yet the file also defines a `filterwarnings` block (lines 27-31) intended to ignore specific warning categories. With the warnings plugin disabled, the filterwarnings rules are inert/contradictory.",
      "recommendation": "Drop `-p no:warnings` and rely on filterwarnings, or remove the filterwarnings block since the plugin is disabled.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0029",
      "location": "requirements-test.txt:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "requirements-test.txt overlaps and diverges from pyproject dev declarations: it pins runtime dep hydra-core==1.3.2 in a test-only file (line 8) and lists black/gitchangelog/memory-profiler/pytest-asyncio/pytest-timeout that are spread differently across pyproject [project.optional-dependencies].dev and [dependency-groups].dev. There is no single authoritative test-dependency set.",
      "recommendation": "Consolidate test deps into one location (pyproject dev extra) and drop the redundant requirements-test.txt or generate it from pyproject.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-011",
      "location": "requirements.txt:14-18",
      "class": "security",
      "severity": "low",
      "evidence": "Several runtime dependencies are completely unpinned: `pyautogui`, `opencv-python`, `Pillow`, `mss`, `PySimpleGUI`, `protenix` (requirements.txt:14-18,20). Pillow and opencv in particular have a history of CVEs; with no version floor or ceiling, builds are non-reproducible and may silently pull a vulnerable or yanked release. Note PySimpleGUI here vs pyproject.toml dearpygui (dependency-set drift).",
      "recommendation": "Pin minimum (and ideally maximum) versions for all dependencies and reconcile requirements.txt with pyproject.toml; use pip-audit (already in CI) against the curated list.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0034",
      "location": "rna_predict/VERSION:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Version is tracked in two places: rna_predict/VERSION (2.0.8) and a static pyproject.toml version (2.0.8 at pyproject.toml:7). The setuptools build backend reads pyproject's static value, not VERSION, while `make release` writes the new version only to rna_predict/VERSION and HISTORY (Makefile:87-93) — so a release will bump VERSION but leave pyproject (the value actually built/published) unchanged, causing the two to drift.",
      "recommendation": "Use a single version source (e.g. setuptools dynamic version from VERSION, or have `make release` also update pyproject.toml).",
      "status": "survived"
    },
    {
      "id": "s2c0l0-021",
      "location": "rna_predict/benchmarks/benchmark.py:166-173",
      "class": "bug",
      "severity": "low",
      "evidence": "`benchmark_decoding_latency_and_memory(N_atom_list=[128,256,512], N_token_list=[32,64,128], ...)` and `benchmark_input_embedding` (lines 315-322) use mutable list literals as default arguments — the classic Python mutable-default anti-pattern. Not currently mutated in-body, so no live corruption observed, but it is a latent footgun.",
      "recommendation": "Use `None` defaults and assign the lists inside the function, mirroring the BenchmarkConfig dataclass (which already uses default_factory).",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0041",
      "location": "rna_predict/benchmarks/benchmark.py:167",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Several public functions use mutable list literals as default arguments — e.g. `benchmark_decoding_latency_and_memory(N_atom_list=[128,256,512], N_token_list=[32,64,128], ...)` (lines 167-168) and `benchmark_input_embedding(...)` (lines 316-317). Mutable default arguments are a well-known Python footgun (shared across calls); BenchmarkConfig (lines 33-34) already uses field(default_factory=...) correctly, so the top-level functions are inconsistent with the file's own pattern.",
      "recommendation": "Default these parameters to None and build the lists inside the function (or delegate to BenchmarkConfig defaults).",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0042",
      "location": "rna_predict/benchmarks/benchmark.py:36",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "The benchmark defaults to `device=\"cuda\"` throughout (BenchmarkConfig.device line 36, and every benchmark_* signature). While resolve_device() (line 15) falls back to CPU when CUDA is absent, running the __main__ entry (lines 393-399) on the CPU-only CI/dev environments the project targets will silently benchmark CPU performance under a 'cuda' label, which can mislead the naive-vs-optimized comparison the script exists to produce (audit/01-understanding.md:16).",
      "recommendation": "Default device to 'auto'/'cpu' or print the actually-resolved device prominently in the benchmark output headers.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-data-yaml-numworkers-and-element-size-conflict",
      "location": "rna_predict/conf/data/default.yaml:5-10 vs rna_predict/conf/config_schema.py:1325-1330",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "data/default.yaml sets num_workers: 8, but DataConfig.num_workers default=0 with help 'Number of DataLoader workers (set to 0 for debugging device mismatch)' (config_schema.py:1330) — the YAML reintroduces the very value the schema warns against. Additionally DataConfig carries two fields for the element-embedding size with conflicting defaults: C_element=128 and ref_element_size=4 (config_schema.py:1325,1327); the YAML then makes ref_element_size resolve to 128 via ${shared.ref_element_size}, so the schema default of 4 is dead/misleading and duplicates C_element.",
      "recommendation": "Reconcile num_workers guidance vs the 8 override; collapse C_element/ref_element_size (and C_char/ref_atom_name_chars_size) into a single field or document the distinction, and set the schema default to the real value (128/256).",
      "status": "survived"
    },
    {
      "id": "s2c1l0-pairformer-cs-zero",
      "location": "rna_predict/conf/model/stageB_pairformer.yaml:32",
      "class": "design_defect",
      "severity": "low",
      "evidence": "stageB_pairformer.yaml sets c_s: 0 ('No single representation in pair stack') overriding PairformerConfig.c_s (schema default 8, config_schema.py:657). A single-representation dimension of 0 would yield zero-width Linear/embedding layers if any consumer constructs modules from c_s; the comment asserts intent but no guard prevents misuse downstream.",
      "recommendation": "Verify no module builds layers from cfg c_s here; if the pair stack truly has no single rep, use a clearly-handled sentinel (e.g. null) rather than 0, or document/assert that consumers ignore c_s in this config.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-stagec-angle-repr-mismatch",
      "location": "rna_predict/conf/model/stageC.yaml:18 vs rna_predict/conf/config_schema.py:767-770",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "stageC.yaml sets angle_representation: 'degrees' (line 18) while StageCConfig.angle_representation defaults to 'cartesian' with help text 'cartesian or internal' (config_schema.py:767-770). The schema's documented allowed values do not include 'degrees', so the YAML value is undocumented relative to the schema and there is no validation to catch the divergence.",
      "recommendation": "Reconcile the allowed/expected values: update the schema help (and any consumer) to the actual accepted set, or align the YAML value.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-stagec-config-duplicate",
      "location": "rna_predict/conf/model/stageC_config.yaml:1-37",
      "class": "design_defect",
      "severity": "low",
      "evidence": "stageC_config.yaml is byte-for-byte identical to stageC.yaml (even its header comment reads '# rna_predict/conf/model/stageC.yaml'). Two divergent-by-accident sources of truth for Stage C config invite drift; only stageC.yaml is referenced in default.yaml defaults (line 9), leaving stageC_config.yaml an unreferenced duplicate.",
      "recommendation": "Delete stageC_config.yaml (or make it a documented variant) to keep a single Stage C config.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-stageD-yaml-missing-sigma-outer",
      "location": "rna_predict/conf/model/stageD.yaml:33,86",
      "class": "bug",
      "severity": "low",
      "evidence": "In stageD.yaml the outer model_architecture omits sigma_data (line 33 commented out) while the nested diffusion.model_architecture includes sigma_data (line 86). Consumers reading cfg.model.stageD.model_architecture.sigma_data would hit a missing key, whereas StageDModelArchConfig declares sigma_data (config_schema.py:871). The two architecture blocks are inconsistent, so behavior depends on which path the code reads.",
      "recommendation": "Restore sigma_data in the outer model_architecture (or rely solely on one canonical block) so both architecture views are consistent.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-testdata-seqlen-mismatch",
      "location": "rna_predict/conf/test_data.yaml:9,12",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "test_data.yaml declares sequence: 'GGGUGCUCAGUACGAGAGGAACCGCACCC' (29 nucleotides) at line 9 but sequence_length: 8 at line 12. Any code that trusts sequence_length instead of len(sequence) would mis-size buffers/loops for this test config.",
      "recommendation": "Set sequence_length to match the sequence (29) or remove the redundant field and use len(sequence).",
      "status": "survived"
    },
    {
      "id": "s2c1l2-testdata-seqlen-contradiction",
      "location": "rna_predict/conf/test_data.yaml:9-12",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "test_data.yaml sets sequence: 'GGGUGCUCAGUACGAGAGGAACCGCACCC' (29 nucleotides, labeled '1SCL_A from Kaggle') but immediately declares sequence_length: 8 with comment 'Length for test'. The advertised length contradicts the actual sequence, and config_schema TestDataConfig.sequence_length default is also 8 against its own default sequence 'ACGUACGU' (length 8) — so the field is a stale literal that does not track the real sequence.",
      "recommendation": "Remove the redundant sequence_length field (derive len(sequence) at runtime) or set it to the true length (29) to avoid a misleading constant.",
      "status": "survived"
    },
    {
      "id": "s2re2-008",
      "location": "rna_predict/dataset/loader.py:211",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Sequence parsing does `seq = ast.literal_eval(seq)[0]` to undo a stringified list/tuple stored in a CSV cell. ast.literal_eval on any non-list-shaped string (a bare RNA sequence like 'GGAC') raises ValueError/SyntaxError, and indexing [0] assumes the literal is a non-empty subscriptable — a malformed/empty literal yields IndexError/TypeError. This brittle round-trip of a Python repr through a data file is a fragile data-contract that will crash the loader on perfectly valid plain-sequence inputs; no loader.py:211 entry exists in the current set.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c1l1-006",
      "location": "rna_predict/dataset/loader.py:286",
      "class": "security",
      "severity": "low",
      "evidence": "`coord_dtype = getattr(torch, self.cfg.data.coord_dtype)` resolves a torch attribute from a config-supplied string (cfg.data.coord_dtype, default 'float32' per conf/data/default.yaml:14 and config_schema.py:1334). Since Hydra/OmegaConf allows arbitrary CLI/file overrides, an attacker able to influence the config could set coord_dtype to any attribute name on the torch module; while it is then used as a dtype in torch.full/torch.zeros (so non-dtype attributes likely raise), reflective attribute lookup driven by external input is fragile and could surface unexpected objects. Config is normally trusted, hence low.",
      "recommendation": "Validate coord_dtype against an explicit allow-list of supported dtype strings (e.g. {'float32','float64','float16'}) before getattr, and raise a clear error otherwise.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-loader-unused-embedding-helpers",
      "location": "rna_predict/dataset/loader.py:31-49,322-323",
      "class": "design_defect",
      "severity": "low",
      "evidence": "element_one_hot (lines 31-38) and atom_name_embedding (lines 40-49) are defined to build element/atom-name features, but _load_atom_features fills elem_emb/name_emb with zeros (lines 322-323, 'let model handle embedding') and never calls them. The helper functions are dead code and the returned embeddings carry no information despite their declared sizes.",
      "recommendation": "Remove the unused helpers or wire them into _load_atom_features if real embeddings are intended.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-calc-dihedral-noop-roundtrip",
      "location": "rna_predict/dataset/preprocessing/angles.py:176-179",
      "class": "bug",
      "severity": "low",
      "evidence": "_calc_dihedral computes phi = np.arccos(cos_angle) (radians) and returns np.deg2rad(np.degrees(phi)). deg2rad(degrees(x)) is an identity, so this is a pointless round-trip that obscures intent (and signals possible confusion about whether the result is degrees or radians).",
      "recommendation": "Return phi directly (already in radians) and drop the deg2rad(degrees(...)) wrapper.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-calc-dihedral-noop-conversion",
      "location": "rna_predict/dataset/preprocessing/angles.py:179",
      "class": "design_defect",
      "severity": "low",
      "evidence": "_calc_dihedral computes phi in radians (via np.arccos, line 176) and then returns np.deg2rad(np.degrees(phi)) — a degrees->radians round-trip that is a mathematical no-op. The convoluted conversion obscures the unit contract and invites a future editor to 'fix' it incorrectly; the function already returns radians, matching the docstring claim of radians.",
      "recommendation": "Return phi directly (it is already radians) and drop the np.deg2rad(np.degrees(...)) wrapper.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-backend-default-drift",
      "location": "rna_predict/dataset/preprocessing/angles.py:18 vs rna_predict/dataset/preprocessing/compute_ground_truth_angles.py:8,34",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Default extraction backend is inconsistent across the surface. extract_rna_torsions defaults backend='dssr' (angles.py:18). compute_ground_truth_angles.py's module docstring says 'using the selected backend (default: MDAnalysis)' (line 8) and its argparse default is 'mdanalysis' (line 34), yet that CLI then prefers cfg.extraction_backend when set, and default.yaml:29 sets extraction_backend: dssr — so the documented MDAnalysis default is overridden to dssr at runtime. The advertised default and the effective default disagree.",
      "recommendation": "Make the documented default, the argparse default, the function default, and default.yaml's extraction_backend agree on a single backend, and update the docstring accordingly.",
      "status": "survived"
    },
    {
      "id": "s2c1l1-007",
      "location": "rna_predict/dataset/preprocessing/angles.py:97",
      "class": "security",
      "severity": "low",
      "evidence": "_select_chain interpolates the caller-supplied chain_id directly into an MDAnalysis selection string: `u.select_atoms(f\"(segid {chain_id}) or (chainID {chain_id})\")` (line 97); similarly _safe_select_atom uses `select_atoms(f\"name {name}\")` (line 149). chain_id originates from dataset rows / config (loader.py:399-409 passes row['chain_id'] or cfg.data.chain_id). A chain_id containing MDAnalysis selection-language tokens would alter the atom selection (selection-syntax injection). Impact is limited to which atoms are selected (no code execution), so severity is low, but unsanitized external input flows into a query DSL.",
      "recommendation": "Validate chain_id against an expected pattern (e.g. alphanumeric chain identifiers) before building the selection string, or use MDAnalysis programmatic selection APIs rather than string interpolation.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-interface-unreachable-debug",
      "location": "rna_predict/interface.py:45-54",
      "class": "bug",
      "severity": "low",
      "evidence": "In the RNAPredictor init except block, line 46 'raise ValueError(...) from e' unconditionally raises, but it is followed by dead code (lines 48-54: 'if hasattr(cfg, ...): print(...)' and a second 'raise') that can never execute. The intended diagnostic config dump is unreachable, so the error path never prints the stageB debugging info it appears designed to show.",
      "recommendation": "Move the diagnostic prints (lines 48-53) before the raise, or remove the dead code.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-datautils-sample-csv-unused",
      "location": "rna_predict/kaggle/data_utils.py:134",
      "class": "bug",
      "severity": "low",
      "evidence": "process_test_sequences calls pd.read_csv(sample_csv) at line 134 but discards the result (not assigned, never used). The sample submission is therefore neither used as a template nor validated; the read only incurs I/O and will still raise if the path is bad, but otherwise serves no purpose.",
      "recommendation": "Either use the sample submission (e.g. to validate IDs/columns) or remove the dead read.",
      "status": "survived"
    },
    {
      "id": "s2c1l1-005",
      "location": "rna_predict/kaggle/kaggle_env.py:161-192",
      "class": "security",
      "severity": "low",
      "evidence": "patch_transformers_for_local() monkey-patches `from_pretrained` on AutoConfig/AutoTokenizer/AutoModel globally (lines 174-183) and rewrites any repo id starting with 'zhihan1996/DNA_bert_' to '/kaggle/working/' + repo (lines 170-172), and dynamically imports `transformers_modules.DNA_bert_3.configuration_bert` (lines 186-189) to override BertModel.config_class process-wide. Loading HuggingFace models with custom code modules (trust_remote_code-style dynamic config import) executes code from the model directory; combined with the symlinks created from `/kaggle/input` (lines 141-159) this trusts model artifacts placed in user-attachable dataset paths. Impact is bounded to the Kaggle setup path but broadens the code-execution surface beyond intent (offline inference setup).",
      "recommendation": "Avoid global from_pretrained patching; scope redirection to explicit, validated local paths; do not import custom model code modules from untrusted dataset directories without integrity checks.",
      "status": "survived"
    },
    {
      "id": "s2re3-008",
      "location": "rna_predict/kaggle/kaggle_env.py:241-242",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The except branch hardcodes a stale fallback version: RNA_PREDICT_VERSION = \"2.0.3\" (:242, with warning text at :241) used to build the wheel install path at :244-245 (rna_predict-{VERSION}-py3-none-any.whl). The actual package version is 2.0.8 (pyproject.toml:6-7, rna_predict/VERSION). If the VERSION file read fails, the code silently looks for a 2.0.3 wheel that will never exist, mis-resolving the install. Distinct from the listed rna_predict/VERSION:1 finding.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c1l2-kaggleenv-docstring-import-and-version",
      "location": "rna_predict/kaggle/kaggle_env.py:6,241-242",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The module docstring tells callers to 'from rna_predict.utils.kaggle_env import setup_kaggle_environment' (line 6), but the file lives at rna_predict/kaggle/kaggle_env.py (the real import is rna_predict.kaggle.kaggle_env, as used in data_utils.py:5 and rna_predict.py:40). Also the hardcoded fallback version defaults to '2.0.3' (line 242) while the package VERSION file is 2.0.8 (rna_predict/VERSION), so the fallback wheel name would be stale if the VERSION read ever fails.",
      "recommendation": "Fix the docstring import path to rna_predict.kaggle.kaggle_env and either remove the hardcoded version fallback or keep it in sync with rna_predict/VERSION.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-legacy-module-undefined-globals",
      "location": "rna_predict/kaggle/legacy_feature_engineering_and_modeling.py:17-18,227",
      "class": "bug",
      "severity": "low",
      "evidence": "This .py module executes notebook-cell code at module top level referencing names that are never defined or imported: train_sequences/validation_sequences/train_labels/validation_labels (lines 17-18) and test_sequences (line 227). Importing the module raises NameError immediately. It is classified as source and marked 'legacy/retained'; it is non-importable as written.",
      "recommendation": "Wrap the cells in functions taking the dataframes as parameters (and guard under __main__), or relocate the file out of the importable package as a notebook.",
      "status": "survived"
    },
    {
      "id": "s2re4-007",
      "location": "rna_predict/kaggle/submission_validator.py:134-135",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "The __main__ block comment at line 134 says 'Replace with actual paths or command-line argument parsing', but line 135 unconditionally hardcodes test_file = \"/kaggle/input/stanford-rna-3d-folding/test_sequences.csv\" with no argparse/sys.argv handling. Run standalone outside Kaggle, line 138's existence check fails and it prints the error branch (line 141) without ever validating. Distinct location from existing submission_validator findings (32-35, 40-44/52-55/69).",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re1-005",
      "location": "rna_predict/kaggle/submission_validator.py:135",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The standalone __main__ block hardcodes test_file=\"/kaggle/input/stanford-rna-3d-folding/test_sequences.csv\" with no argparse/env override. Running the validator as a script anywhere but a Kaggle kernel hits the else branch and prints an error; the module is effectively un-runnable as documented in its own __main__ guard. Distinct line from previously-listed submission_validator.py:32-35 and :40-44.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2re5-subval-hardcoded-testfile",
      "location": "rna_predict/kaggle/submission_validator.py:135 (defect: hardcoded non-overridable default; NOT a path-internal trailing space; behavior is guarded/reported at :138-141, not silent)",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "The validator's __main__ block hardcodes test_file = \"/kaggle/input/stanford-rna-3d-folding/test_sequences.csv\" (with a trailing space) as a non-overridable default. Running submission_validator.py directly off-Kaggle, or against a competition whose dataset slug differs, silently targets a nonexistent path. Distinct line/concern from the listed submission_validator.py:32-35 and :40-44 findings.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c1l0-validator-seqcol-hardcoded",
      "location": "rna_predict/kaggle/submission_validator.py:40-44,52-55,69",
      "class": "bug",
      "severity": "low",
      "evidence": "run_sanity_checks resolves the id column robustly via auto_column (lines 40-41) but then accesses test_sequences['sequence'] literally (lines 44,54,69). If the test CSV uses a different sequence column name (the codebase elsewhere accepts 'Sequence'/'seq'/'SEQ' via auto_column, data_utils.py:138), expected_rows/full_id_set/coverage all raise KeyError despite the flexible id handling.",
      "recommendation": "Resolve the sequence column via auto_column(test_sequences, ['sequence','Sequence','seq','SEQ']) for consistency.",
      "status": "survived"
    },
    {
      "id": "s2c1l2-main-demo-stub-vs-docstring",
      "location": "rna_predict/main.py:1-44",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "main.py's docstring calls it 'Entry point for RNA_PREDICT package' for 'demonstrating and testing the RNA structure prediction pipeline', but main() only prints the resolved config and calls demo_run_input_embedding(), which is a stub that just prints 'Now streaming the bprna-spot dataset...' and returns True (lines 34-40) — no pipeline runs. It also registers RNAConfig under the name 'rna_predict_config' (line 15) that is never used (Hydra loads config_name='default'). Per Stage-1 this file is byte-identical demo scaffolding (audit/01-understanding.md:21,30).",
      "recommendation": "Either wire main.py to the real pipeline (run_full_pipeline / RNAPredictor) or relabel it explicitly as a no-op smoke-test demo and drop the unused ConfigStore registration.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-004",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:166-167,208-209",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Production helpers contain hardcoded answers keyed on specific test inputs: seq2dot returns the literal '(.))' when seq == [2,0,3,0] (lines 166-167), and visual_get_bases returns the literal '1,5','2,6','3','4,7,8' when seq == 'AUGCAUGG' (lines 208-209). These short-circuit the real logic for exactly those inputs, masking whether the general code paths are correct and coupling library code to test fixtures.",
      "recommendation": "Delete the input-specific special cases and move the expected values into the test files; rely on the general algorithm for all inputs.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-visualbases-hardcoded-test",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:207-209",
      "class": "design_defect",
      "severity": "low",
      "evidence": "visual_get_bases() begins with `if seq == \"AUGCAUGG\": return \"1,5\",\"2,6\",\"3\",\"4,7,8\"` — a hardcoded return for a specific test sequence, before the general base-index mapping logic.",
      "recommendation": "Remove the hardcoded test branch; rely on the general mapping which already computes the same result.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-attn-unused-params",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:273-302",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Attn.__init__ accepts `expansion_factor=2.0` and `dropout=0.1` and constructs `self.dropout = nn.Dropout(dropout)`, but Attn.forward never applies self.dropout and expansion_factor is never used. The dropout parameter is therefore silently ineffective.",
      "recommendation": "Apply dropout where intended or remove the unused expansion_factor/dropout parameters and the unused self.dropout module.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-attn-scale-doc-drift",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:296-297",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "In Attn.forward the comment states 'Scale the dot product by sqrt of query dimension for better numerical stability' but the code divides by sqrt of sequence length: `sim = einsum(...) / (seq_len**0.5)`. seq_len is the number of tokens, not the query/key feature dimension (query_key_dim), so the scaling does not match the documented intent and is non-standard for scaled dot-product attention.",
      "recommendation": "Decide on the intended scale (typically 1/sqrt(query_key_dim)) and make code and comment agree.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-rfold-constraint-matrix-unused-base",
      "location": "rna_predict/pipeline/stageA/adjacency/RFold_code.py:89-92",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "constraint_matrix() comments read 'Combine all pairs and apply the base_matrix constraint' and 'Apply base matrix constraints while preserving the correct pairs', but the function simply `return constraint` without ever multiplying by or using base_matrix. The module-level base_matrix() helper (:66-72) is defined but never referenced anywhere in the file, so the documented constraint is not applied.",
      "recommendation": "Either apply base_matrix as the comments describe (e.g. `constraint * base_matrix(...)`) or remove the misleading comments and the dead base_matrix() helper.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-006",
      "location": "rna_predict/pipeline/stageA/adjacency/rfold_predictor.py:152-160",
      "class": "bug",
      "severity": "low",
      "evidence": "When required_fields are missing the constructor enters dummy mode and overwrites self.device with `device if device is not None else torch.device('cpu')` (line 157), discarding the already-validated self.device resolved at line 105 from stage_cfg.device. If the predictor was constructed with a config device but no explicit `device` argument, dummy mode silently relocates it to CPU, diverging from the configured device contract the class otherwise enforces (lines 96-103).",
      "recommendation": "In the dummy-mode branch keep the previously resolved self.device (do not reassign), or reassign only when device is explicitly provided.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-025",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/checkpointing.py:88-91",
      "class": "bug",
      "severity": "low",
      "evidence": "checkpoint_blocks silently coerces an out-of-range blocks_per_ckpt: `elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks): blocks_per_ckpt = len(blocks)` (lines 90-91). The upstream openfold contract treats blocks_per_ckpt < 1 as a programming error (ValueError); here a zero/negative value is silently reinterpreted as 'one big chunk', hiding caller misconfiguration that disables the intended activation-checkpointing memory savings.",
      "recommendation": "Raise ValueError for blocks_per_ckpt < 1 (preserving upstream semantics) and only clamp the upper bound to len(blocks), or log a warning when clamping so the misconfiguration is visible.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-checkpointing-silent-clamp",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/checkpointing.py:90-91",
      "class": "design_defect",
      "severity": "low",
      "evidence": "checkpoint_blocks silently clamps an out-of-range blocks_per_ckpt (`< 1 or > len(blocks)`) to len(blocks) with a comment 'Default to using all blocks in one chunk' (lines 90-91). The upstream OpenFold implementation this is vendored from (header :1-2) raises ValueError for the same condition, so an invalid configuration is now masked rather than reported.",
      "recommendation": "Either raise on invalid blocks_per_ckpt (matching upstream) or document this intentional behavioural divergence from the vendored source.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-008",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/__init__.py:51-72",
      "class": "bug",
      "severity": "low",
      "evidence": "__all__ lists 'DenseTrunkConfig' (line 71) but that name is never imported or defined in this package __init__. `from ...primitives import *` will raise AttributeError ('module ... does not define ... DenseTrunkConfig') because every name in __all__ must be resolvable.",
      "recommendation": "Remove 'DenseTrunkConfig' from __all__ or add `from .attention.config_types import DenseTrunkConfig` to actually export it.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-primitives-init-all-missing-densetrunkconfig",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/__init__.py:71",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "primitives/__init__.py declares `\"DenseTrunkConfig\"` in __all__ (line 71) but never imports or defines DenseTrunkConfig (it lives in primitives/attention/config_types.py). `from ...primitives import *` would raise AttributeError for the undefined name, and the export list misrepresents the public API.",
      "recommendation": "Either import DenseTrunkConfig into __init__ before listing it, or remove it from __all__.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-adaln-misplaced-docstrings",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm.py:101-167",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "In _apply_conditioning (line 101) and forward (line 151) the first statement is a print() call; the triple-quoted description blocks that follow (lines 103-112 and 157-167) are therefore expression statements, not docstrings. __doc__ for these methods is None, so the documented Args/Returns are not attached to the functions.",
      "recommendation": "Move the docstrings to be the first statement of each method (before any print) so they are real docstrings.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-011",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/adaptive_layer_norm_utils.py:110-117",
      "class": "perf",
      "severity": "low",
      "evidence": "interpolate_sequence_dim contains an unconditional print(f\"[DEBUG][AdaLN][interpolate_sequence_dim] ...\") at line 117 that executes every time the helper is called (used by adjust_tensor_shapes during AdaLN broadcasting fallbacks), emitting debug output to stdout in production with no flag to disable it.",
      "recommendation": "Replace the print with a logger.debug call or remove it.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-atompair-noop-statement",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/atom_pair_transforms.py:106",
      "class": "design_defect",
      "severity": "low",
      "evidence": "_map_tokens_to_atoms contains the bare statement `config.atom_to_token_idx.shape[1]` (line 106) whose value is computed and discarded, under a comment 'Create gather indices for mapping tokens to atoms'. It is dead leftover from a refactor and does nothing.",
      "recommendation": "Remove the dead expression statement (and the misleading comment) or assign/use the value if it was intended.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-018",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_core.py:126-132",
      "class": "bug",
      "severity": "low",
      "evidence": "In attention(), the manual path applies dropout via F.dropout(attn_weight, p=inputs.attn_weight_dropout_p) (line 129) without passing training=... ; F.dropout defaults to training=True, so when attn_weight_dropout_p > 0 dropout is applied during eval/inference as well as training, corrupting inference attention weights. (The efficient SDPA path at line 105-111 has the same dropout_p-always-applied behavior.)",
      "recommendation": "Thread the module's training flag into the AttentionInputs and pass training=self.training to F.dropout / set dropout_p=0 at eval; or only apply dropout when torch.is_grad_enabled()/self.training.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attncore-dropout-no-training-flag",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_core.py:128-129",
      "class": "bug",
      "severity": "low",
      "evidence": "attention() applies `attn_weight = F.dropout(attn_weight, p=inputs.attn_weight_dropout_p)` without passing `training=`. F.dropout defaults training=True, so when attn_weight_dropout_p>0 dropout is applied even during evaluation/inference, unlike the SDPA fast path which respects module training state.",
      "recommendation": "Pass an explicit training flag (e.g. derive from the owning module's self.training) so dropout is disabled at inference.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-017",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils.py:170-176",
      "class": "bug",
      "severity": "low",
      "evidence": "_determine_chunking indexes q.shape[-4] (line 172) to test 'small batch size' when chunk_size is None. If the query tensor has fewer than 4 dimensions at this point, q.shape[-4] raises IndexError, aborting local attention rather than chunking. The reachable shape of q here (after the small-tensor bypass) is not guaranteed to be 4D.",
      "recommendation": "Guard the index (e.g. use q.dim() check or q.shape[0]) or compute the batch heuristic from a dimension known to exist, falling back to a default chunk size when the tensor rank is smaller than expected.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-attnutils-fix-dim-test-hardcode",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/primitives/attention_utils.py:410-432",
      "class": "design_defect",
      "severity": "low",
      "evidence": "_fix_dimension_mismatch hardcodes branches for `q_dim_2==5 and bias_dim_2==4` and `q_dim_2==4 and bias_dim_2==5` (lines 423,428), and its caller comments 'Check for dimension mismatch at dim 2 (common issue in tests)' (line 472). The bias adaptation is tuned to specific test dimensionalities rather than a general rule.",
      "recommendation": "Replace the hardcoded 4<->5 dim cases with general broadcasting/validation logic and drop the test-oriented comments.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-024",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/shape_adapter.py:93-114",
      "class": "design_defect",
      "severity": "low",
      "evidence": "adapt_tensors_for_addition only resolves non-broadcastable shape mismatches when both tensors have >=6 dims (line 97), using a hardcoded 'transformer case' that mean-reduces tensor_b over dims 3/4/5 (lines 104-112). For any mismatch where the tensors are not >=6D, mismatch_dims is non-empty but the function returns the tensors unchanged (line 114), so the subsequent addition the caller intends will still fail — the 'adapter' provides a false sense of safety and silently mean-collapses real data in the one case it handles.",
      "recommendation": "Make the general path explicit: raise a clear error on genuinely non-broadcastable shapes instead of returning unchanged tensors, and replace the dimension-specific mean/expand hack with a documented, validated reshape contract.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-023",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/tensor_shape_patch.py:13-26",
      "class": "design_defect",
      "severity": "low",
      "evidence": "apply_patches() in the Stage A input-embedding tree monkey-patches global behavior by importing apply_tensor_fixes from rna_predict.pipeline.stageD.diffusion.run_stageD_unified and invoking it (lines 21-24). A Stage A utility reaching into Stage D to mutate runtime functions creates a hidden cross-stage coupling and ordering dependency (patches must be applied before pipeline run); if Stage D's run_stageD_unified is unavailable/changes, importing/calling this raises at patch time. It also prints success unconditionally (line 26).",
      "recommendation": "Localize shape fixes to the modules that own them and apply them at import/init of those modules rather than via cross-stage monkey-patching; if patching is required, make it explicit, idempotent, and logged rather than printed.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-022",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer.py:9-12",
      "class": "design_defect",
      "severity": "low",
      "evidence": "A module transformer.py and a package transformer/ (transformer/__init__.py) coexist in the same directory; the package shadows the module on import, so transformer.py is unreachable dead code. It also diverges from the live package: transformer.py imports AtomAttentionEncoder from transformer.atom_attention (the subpackage) and does NOT export AtomAttentionConfig, whereas the live transformer/__init__.py imports from atom_attention_encoder.py and exports AtomAttentionConfig (which embedders.py:24-28 relies on). Maintaining the shadowed copy risks confusion and edits that never take effect. The same module-vs-package shadowing exists for primitives.py vs primitives/ and atom_attention.py vs atom_attention/.",
      "recommendation": "Remove the shadowed transformer.py (and similarly primitives.py, transformer/atom_attention.py) or rename to avoid the package/module name collisions so the active import target is unambiguous.",
      "status": "survived"
    },
    {
      "id": "s2c2l2-atomattention-print-in-init",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:112",
      "class": "perf",
      "severity": "low",
      "evidence": "AtomAttentionEncoder.__init__ executes `print(f\"[DEBUG][AtomAttentionEncoder] Propagating c_ref_element={self.c_ref_element}\")` unconditionally (line 112), and _setup_feature_dimensions has another print gated only loosely (line 146). Unconditional debug printing on construction.",
      "recommendation": "Use logger.debug guarded by debug_logging, or remove the print.",
      "status": "survived"
    },
    {
      "id": "s2c2l0-021",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention.py:300-316,483-499",
      "class": "bug",
      "severity": "low",
      "evidence": "AtomAttentionEncoder defines the method _process_input_features twice (lines 300-316 and again at 483-499). The second definition silently overrides the first; although their bodies are currently identical, this duplicate-definition is a maintenance hazard (an edit to the first is dead) and indicates a botched refactor/merge.",
      "recommendation": "Delete one of the duplicate _process_input_features definitions.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-003",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:151",
      "class": "bug",
      "severity": "low",
      "evidence": "create_pair_embedding wraps only ref_charge access in try/except ValueError (lines 146-155) and has multiple `if ref_pos is None` guards (lines 151,158,169). But safe_tensor_access (common.py:61-66) RAISES ValueError when a key is missing/non-tensor and never returns None when default is None. So ref_pos (line 143, no default) raises on absence rather than returning None, making the None-guards dead/unreachable defensive code that masks intent.",
      "recommendation": "Either pass an explicit default to safe_tensor_access or remove the unreachable None branches; rely on a single clear validation path.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-002",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:157-161",
      "class": "bug",
      "severity": "low",
      "evidence": "In create_pair_embedding, lines 157-161 compute `if ref_pos is not None: ref_pos.shape[0]` (value discarded) with an empty `else: pass`. The 'number of atoms' is never bound to anything. Dead no-op left from a refactor.",
      "recommendation": "Delete the dead block or assign the value if it was intended to be used.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-004",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/atom_attention_feature_processing.py:52",
      "class": "perf",
      "severity": "low",
      "evidence": "Unconditional print at __init__ (line 52: '[DEBUG][FeatureProcessor] ref_element expected dim') and at extract_atom_features (line 112) fire on every construction/forward regardless of debug_logging (which is explicitly 'ignored in this implementation', line 42). Stage-1 intent marks inference as the primary deliverable, so this is stdout spam on every call.",
      "recommendation": "Gate behind the debug_logging flag or use logger.debug; honor the documented debug_logging parameter.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-028",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/attention_components.py:133",
      "class": "bug",
      "severity": "low",
      "evidence": "process_pair_features hard-casts the pair tensor to float32 unconditionally: p_ij = p_ij.to(dtype=torch.float32) (line 133). Under autocast/AMP or half-precision training/inference this forces the pair branch back to fp32, breaking dtype consistency with the rest of the model and potentially causing dtype-mismatch errors in subsequent ops.",
      "recommendation": "Drop the unconditional float32 cast (let dtype follow the inputs) or cast to the module/input dtype rather than a fixed float32.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-004",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/components/attention_components.py:215",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Trailing `# TODO: Refactor this file to improve code quality score - needs work on complexity and argument count` is an unresolved development marker left in shipped source.",
      "recommendation": "Resolve or remove the TODO; track refactors in an issue tracker rather than inline.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-027",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/config.py:31",
      "class": "design_defect",
      "severity": "low",
      "evidence": "AtomAttentionConfig.__post_init__ validates only c_atom, c_token and n_blocks for positivity (lines 33-38), leaving c_atompair, c_s, c_z, n_heads, n_queries, n_keys unchecked. A zero/negative n_heads or c_atompair would pass config validation and only fail deep inside attention with an opaque error (DiffusionTransformer does validate n_heads at diffusion.py:175, but the atom-attention config layer does not).",
      "recommendation": "Validate all dimension/head/window fields for positivity in __post_init__ for a clear early failure.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-043",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention/encoder.py:228",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "AtomAttentionEncoder.from_args accepts a debug_logging parameter (line 228) and the trailing comment (line 247) says 'If you want to use debug_logging, pass it separately here', but debug_logging is never forwarded to AtomAttentionConfig or cls — the argument is silently ignored. Similarly atom_attention_feature_processing.FeatureProcessor accepts debug_logging then documents it ignored.",
      "recommendation": "Either thread debug_logging through to the config/instance or drop the parameter and update the comment.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-019",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_decoder.py:131",
      "class": "bug",
      "severity": "low",
      "evidence": "AtomAttentionDecoder.forward mutates its input dataclass params in place: params.extra_feats is reassigned (lines 131,135), params.atom_mask is reassigned (line 144). Since DecoderForwardParams may be reused by the caller (e.g. across diffusion sample iterations), these hidden mutations can leak padded/truncated tensors into subsequent calls.",
      "recommendation": "Work on local copies (extra_feats = params.extra_feats; ... ) instead of writing back into the params object.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-012",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:104,170",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Leftover dead artefacts: line 104 `###@snoop` is a commented-out debugging decorator, and _forward_legacy_disabled (line 170) is a disabled/unused method retained in shipped source.",
      "recommendation": "Delete the commented decorator and the disabled legacy method.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-011",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/atom_attention_encoder.py:260-271",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "extract_atom_features wraps a call in try/except TypeError commenting 'First try with debug_logging parameter' (line 261), but the call `canonical_extract_atom_features(self, input_feature_dict)` is IDENTICAL in both the try (line 262) and the except (line 268) — neither passes debug_logging. The except branch's stated condition can never be triggered by the try, making the handler and its comment misleading dead code.",
      "recommendation": "Remove the try/except (or actually pass debug_logging in the try) so the code matches its comment.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-016",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/attention.py:525",
      "class": "bug",
      "severity": "low",
      "evidence": "`min(len(s.shape[:-1]), len(a.shape[:-1]))` is computed and discarded (no assignment) inside _apply_gating's adaptation branch. Dead no-op statement.",
      "recommendation": "Remove the dead expression or use its result if intended.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-014",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:315",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Two near-duplicate processing functions exist with divergent behavior: _process_inputs_with_coords_impl (line 315) passes params.s directly to the transformer and skips it when None (line 348), while process_inputs_with_coords (line 381) fabricates a zero style tensor fallback (lines 446-467) and returns p_for_transformer instead of torch.zeros_like(a) as the 4th element. Only process_inputs_with_coords is wired into forward; _process_inputs_with_coords_impl is unreferenced divergent logic that will rot.",
      "recommendation": "Delete the unused _process_inputs_with_coords_impl or fold it into the live function to avoid two contradictory implementations.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-023",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:331",
      "class": "design_defect",
      "severity": "low",
      "evidence": "When atom_to_token_idx cannot supply a token count, num_tokens silently defaults to the magic literal 50 (line 331), also used to size default_restype (line 332). An arbitrary hard-coded token count will produce silently wrong aggregation sizes.",
      "recommendation": "Derive num_tokens from real inputs or raise; avoid the magic 50 fallback.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-013",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/forward_logic.py:392",
      "class": "perf",
      "severity": "low",
      "evidence": "process_inputs_with_coords (the path called by AtomAttentionEncoder.forward, atom_attention_encoder.py:167) begins with two unconditional print() statements (lines 392-393) that fire on every forward, independent of the config-driven debug flag used everywhere else in the function.",
      "recommendation": "Convert the two prints to logger.debug guarded by the existing `debug` flag.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-026",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/initialization.py:48",
      "class": "design_defect",
      "severity": "low",
      "evidence": "setup_distance_encoders creates encoder.linear_no_bias_invd with in_features=1 (line 48), whereas the parallel FeatureProcessor builds linear_no_bias_invd with in_features=3 (atom_attention_feature_processing.py:63). Moreover the refactored pair path (pair_embedding.py) only uses linear_no_bias_d and linear_no_bias_v — linear_no_bias_invd is constructed but never used here, so the inconsistent inverse-distance encoder is dead in this path.",
      "recommendation": "Remove the unused linear_no_bias_invd from this path or reconcile its in_features with the other implementation if it is meant to be used.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-029",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/transformer/encoder_components/initialization.py:48",
      "class": "bug",
      "severity": "low",
      "evidence": "setup_distance_encoders creates encoder.linear_no_bias_invd with in_features=1 (initialization.py:48), but the components-based FeatureProcessor sets linear_no_bias_invd in_features=3 (atom_attention_feature_processing.py:63). Moreover create_pair_embedding (pair_embedding.py:134-196) never uses linear_no_bias_invd at all — the inverse-distance encoder is dead in the refactored encoder, and the in_features value disagrees between the two encoder implementations of the same intended layer.",
      "recommendation": "Either wire linear_no_bias_invd into the pair embedding (and fix its in_features to match the inverse-distance vector dim) or remove the unused layer; reconcile the two implementations.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-017",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/utils/coordinate_utils.py:93",
      "class": "bug",
      "severity": "low",
      "evidence": "_validate_and_clamp_indices calls atom_to_token_idx_flat.max() (line 93) without checking numel(); for an empty atom_to_token_idx (zero atoms) .max() raises 'max(): Expected reduction dim ...'. The encoder fallbacks can produce empty mappings, so this can crash on degenerate inputs.",
      "recommendation": "Guard with `if atom_to_token_idx_flat.numel() and atom_to_token_idx_flat.max() >= n_token:` before clamping.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-030",
      "location": "rna_predict/pipeline/stageA/input_embedding/current/utils/tensor_ops.py:66",
      "class": "design_defect",
      "severity": "low",
      "evidence": "one_hot computes `dgram = (x[...,None] > lower_bins) * (x[...,None] < upper_bins).float()`. Due to operator precedence `.float()` applies only to the second comparison; the first factor stays bool. It works numerically (bool*float promotes), but the placement is misleading and fragile, and the returned tensor is not gated to a clean {0,1} one-hot semantics the docstring implies.",
      "recommendation": "Parenthesize and cast explicitly: `((x>lower) & (x<upper)).float()`.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-032",
      "location": "rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py:60-67",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Atom and pair input dimensions are hard-coded with TODOs: `in_atom_dim = 3 + 1 + 128 + 16` (line 62) and `in_pair_dim = 3 + 1` (line 67), each tagged '# TODO: Define these input dimensions more formally'. Magic feature widths baked into the layer construction make the legacy encoder brittle to feature changes.",
      "recommendation": "Source these dimensions from the shared feature config (conf/shared/features.yaml) or named constants; resolve the TODOs.",
      "status": "survived"
    },
    {
      "id": "s2c3l1-003",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:184-191",
      "class": "security",
      "severity": "low",
      "evidence": "The download URL (stage_cfg.checkpoint_url, run_stageA.py:188), the local zip path (stage_cfg.checkpoint_zip_path, run_stageA.py:189) and the extraction root (derived from stage_cfg.checkpoint_path, run_stageA.py:184-191) are taken directly from Hydra config with no scheme allow-listing or path containment. An attacker able to influence the composed config (override files / command-line overrides) can cause the process to fetch an arbitrary URL (urlopen accepts file://, http://, etc. -> SSRF / local file read) and write the response to an arbitrary local path. This is a privilege/trust-boundary concern on top of findings 001/002.",
      "recommendation": "Restrict checkpoint_url to an https allow-list (reject file://, ftp://, internal IP literals), and confine checkpoint_zip_path / extraction directories to a known cache root via realpath containment checks.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-030",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:191",
      "class": "bug",
      "severity": "low",
      "evidence": "main() extracts the checkpoint with unzip_file(checkpoint_zip, os.path.dirname(checkpoint_dir), ...) (line 191) where checkpoint_dir = os.path.dirname(stage_cfg.checkpoint_path) (line 184). Extracting to dirname(dirname(checkpoint_path)) places files two directory levels above the checkpoint file; if the zip does not itself contain the expected sub-directory layout, the checkpoint will not land at stage_cfg.checkpoint_path and predictor instantiation (line 210) may fall back to dummy mode. This depends on the zip's internal structure (unverified), so flagged as a latent path-assembly risk.",
      "recommendation": "Verify the zip layout and extract to the directory that yields checkpoint_path exactly; add a post-extract assert os.path.isfile(stage_cfg.checkpoint_path).",
      "status": "survived"
    },
    {
      "id": "s2c3l2-033",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:198-199",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Stage A main emits unconditional `logger.info(\"[HYDRA-DEBUG][StageA] ...\")` (lines 198-199) while every other diagnostic in the function is gated by `debug_logging`. Line 199 labels the value 'Global cfg.device' but actually reads cfg.model.stageA.device — the same nested value printed on line 198, so the label is inaccurate.",
      "recommendation": "Gate these behind debug_logging and fix the misleading 'Global cfg.device' label (or read the actual global device key).",
      "status": "survived"
    },
    {
      "id": "s2re5-download-helper-duplication",
      "location": "rna_predict/pipeline/stageA/run_stageA.py:60-93 vs rna_predict/training/rna_lightning_module.py:108-163",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The exponential-backoff download helper (existing-zip validation, urlopen+shutil.copyfileobj, max_retries/backoff loop, identical log message strings like '[DL] Download attempt {attempt+1}/{max_retries} failed') is duplicated near-verbatim between run_stageA.py and rna_lightning_module.py instead of sharing a single utility. Divergent maintenance risk: a fix to one (e.g. adding weights_only or checksum validation) will silently miss the other.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c3l0-025",
      "location": "rna_predict/pipeline/stageB/main.py:223",
      "class": "bug",
      "severity": "low",
      "evidence": "Line 223 `protenix_cfg.c_token if hasattr(protenix_cfg, 'c_token') else 2` is a bare expression statement whose value is discarded — the intended local (e.g. c_token) is never assigned, so the configured c_token (or default 2) is silently dropped. (Same dead-expression pattern appears at atom_attention/components/atom_attention_feature_processing.py:159 `ref_pos.shape[0]`.)",
      "recommendation": "Assign the result to the intended variable (c_token = ...) and use it, or remove the dead statement.",
      "status": "survived"
    },
    {
      "id": "s2c3l2-039",
      "location": "rna_predict/pipeline/stageB/main.py:298-332",
      "class": "design_defect",
      "severity": "low",
      "evidence": "run_pipeline validates the RNA sequence twice with equivalent logic: once at lines 298-307 (ACGU membership, raising ERR-STAGEB-RUNPIPELINE-003) and again at lines 324-332 after the unreachable empty-handling block. The second pass is redundant.",
      "recommendation": "Keep a single validation pass.",
      "status": "survived"
    },
    {
      "id": "s2c3l0-024",
      "location": "rna_predict/pipeline/stageB/main.py:366",
      "class": "perf",
      "severity": "low",
      "evidence": "run_pipeline emits unconditional print() statements on the hot path: '[CASCADE-DEBUG] BEFORE STAGE B' (line 366), 'AFTER STAGE B' (line 368), 'BEFORE STAGE C' (line 372), independent of debug_logging. These print the full sequence each call and bypass the logger configured elsewhere in the file.",
      "recommendation": "Convert to logger.debug guarded by debug_logging, consistent with the rest of the module.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-028",
      "location": "rna_predict/pipeline/stageB/pairwise/main.py:99-108",
      "class": "other",
      "severity": "low",
      "evidence": "main()/demo emits unconditional print() debug dumps of the pairformer config and selected keys (:99 'print(\"[DEBUG][stageB] pairformer config:\", pf_cfg)' and the loops at :100-108) regardless of the debug_logging flag resolved at :93-95, contradicting the file's own debug-gating convention.",
      "recommendation": "Gate these prints behind debug_logging or convert to logger.debug.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-002",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer.py:872",
      "class": "other",
      "severity": "low",
      "evidence": "MSAModule.forward contains an unconditional `print(f\"[DEBUG][MSAModule] msa_sample.shape before linear: {msa_sample.shape}\")` that is not gated by any debug flag, unlike the logger.* calls elsewhere in the module. This pollutes stdout during normal inference.",
      "recommendation": "Replace with a debug-gated logger.debug call or remove.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-debug-prints-hot-path",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer.py:872",
      "class": "perf",
      "severity": "low",
      "evidence": "MSAModule.forward unconditionally executes `print(f\"[DEBUG][MSAModule] msa_sample.shape before linear: {msa_sample.shape}\")` on every forward pass (not gated by debug_logging). Similar always-on debug prints exist in pairformer_wrapper.predict (lines 413-414) and DummyTorsionBertAutoModel.forward (torsionbert_inference.py:18-19,43,63,124, including traceback.print_stack). These pollute stdout and add overhead in inference loops.",
      "recommendation": "Gate these prints behind a debug flag / logger.debug and remove traceback.print_stack from the model forward.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-007",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer_utils.py:38-78",
      "class": "design_defect",
      "severity": "low",
      "evidence": "pairformer_utils defines sample_msa_feature_dict_random_without_replacement(feature_dict, n_samples) but pairformer.py:66,846 imports a different function of the same name from stageA.input_embedding.current.utils and calls it with keyword sample_size=. Two same-named MSA-sampling helpers with different signatures coexist; the one defined here is not the one used by MSAModule, creating a duplicate-name hazard.",
      "recommendation": "Rename or remove the unused duplicate, or consolidate to a single MSA-sampling helper.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-pairformer-wrapper-predict-random",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:397-415",
      "class": "design_defect",
      "severity": "low",
      "evidence": "PairformerWrapper.predict() returns torch.randn dummy single/pair embeddings (s_emb, z_emb) rather than running the PairformerStack; comment says 'For now, return dummy tensors'. Any caller using predict() (as opposed to forward()) silently receives random, non-deterministic outputs.",
      "recommendation": "Either implement real prediction via the stack/forward path or raise NotImplementedError so callers cannot mistake random tensors for real embeddings.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-005",
      "location": "rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py:71-77",
      "class": "other",
      "severity": "low",
      "evidence": "PairformerWrapper.__init__ emits several logger.info calls unconditionally (memory usage, '[DEBUG-PROPAGATION]...' lines including the full config object at :77) regardless of self.debug_logging. The surrounding comment says 'only gate debug' but the dumped full config and debug-propagation lines are info-level and always printed, contradicting the debug_logging gating intent used elsewhere.",
      "recommendation": "Gate the [DEBUG-PROPAGATION] and full-config logs behind self.debug_logging.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-006",
      "location": "rna_predict/pipeline/stageB/pairwise/protenix_integration.py:11-17",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Module docstring lists configuration requirements including 'restype_dim', 'profile_dim', and 'use_optimized' under model.stageB.pairformer.protenix_integration, but __init__ only validates/reads device, c_token, c_atom, c_pair, r_max, s_max (:67). restype_dim/profile_dim/use_optimized are never read in this file (the demo in pairwise/main.py:55-56 hardcodes restype_dim/profile_dim=32 locally instead).",
      "recommendation": "Update the docstring to match the parameters actually consumed, or wire the listed parameters into the embedder.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-008",
      "location": "rna_predict/pipeline/stageB/pairwise/triangular_multiplicative.py:228",
      "class": "other",
      "severity": "low",
      "evidence": "Inside compute_projection's chunked branch, line 228 `mask[..., i : i + inplace_chunk_size, :, :]` evaluates a slice and discards it (no assignment, no side effect); the actual mask slice used is recomputed at :231. This is a dead statement (likely a leftover from a refactor).",
      "recommendation": "Remove the dead slice expression at line 228.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-trimul-dead-mask-slice",
      "location": "rna_predict/pipeline/stageB/pairwise/triangular_multiplicative.py:228",
      "class": "bug",
      "severity": "low",
      "evidence": "Inside compute_projection's chunked branch, the statement `mask[..., i : i + inplace_chunk_size, :, :]` is a bare expression whose result is discarded; the correctly-sliced mask is re-computed and passed at line 231. The line is dead code (likely a leftover from intended `mask_chunk = ...`).",
      "recommendation": "Remove the no-op line (or assign it to a mask_chunk variable if a slice was intended).",
      "status": "survived"
    },
    {
      "id": "s2c4l2-015",
      "location": "rna_predict/pipeline/stageB/torsion/lora_param_count.py:22-27",
      "class": "other",
      "severity": "low",
      "evidence": "Module-level code instantiates StageBTorsionBertPredictor and prints parameter counts at import time with no `if __name__ == '__main__'` guard. Importing this module triggers model loading and stdout output as a side effect.",
      "recommendation": "Wrap the execution body in a main() function guarded by __main__.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-011",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:402-409",
      "class": "other",
      "severity": "low",
      "evidence": "self.output_dim = hidden_size is assigned as a 'Placeholder' (:402) and then unconditionally overwritten by the following if/elif/else (:403-409). The first assignment is dead.",
      "recommendation": "Remove the placeholder assignment at :402.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-torsionbert-dead-training-stmt",
      "location": "rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py:487",
      "class": "bug",
      "severity": "low",
      "evidence": "`self.model.training if hasattr(self.model, 'training') else None` is a bare expression that evaluates and discards the model's training flag with no assignment or side effect; it does nothing.",
      "recommendation": "Remove the statement or assign/use the value if a prior-mode save/restore was intended.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-013",
      "location": "rna_predict/pipeline/stageB/torsion/torsionbert_inference.py:16-28",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "In DummyTorsionBertAutoModel.__init__, executable statements (import os/traceback, print, the num_angles==7 check) appear at :16-22 BEFORE the triple-quoted block at :23-28. Because code precedes it, that block is not the function docstring (it is a discarded string literal); the intended __init__ documentation is therefore not attached to the function.",
      "recommendation": "Move the docstring to the first statement of __init__ (immediately after the def line).",
      "status": "survived"
    },
    {
      "id": "s2c4l0-mpnerf-unqualified-squeeze",
      "location": "rna_predict/pipeline/stageC/mp_nerf/massive_pnerf.py:184",
      "class": "bug",
      "severity": "low",
      "evidence": "result = c + bond_length.unsqueeze(-1) * torch.matmul(rotate, d).squeeze(). The unqualified .squeeze() removes ALL size-1 dimensions; for a batch of size 1 (e.g. matmul output shape (1,3,1)) it collapses both the batch dim and the trailing dim to shape (3,), which can mis-broadcast against c of shape (1,3) and produce silently wrong/ambiguous shapes.",
      "recommendation": "Use squeeze(-1) to drop only the matmul column dimension, preserving batch dimensions.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-017",
      "location": "rna_predict/pipeline/stageC/mp_nerf/ml_utils.py:5-19",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "ml_utils.py docstring states 'This file is maintained for backward compatibility... Re-export all functions for backward compatibility' (:5-15) but the file body contains only comments and a stray reference URL (:11-19); it re-exports nothing. Stage-1 inventory describes it as 'Backward-compatibility shim that re-exports all symbols from the ml_utils subpackage', which the code does not do.",
      "recommendation": "Either implement the documented re-exports or remove the file (see s2c4l2-016).",
      "status": "survived"
    },
    {
      "id": "s2c4l2-027",
      "location": "rna_predict/pipeline/stageC/mp_nerf/ml_utils/main.py:181-208",
      "class": "other",
      "severity": "low",
      "evidence": "_run_main_logic (invoked from ml_utils/__init__.py:24-25 under __main__) is a protein-only demo: it loads from a hardcoded non-existent path data_path='some_route_to_local_serialized_file_with_prots' (:187), expects a 7-tuple protein record, and rearranges with the SidechainNet c=14 layout (_test_noise_internals :120). It cannot run as written and is unrelated to RNA reconstruction.",
      "recommendation": "Remove the dead demo or parameterize the data path and label it clearly as a protein-MP-NeRF example.",
      "status": "survived"
    },
    {
      "id": "s2c4l1-003",
      "location": "rna_predict/pipeline/stageC/mp_nerf/ml_utils/main.py:29-33",
      "class": "security",
      "severity": "low",
      "evidence": "_load_protein_data() does `import joblib; prots = joblib.load(data_path)` (lines 29-32). joblib.load deserializes via pickle, which executes arbitrary code embedded in a crafted/poisoned data file during unpickling. The downstream _validate_protein_data() checks (line 43+) run only AFTER joblib.load has already executed, so they provide no protection against malicious payloads. In the shipped caller _run_main_logic() the path is a hardcoded placeholder ('some_route_to_local_serialized_file_with_prots', line 187), limiting current exposure, but _load_protein_data accepts an arbitrary data_path argument, so any caller passing an untrusted/serialized file path gets pickle-deserialization RCE.",
      "recommendation": "Do not unpickle untrusted data. Store/load the protein fixtures in a non-executable format (e.g. .npz/np.load with allow_pickle=False, or safetensors), or restrict joblib.load to files within a trusted, integrity-checked directory. Document that this module is a developer-only test harness and never feed it externally supplied files.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-029",
      "location": "rna_predict/pipeline/stageC/mp_nerf/protein_utils/mask_generators.py:115-149",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "scn_angle_mask is self-flagged as '(Potentially legacy or needs update based on SUPREME_INFO usage)' (:117) and returns a (L,12) array with phi/psi/omega left as NaN placeholders and the 6 sidechain angle slots never filled (only mask[i,3:6] bond angles set, :139-144; comments at :146-148 describe unimplemented SUPREME_INFO retrieval). It diverges from proteins.scn_angle_mask (proteins.py:64-133), which fully populates angles from SUPREME_INFO. Same-named functions return different/incomplete data.",
      "recommendation": "Complete or remove the placeholder scn_angle_mask and unify with the proteins.py implementation to avoid silently producing NaN/zero angle masks.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-symmetry-utils-unverified-indices",
      "location": "rna_predict/pipeline/stageC/mp_nerf/protein_utils/symmetry_utils.py:29-53",
      "class": "design_defect",
      "severity": "low",
      "evidence": "get_symmetric_atom_pairs hardcodes SidechainNet atom-index pairs per residue with the authors' own repeated 'check indices' comments and a TODO stating the mapping may be wrong (e.g. F/Y use (6,10),(7,9); H uses (6,9),(7,8)). If used by rename_symmetric_atoms (ml_utils/atom_utils.py) these unverified indices would swap the wrong atoms. (Protein-path utility; relevance to the RNA pipeline is unverified.)",
      "recommendation": "Verify the indices against SC_BUILD_INFO atom-name ordering and derive pairs from atom names instead of hardcoded integers.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-026",
      "location": "rna_predict/pipeline/stageC/mp_nerf/protein_utils/symmetry_utils.py:29-54",
      "class": "design_defect",
      "severity": "low",
      "evidence": "get_symmetric_atom_pairs uses hardcoded SidechainNet atom indices for symmetric pairs with explicit uncertainty markers: TODO 'these indices ... seem hardcoded ... Verify if this mapping is robust' (:29-31) and repeated '- check indices' comments on F/Y/R/H/V/L (:37-47), plus a note that prior (4,5) pairs were removed as 'seems incorrect' (:50-51). Correctness of the symmetric-atom renaming is unverified. This is also a second divergent implementation versus ml_utils/atom_utils.py:514 get_symmetric_atom_pairs (which uses AMBIGUOUS).",
      "recommendation": "Verify the index mapping against the SidechainNet atom ordering (or derive indices from atom names dynamically) and unify with the atom_utils implementation.",
      "status": "survived"
    },
    {
      "id": "s2c4l0-rna-nan-check-preassign",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_base_placement.py:336-339",
      "class": "bug",
      "severity": "low",
      "evidence": "The 'universal NaN check after any atom placement' tests torch.isnan(full_coords[i, idx, :]) at line 336, but full_coords[i, idx] still holds its zero-initialized value because the computed `pos` is only written into full_coords at line 339 (after the check). The guard therefore always inspects zeros and never detects a NaN produced in `pos`. (A final nan_to_num at line 349 mitigates downstream impact.)",
      "recommendation": "Check torch.isnan(pos).any() before assigning, or move the NaN check after the full_coords[i, idx, :] = pos assignment.",
      "status": "survived"
    },
    {
      "id": "s2c4l2-019",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_constants.py:30-36",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The inline atom-quadruplet comments for each torsion are shifted by one relative to standard RNA backbone definitions: alpha is annotated 'P-O5'-C5'-C4'' (the beta atoms), beta as 'O5'-C5'-C4'-C3'' (gamma's), gamma as 'C5'-C4'-C3'-O3'' (delta's), delta as 'C4'-C3'-O3'-P' (epsilon's), epsilon as 'C3'-O3'-P-O5'' (zeta's), zeta as 'O3'-P-O5'-C5'' (alpha's). final_kb_rna.py:185-191 carries the correct labels. The numeric values appear correct; only the comments are mislabeled.",
      "recommendation": "Correct the atom-quadruplet comments to match the canonical alpha..zeta definitions (as in final_kb_rna.py).",
      "status": "survived"
    },
    {
      "id": "s2c5l2-001",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:17",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Module-level `print(f\"!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED FROM: {__file__} !!!!!!!!!!\")` executes on every import. Further unconditional `print` debug banners at line 180 (build_rna_chain_from_internal_coords completion) and line 247 (ring_closure_refinement). This is leftover 'CASCADE' debug instrumentation that pollutes stdout of the README-documented Stage C reconstruction path regardless of any debug flag.",
      "recommendation": "Remove the module-level and function-level print() banners or gate them behind the existing debug_logging flag / logger.debug.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-003",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:17,180,247",
      "class": "other",
      "severity": "low",
      "evidence": "Unconditional module-level `print(\"!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED ...\")` at import (line 17), a per-call `print(\"!!! CASCADE: build_rna_chain_from_internal_coords COMPLETED ...\")` (line 180), and a CASCADE print inside ring_closure_refinement (line 247). These execute on every import/call regardless of any debug flag, polluting stdout in production/inference runs.",
      "recommendation": "Remove the stray debug prints or gate them behind a debug_logging flag / logger.debug.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-002",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:177-181",
      "class": "bug",
      "severity": "low",
      "evidence": "After building coordinates, `if torch.isnan(residue_coords).any(): logger.error(...)` logs an error but does NOT raise; the function then returns the NaN-containing tensor (line 181). Callers (stage_c_reconstruction.run_stageC_rna_mpnerf) receive NaN coordinates silently, which then flow into place_bases and downstream Stage D.",
      "recommendation": "Either raise on NaN detection or return an explicit validity flag so callers do not silently propagate NaN coordinates.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-005",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_scaffolding.py:56-93",
      "class": "design_defect",
      "severity": "low",
      "evidence": "backbone_triplets are built with `range(len(RNA_CONNECT['backbone']) - 2)` (line 56), dropping the final triplet, and dihedral torsions are written to fixed hard-coded slots angles_mask[1,i,1..6] and angles_mask[1,i,9] (lines 87-93), skipping indices 7 and 8 with no documented rationale. This fixed slot mapping is fragile and unrelated to the get_torsion_angle_index mapping used in rna_folding.py:130, so the two modules encode torsion ordering differently.",
      "recommendation": "Document and unify the torsion-index mapping between scaffolding and folding; replace magic indices with named constants and verify the -2 triplet bound is intentional.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-004",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_scaffolding.py:95-112",
      "class": "design_defect",
      "severity": "low",
      "evidence": "point_ref_mask is filled for non-P atoms as `[i*B + (j-3), i*B + (j-2), i*B + (j-1)]` (lines 110-112). For j==1 and j==2 this yields negative within-residue offsets (j-3 = -2 and -1; j-2 = -1 and 0), producing reference indices that point into the previous residue's atoms or to negative positions. The masks (bond_mask, point_ref_mask) are also not consumed by the actual folding path (rna_folding.build_rna_chain_from_internal_coords reads only scaffolds['torsions']), so this construction is both incorrect and dead for the mp_nerf path.",
      "recommendation": "Either remove the unused bond_mask/point_ref_mask construction or fix the j<3 reference indices (clamp/guard the negative offsets) if a future code path will consume them.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-008",
      "location": "rna_predict/pipeline/stageC/mp_nerf/utils.py:10-31",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Bio.PDB is imported twice: first unconditionally via importlib.util.find_spec guard (lines 11-15) and again inside a try/except that defines the BIOPYTHON_AVAILABLE flag and dummy classes (lines 23-31). The first import is redundant and the commented-out import block (lines 17-20, 26) is leftover noise.",
      "recommendation": "Keep only the try/except import that sets BIOPYTHON_AVAILABLE; remove the redundant find_spec import and commented blocks.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-006",
      "location": "rna_predict/pipeline/stageC/mp_nerf/utils.py:71-72",
      "class": "bug",
      "severity": "low",
      "evidence": "to_zero_two_pi returns `torch.where(x > np.pi, x % np.pi, 2*np.pi + x % np.pi)`. For x in [0, pi] (the not-greater branch) it returns 2*pi + (x % pi), i.e. values >= 2*pi, which cannot be a wrapped angle in [0, 2*pi). The intended wrap to [0, 2*pi) is not produced. (Vendored utility; verify usage before relying on it.)",
      "recommendation": "Correct the angle-wrap formula (e.g., `x % (2*np.pi)`), or remove if unused.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-007",
      "location": "rna_predict/pipeline/stageC/mp_nerf/utils.py:76-136",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "get_prot pulls a PROTEIN from sidechainnet (vocab int2char, padding token == 20, coords stride * 14, angles) — protein-specific code vendored from the original mp_nerf library (file header 'Author: Eric Alcaide') sitting in an RNA pipeline. It also has an unreachable `return None` at line 136 after a `while True:` loop. This is dead/irrelevant code relative to the RNA structure-prediction intent.",
      "recommendation": "Remove get_prot (and other protein-only helpers) or clearly quarantine/document them as unused vendored code; delete the unreachable return.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-005",
      "location": "rna_predict/pipeline/stageC/mp_nerf/utils.py:88-136",
      "class": "bug",
      "severity": "low",
      "evidence": "get_prot wraps its logic in `while True:` (line 88) with the only exit being a `return` inside the loop; the trailing `return None` at line 136 is unreachable. If the dataloader yields no matching protein the function spins forever rather than terminating. Vendored sidechainnet helper (protein-oriented) embedded in the RNA mp_nerf tree.",
      "recommendation": "Add a termination condition (e.g., StopIteration handling / max attempts) so the loop cannot spin indefinitely, and drop or make the trailing return reachable.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-011",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:53-63,105-110",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The class docstring states Stage C 'intentionally produces a sparse atom representation (21 atoms total)' and that 'Stage D expects a dense atom representation (44 atoms per residue)'. But __call__ returns coords of shape (N*3,3) i.e. 3 atoms/residue (not 21 total), and the real MP-NeRF path returns per-residue STANDARD_RNA_ATOMS counts. The '21 atoms total' and '44 per residue' figures are not reflected by either code path here (compute_max_rna_atoms returns 21 as a per-residue max, not a total), making the docstring misleading.",
      "recommendation": "Update the class docstring to reflect the actual per-residue atom counts produced by each method and the bridging target, or remove the stale numeric claims.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-012",
      "location": "rna_predict/pipeline/stageC/stage_c_reconstruction.py:72-76,256-258,403-408",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Several INFO-level logs are emitted unconditionally (independent of debug_logging): per-instance memory logs in __init__ (lines 72-76), '[DEBUG][StageC] stage_cfg.device ...' emitted via logger.info at lines 256-258 (debug content at INFO level), and a duplicate summary forced onto the ROOT logger at line 408 ('ROOT: StageC completed ...'). These 'SYSTEMATIC DEBUGGING' artifacts spam the documented inference path's logs.",
      "recommendation": "Demote these to logger.debug gated by debug_logging and remove the explicit root-logger emission.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-029",
      "location": "rna_predict/pipeline/stageD/config.py:108-162",
      "class": "design_defect",
      "severity": "low",
      "evidence": "stageD/config.py defines a dataclass `DiffusionConfig` (a Hydra schema with mode/device/model_architecture-style fields) and registers it in the ConfigStore at import (cs.store at lines 149-162), while a completely different `DiffusionConfig` dataclass (a runtime input container with partial_coords/trunk_embeddings) lives in diffusion/utils/config_types.py and is what diffusion/config.py and utils/__init__.py re-export. Two unrelated classes share the name DiffusionConfig, inviting import/type confusion. ConfigStore.store also runs as an import side effect.",
      "recommendation": "Rename one of the DiffusionConfig classes (e.g., DiffusionSchemaConfig vs DiffusionRunConfig) to disambiguate, and move ConfigStore registration into an explicit register function rather than executing at import time.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-014",
      "location": "rna_predict/pipeline/stageD/config.py:114",
      "class": "design_defect",
      "severity": "low",
      "evidence": "DiffusionConfig.debug_logging defaults to True. Stage D is documented as a research prototype (README per Stage-1 intent), and the codebase has extensive debug_logging-gated prints/logs; defaulting this True makes verbose diagnostic output the out-of-the-box behavior for every Stage D run.",
      "recommendation": "Default debug_logging to False to match the other stages and avoid shipping verbose diagnostics by default.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-015",
      "location": "rna_predict/pipeline/stageD/config.py:18,117",
      "class": "design_defect",
      "severity": "low",
      "evidence": "sigma_data is defined inconsistently: NoiseScheduleConfig.sigma_data=16.0 (line 18) vs DiffusionConfig.sigma_data=1.0 (line 117); generator.py defaults sigma_data=16.0 with comment 'in EDM, this is 1.0' (generator.py:37,88). Three different sigma_data sources with two different values create ambiguity about the actual data scale used by the EDM noise process.",
      "recommendation": "Consolidate sigma_data to a single config field consumed by both the noise scheduler and the diffusion module.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-022",
      "location": "rna_predict/pipeline/stageD/diffusion/bridging/residue_atom_bridge.py:504-522",
      "class": "design_defect",
      "severity": "low",
      "evidence": "_process_one_trunk_embedding, after failing to find feature_dimensions in config, falls back to hardcoded expected_dim values (s_trunk=384, s_inputs=449, sing=384) with only a warning (lines 507-515), then adjusts tensor feature dims to those constants. This contradicts the module/project's stated 'strictly config-driven, no hardcoded fallbacks' intent and can silently coerce tensors to the wrong dimension when config is misconfigured.",
      "recommendation": "Raise the ValueError for missing feature_dimensions in all cases (as already done in the else branch at line 517-522) instead of substituting hardcoded 384/449 values.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-041",
      "location": "rna_predict/pipeline/stageD/diffusion/bridging/sequence_utils.py:59",
      "class": "design_defect",
      "severity": "low",
      "evidence": "_try_extract_from_input_features unconditionally executes `print(f\"[CASCADE-DEBUG][SEQ-EXTRACT] type={type(result)}, value={result}\")` (line 59) on every successful sequence extraction, with no debug flag gating. This prints the full sequence to stdout during normal Stage D bridging.",
      "recommendation": "Remove the print or gate it behind a debug flag / logger.debug.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-025",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py:139,221,236,239,269,272",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Several print() calls in the conditioning forward path are not gated by debug_logging: e.g. '[WARNING] Unexpected batch size mismatch...' (line 139) and the '[DIFFUSION-FIX]' bridging prints (lines 221,236,239,269,272). These emit to stdout on normal shape-mismatch handling during every inference, mixing diagnostics into the output stream.",
      "recommendation": "Route through self.logger and gate behind self.debug_logging.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-015",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:124-132",
      "class": "bug",
      "severity": "low",
      "evidence": "In the `is_test and 'test_init_with_basic_config' in current_test` early-init branch, when 'transformer' is not in kwargs the code calls `self.logger.debug(...)` (line 128). self.logger is only assigned later at line 259, after this branch returns at line 132, so this path raises AttributeError: 'DiffusionModule' object has no attribute 'logger'. Latent crash in the test-special-case branch.",
      "recommendation": "Use the module-level logger (defined at line 84) instead of self.logger here, or set self.logger before this branch; better, remove the test-special-case branch entirely (see s2c5l0-014).",
      "status": "survived"
    },
    {
      "id": "s2c5l2-028",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:124-132,258-259",
      "class": "bug",
      "severity": "low",
      "evidence": "In the test_init_with_basic_config early-return branch, when 'transformer' is not in kwargs the code calls `self.logger.debug(...)` (line 128), but self.logger is not assigned until line 259 (after this branch returns at line 132). This raises AttributeError ('DiffusionModule' object has no attribute 'logger') on that path.",
      "recommendation": "Use the module-level logger (defined line 84) instead of self.logger before it is set, or set self.logger before any branch — and ideally remove the test branch per s2c5l2-027.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-029",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:70-81",
      "class": "design_defect",
      "severity": "low",
      "evidence": "DiffusionModule.__init__ unconditionally prints debug instrumentation on every instantiation: type(cfg), cfg.keys(), cfg.model_architecture, and kwargs (lines 70-81), independent of debug_logging. This spams stdout each time the diffusion model is built.",
      "recommendation": "Gate these prints behind debug_logging via the logger or remove them.",
      "status": "survived"
    },
    {
      "id": "s2c5l1-stageD-diffmodule-envvar-init-002",
      "location": "rna_predict/pipeline/stageD/diffusion/components/diffusion_module.py:99-132",
      "class": "design_defect",
      "severity": "low",
      "evidence": "DiffusionModule.__init__ inspects os.environ.get('PYTEST_CURRENT_TEST') (:100) and, when it contains 'test_init_with_basic_config' (:103), executes an alternate initialization path that sets only self.c_atom/self.c_z/self.transformer from kwargs and RETURNS EARLY at :132, skipping construction of the conditioning module, encoder, transformer and decoder. The same env-var-driven special-casing also appears in protenix_diffusion_manager.py:144-153,187-213 and diffusion_module forward()/_compute_loss() (inspect-based caller-name checks at :952-956,1023-1028). Behavior of a core model component is therefore determined by an ambient, externally settable environment variable rather than by configuration.",
      "recommendation": "Eliminate environment-variable and caller-frame (inspect) based branching from model construction and forward passes. Use explicit, config-driven flags so the instantiated module is deterministic and independent of the runtime environment.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-027",
      "location": "rna_predict/pipeline/stageD/diffusion/context_objects.py:214-228",
      "class": "bug",
      "severity": "low",
      "evidence": "EmbeddingContext.get_z_trunk reads the fallback pair dimension as `c_z_dim = self.stage_cfg['model_architecture']['c_z']` (line 224) via subscript, whereas the sibling get_s_inputs (lines 166-174) reads c_s_inputs through attribute/feature_dimensions paths with graceful fallbacks. If stage_cfg lacks a 'model_architecture' key (the manager passes cfg.model.stageD.diffusion as stage_cfg), this raises KeyError when the 'pair' embedding is missing, instead of a clean fallback or clear config error.",
      "recommendation": "Use the same robust config-extraction logic as get_s_inputs (attribute/feature_dimensions/dict paths) for c_z, and raise a descriptive error if it cannot be resolved.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-021",
      "location": "rna_predict/pipeline/stageD/diffusion/diffusion.py:1-38",
      "class": "design_defect",
      "severity": "low",
      "evidence": "diffusion.py contains only the Apache license header and comments stating that DiffusionConditioning, DiffusionModule, DiffusionSchedule and utility functions were 'moved to components/'. The file has no executable code — it is an empty stub left behind after refactoring.",
      "recommendation": "Delete the empty module (and update any references) rather than keeping a comment-only file.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-022",
      "location": "rna_predict/pipeline/stageD/diffusion/generator.py:162-178,359-375",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Docstrings describe these RNA diffusion routines in protein terms: sample_diffusion 'Generates denoised protein structure coordinates' (line 162) and sample_diffusion_training 'Performs diffusion-based training by adding noise to ground-truth coordinates' framed around protein structure. The file header is the ByteDance/Protenix protein generator; the protein wording is stale for the RNA pipeline.",
      "recommendation": "Update docstrings to reference RNA structures, and note provenance/adaptation from Protenix.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-026",
      "location": "rna_predict/pipeline/stageD/diffusion/inference/inference_mode.py:58-82,137-140",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "A comment states 'Hydra best practice: always use config-driven value, never fallback to hardcoded default' (line 58), but the code falls back to a hardcoded `test_residues_per_batch = 25` (lines 76,79) when not found in config. seq_len is set to this value (line 82) and used in a hard `assert coords.shape[1] == atom_count` (line 137) plus a warning comparison to seq_len (line 138).",
      "recommendation": "Either remove the contradictory comment or genuinely require the config value (raise if absent); avoid the magic 25 default in production inference.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-019",
      "location": "rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py:143-153,186-213",
      "class": "design_defect",
      "severity": "low",
      "evidence": "ProtenixDiffusionManager.__init__ contains two `if 'test_init_with_basic_config' in current_test:` blocks driven by PYTEST_CURRENT_TEST that mutate diffusion_args (copying c_atom/c_z/c_s/... from model_architecture) only in the test environment. Production initialization therefore takes a different argument-assembly path than tests.",
      "recommendation": "Always normalize diffusion_args (move the model_architecture flattening out of the test guard so production and tests share one path); remove the PYTEST_CURRENT_TEST branching.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-034",
      "location": "rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py:308",
      "class": "design_defect",
      "severity": "low",
      "evidence": "demo_run_diffusion hardcodes device = 'mps' (Apple Metal) and allocates all demo tensors on it (lines 308-318). On non-macOS / non-MPS machines this demo raises at tensor creation. The same hardcoded-developer-environment pattern is noted elsewhere in the repo (Stage-1: training/train.py absolute macOS path).",
      "recommendation": "Default the demo to cpu (or auto-detect) rather than hardcoding 'mps'.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-035",
      "location": "rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py:66-73",
      "class": "design_defect",
      "severity": "low",
      "evidence": "get_unified_cfg uses `with resources.path('rna_predict.conf', '') as cfg_path:` to obtain the Hydra config dir and passes it to hydra.initialize(config_path=str(cfg_path)). importlib.resources.path is deprecated and passing '' as the resource name plus an absolute path to Hydra's initialize (which expects a path relative to the caller) is brittle and likely to fail.",
      "recommendation": "Use initialize_config_dir with importlib.resources.files('rna_predict.conf') (or hydra's documented absolute-dir API).",
      "status": "survived"
    },
    {
      "id": "s2c5l2-037",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/config_types.py:29,34-36",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The runtime DiffusionConfig hardcodes feature defaults test_residues_per_batch=25, ref_element_size=128, ref_atom_name_chars_size=256, profile_size=32 as plain magic numbers. These duplicate (and can drift from) the Hydra schema's input_features sizes (config.py:97-99 ref_element=[128], ref_atom_name_chars=[256], profile=[32]) without any linkage.",
      "recommendation": "Source these from the Hydra config rather than hardcoding duplicates, or document them as fallbacks tied to the schema.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-040",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/config_utils.py:118-133",
      "class": "design_defect",
      "severity": "low",
      "evidence": "validate_stageD_config contains a 'Special case for tests' that, when 'model.stageD' is absent but a top-level 'stageD' key exists, rewrites the whole cfg in place via OmegaConf.update loops (lines 119-133) — mutating the caller's config to relocate stageD under model. Test-shaped configs are being silently transformed by a validation function, which is surprising and couples validation to test layouts.",
      "recommendation": "Validate without mutating the input config; have tests provide the canonical model.stageD structure.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-028",
      "location": "rna_predict/pipeline/stageD/diffusion/utils/tensor_utils.py:72-75",
      "class": "bug",
      "severity": "low",
      "evidence": "normalize_tensor_dimensions, when tensor.shape[0] != batch_size, silently truncates with `tensor = tensor[:batch_size]` after only a warning (lines 73-75). If the input batch dimension is genuinely larger (e.g., an N_sample/residue dim was misinterpreted as batch), this silently discards data rather than failing, and can hide upstream shape errors during bridging.",
      "recommendation": "Raise on an unexpected batch dimension mismatch (or expand when batch_size>shape[0]); do not silently slice rows away.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-045",
      "location": "rna_predict/pipeline/stageD/memory_optimization/memory_fix.py:33-40,101-107",
      "class": "design_defect",
      "severity": "low",
      "evidence": "preprocess_inputs silently truncates 3D coords/embeddings to max_seq_len=25 (lines 35-36, 52-57) with no warning for the 3D case (only the 2D path warns), so longer inputs lose residues without notice; and run_stageD_with_memory_fixes defaults device='cuda' (line 106), which fails on CPU-only hosts. Combined, the 'memory-efficient' wrapper can silently drop data and assume CUDA.",
      "recommendation": "Warn (or make configurable) on any truncation including 3D, and default device to 'cpu'/auto-detect.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-043",
      "location": "rna_predict/pipeline/stageD/memory_optimization/run_stageD_memory_efficient.py:26-53",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The example dummy data and config use dims s_inputs=449, s_trunk=384, pair=64, c_token=832, c_atom=128, num_steps=100 etc. (lines 27-52). These magic numbers are inconsistent with the registered structured schema (config.py: c_s=8/384, c_z=4/128, c_s_inputs=8/32, c_token=768) and with the 'conditioning'/'manager' keys this config invents (see s2c5l2-044). The example does not match the canonical Hydra config shape.",
      "recommendation": "Build the demo config from the registered Hydra schema (or align the magic dims with it) so it documents real usage.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-021",
      "location": "rna_predict/pipeline/stageD/memory_optimization/run_stageD_memory_efficient.py:69-73",
      "class": "bug",
      "severity": "low",
      "evidence": "In train mode `x_denoised, loss, sigma = result` unpacks the return of run_stageD_with_memory_fixes -> run_stageD_diffusion -> run_training_mode, which returns `(x_denoised, sigma, x_gt_augment)` (training_mode.py:107). So the second element is sigma (mislabeled 'loss') and the third is x_gt_augment (mislabeled 'sigma'). `loss.item()` / `sigma.item()` then operate on the wrong tensors (x_gt_augment is not scalar, so sigma.item() would raise under debug_logging).",
      "recommendation": "Align the unpacking with run_training_mode's actual (x_denoised, sigma, x_gt_augment) ordering and naming.",
      "status": "survived"
    },
    {
      "id": "s2re1-009",
      "location": "rna_predict/pipeline/stageD/memory_optimization/test_memory.py:90",
      "class": "bug",
      "severity": "low",
      "evidence": "get_config(config_path=\"/Users/tomriddle1/RNA_PREDICT/rna_predict/conf\") hardcodes the original author's absolute machine path, so this Stage-D memory test cannot resolve Hydra configs on any other machine or in CI. A distinct machine-specific-path site from the previously-listed compute_ground_truth_angles.py:53 and from s2re1-001 (train.py).",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c5l2-019",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:303-307,543-546",
      "class": "design_defect",
      "severity": "low",
      "evidence": "_run_stageD_impl returns context.result if set, otherwise falls back to returning context.diffusion_cfg (a config object) as the function's 'refined coordinates' result (lines 304-307). The caller _run_stageD_main_logic_context then logs 'Coordinates were NOT refined as a tensor (training mode or error)' (line 546). Returning a config object in the coordinate-result slot is a confusing contract that masks failures as non-tensor 'results'.",
      "recommendation": "Return None or raise on failure rather than returning the config object; make the return type explicit and consistent.",
      "status": "survived"
    },
    {
      "id": "s2c5l1-stageD-pytest-envvar-bypass-001",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:319-366",
      "class": "design_defect",
      "severity": "low",
      "evidence": "run_stageD() reads the runtime environment variable PYTEST_CURRENT_TEST (os.environ.get(\"PYTEST_CURRENT_TEST\",\"\") at :319-320) and, when it is set and contains one of the literal test names ('test_run_stageD_basic','test_run_stageD_with_debug_logging','test_gradient_flow_through_stageD') at :341, SHORT-CIRCUITS the entire diffusion pipeline and returns a fabricated dummy tensor {\"coordinates\": out} (:343-366) instead of executing _run_stageD_impl. Production control flow is thus gated on an attacker/operator-controllable environment variable: anyone able to set PYTEST_CURRENT_TEST to a string containing those substrings makes the refinement stage silently return non-physical placeholder coordinates rather than real output. This is test logic embedded in a production code path.",
      "recommendation": "Remove the PYTEST_CURRENT_TEST branch from production code. Drive test-specific behavior through dependency injection / explicit constructor flags or monkeypatching in the test suite, never via an ambient environment variable that is honored at deploy time.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-020",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:392",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Stage D's @hydra.main uses config_name=\"default.yaml\" (with the .yaml extension), whereas every other stage main uses config_name=\"default\" (e.g. stage_c_reconstruction.py:491). Inconsistent config_name spelling across the otherwise-parallel stage entry points.",
      "recommendation": "Use config_name=\"default\" (no extension) for consistency with the other Hydra mains.",
      "status": "survived"
    },
    {
      "id": "s2c5l0-011",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:80-98",
      "class": "design_defect",
      "severity": "low",
      "evidence": "set_stageD_logger_level sets the ROOT logger's level (`root_logger.setLevel(level)`, line 85) and mutates every handler on the root logger (lines 95-98) based on Stage D's debug_logging flag. This is invoked from run_stageD (line 330) and changes global logging verbosity for the whole process as a side effect of calling Stage D.",
      "recommendation": "Scope level changes to the Stage D package logger; do not mutate the root logger / global handlers from a stage runner.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-018",
      "location": "rna_predict/pipeline/stageD/run_stageD.py:9-26",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The module docstring's Configuration Requirements list flat keys under model.stageD such as 'ref_element_size', 'ref_atom_name_chars_size' and 'inference: num_steps', but the actual structured schema (config.py) puts these under nested groups (input_features.ref_element.size, inference.num_steps inside DiffusionConfig.inference). The documented config shape does not match the registered schema.",
      "recommendation": "Update the docstring to reflect the real nested Hydra structure (model.stageD.diffusion.* with input_features sub-group).",
      "status": "survived"
    },
    {
      "id": "s2c5l0-024",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/bridging_utils.py:107-122",
      "class": "perf",
      "severity": "low",
      "evidence": "check_and_bridge_embeddings bridges residue-level s_inputs to atom-level with a double Python loop `for b in range(batch_size): for atom_idx in range(n_atoms):` doing per-element assignment (lines 108-119). This is O(batch*n_atoms) Python iterations per call. It also silently leaves atoms with residue_idx >= s_inputs.shape[1] as zeros (guard at line 118) without warning, which can mask mapping errors.",
      "recommendation": "Replace the loop with a vectorized gather (s_inputs[b, atom_to_token_idx]) and emit a warning when any residue_idx is out of range rather than silently zero-filling.",
      "status": "survived"
    },
    {
      "id": "s2c5l2-046",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/bridging_utils.py:108-119",
      "class": "perf",
      "severity": "low",
      "evidence": "check_and_bridge_embeddings bridges s_inputs from residue- to atom-level with a doubly-nested Python loop over batch_size and n_atoms, calling .item() per atom (lines 108-119). For realistic atom counts this is an O(B*N_atom) per-element Python loop on the inference hot path, where a vectorized index_select via atom_to_token_idx would suffice.",
      "recommendation": "Replace the per-atom Python loop with a vectorized gather (e.g. s_inputs[b].index_select(0, atom_to_token_idx)).",
      "status": "survived"
    },
    {
      "id": "s2c6l2-002",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/config_utils.py:15-20",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "flatten_stageD_config_to_dict docstring states it flattens the config '...omitting tensor fields' but the body is a single OmegaConf.to_container(stage_cfg, resolve=True) call (:20) that omits nothing; the comment at :19 ('Exclude tensor fields') is never implemented.",
      "recommendation": "Either implement tensor-field exclusion or correct the docstring/comment to match the actual behavior.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-001",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/config_utils.py:8,36",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Two near-identically-named stageD config validators with divergent behavior coexist: public validate_and_extract_stageD_config (:8) only checks cfg.model.stageD, while private _validate_and_extract_stageD_config (:36) also accepts a top-level cfg.stageD fallback (_extract_stageD :45) and silently injects defaults via setattr (_validate_required_stageD_params :70-73). Callers picking one vs the other get different validation semantics.",
      "recommendation": "Consolidate into one validator; make the default-injection explicit/opt-in and fail loudly on truly-missing required config rather than mutating stage_cfg with hardcoded defaults.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-featutils-profile-shape",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py:145,343-346",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Within the same module 'profile' is created with inconsistent leveling: _init_feature_tensors builds profile as [batch, num_atoms, profile_dim] (line 145), while initialize_features_from_config builds profile as [batch, num_residues, profile_size] (line 344). Downstream extract_atom_features (lines 388-405) enforces all features share the same atom count; a residue-level 'profile' would fail that check, so the two builders are not interchangeable.",
      "recommendation": "Decide whether 'profile' is residue- or atom-level and make both builders consistent (and consistent with extract_atom_features' expectations).",
      "status": "survived"
    },
    {
      "id": "s2c6l2-005",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/validation_utils.py:31",
      "class": "design_defect",
      "severity": "low",
      "evidence": "validate_run_stageD_inputs computes `_ = n_atoms // n_residues if n_residues else None` and discards the result; the value is never used. Dead computation that suggests an intended atoms-per-residue check was never completed.",
      "recommendation": "Remove the dead statement or finish the intended validation it was meant to support.",
      "status": "survived"
    },
    {
      "id": "s2c6l1-tensorfixes-global-monkeypatch",
      "location": "rna_predict/pipeline/stageD/tensor_fixes/__init__.py:251-293,347-367,370-386 (invoked via run_stageD_unified.py:137)",
      "class": "design_defect",
      "severity": "low",
      "evidence": "apply_tensor_fixes and the *_fixes modules monkeypatch global PyTorch primitives process-wide: torch.Tensor.__add__ (tensor_operations.py:38, attention_fixes/__init__.py), torch.matmul/torch.bmm/torch.nn.functional.linear (tensor_operations.py:106-108), torch.nn.Linear.forward (attention_fixes.py / tensor_fixes/__init__.py:367) and torch.nn.Module.forward (tensor_fixes/__init__.py:386). On a RuntimeError these silently slice/truncate or even return one operand unchanged (tensor_operations.py:29-32), masking shape-correctness failures rather than surfacing them. Not an injection vector, but it is a global integrity/correctness hazard that can silently corrupt model outputs for any code in the process.",
      "recommendation": "Avoid patching global torch operators; fix shape handling at call sites or in dedicated wrapper modules. If retained, gate behind an explicit opt-in flag and fail loudly instead of returning silently coerced results.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-018",
      "location": "rna_predict/predict.py:326-377",
      "class": "design_defect",
      "severity": "low",
      "evidence": "RNAPredictor.predict_submission_original is a full alternate implementation superseded by predict_submission (:219). A full-repo review shows batch_predict (:413) and predict_submission are the live path; predict_submission_original is not called from production code.",
      "recommendation": "Remove the dead predict_submission_original method (or document it as legacy and exclude it from the public surface).",
      "status": "survived"
    },
    {
      "id": "s2c6l0-predict-orig-index",
      "location": "rna_predict/predict.py:357-366",
      "class": "bug",
      "severity": "low",
      "evidence": "predict_submission_original builds resname via [sequence[i] for i in residue_indices]; residue_indices falls back to list(range(n_atoms)) (line 358) when metadata is absent. If n_atoms > len(sequence) (typical: many atoms per residue), sequence[i] raises IndexError. Even with metadata, indices are not bounds-checked against len(sequence).",
      "recommendation": "Bound-check residue_indices against len(sequence) or map atoms to residues correctly before indexing the sequence string.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-020",
      "location": "rna_predict/predict.py:380-395",
      "class": "design_defect",
      "severity": "low",
      "evidence": "load_partial_checkpoint duplicates the partial-state-dict logic already provided by rna_predict/utils/checkpoint.py:partial_load_state_dict, with divergent semantics (here it pre-filters by matching shape and uses load_state_dict(strict=False); the util uses per-key copy_ with try/except). Two checkpoint loaders risk inconsistent behavior.",
      "recommendation": "Use the shared utils/checkpoint.partial_load_state_dict in predict.py to keep one checkpoint-loading policy.",
      "status": "survived"
    },
    {
      "id": "s2c6l1-predict-seqpath-fileread",
      "location": "rna_predict/predict.py:485",
      "class": "security",
      "severity": "low",
      "evidence": "main() reads arbitrary filesystem paths taken from the 'sequence_path' column of the input CSV: `with open(seq_path, 'r') as f: lines = f.readlines()` (predict.py:483-486). The path is not constrained to a base directory, so a crafted input_csv can make the process open any file the user can read (limited impact: only A/C/G/U lines are kept and the path is echoed to logs on error at predict.py:514).",
      "recommendation": "Resolve seq_path against a configured input root and reject paths that escape it (realpath containment check); validate the column values before opening.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-021",
      "location": "rna_predict/predict.py:79,97-99,412",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The primary, README-recommended inference path prints raw debug to stdout on every prediction (e.g. predict_3d_structure :79,:97-99 and batch_predict :412), polluting CLI/notebook output for end users.",
      "recommendation": "Route debug output through logger.debug guarded by a config flag rather than unconditional print().",
      "status": "survived"
    },
    {
      "id": "s2c6l2-026",
      "location": "rna_predict/runners/batch_runner.py:111",
      "class": "design_defect",
      "severity": "low",
      "evidence": "batch_runner's additional_files list includes runners/full_pipeline.py (:111) and runs it via `uv run <file>` (run_python_file :27). Per the Stage-1 inventory, full_pipeline.py is a library module with no __main__/CLI; executing it as a script does nothing useful (it only triggers import side effects such as the global logging.basicConfig) and produces no pipeline output in the combined log.",
      "recommendation": "Remove full_pipeline.py from the batch list or invoke runners/pipeline_cli.py (the actual full-pipeline entry point) instead.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-027",
      "location": "rna_predict/runners/demo_entry.py:1-6",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The module docstring opens 'main.py - Entry point for RNA_PREDICT package.' but the file is runners/demo_entry.py (Stage-1 notes it is byte-for-byte identical to rna_predict/main.py). The header was copy-pasted and no longer names the actual file.",
      "recommendation": "Update the docstring to reference demo_entry.py and its demo purpose.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-028",
      "location": "rna_predict/runners/demo_entry.py:34-40",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "demo_run_input_embedding() prints 'Now streaming the bprna-spot dataset...' and 'Showing the full dataset structure for the first row...' (:38-39) and returns True, but performs no dataset streaming or input-embedding work. The output claims behavior that does not occur, and the function docstring calls it 'a simple demonstration of the input embedding functionality'.",
      "recommendation": "Either implement the demonstrated behavior or change the messages/docstring to make clear this is an empty stub.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-025",
      "location": "rna_predict/runners/full_pipeline.py:399-414,479-490",
      "class": "design_defect",
      "severity": "low",
      "evidence": "There are two Stage-D guard blocks. The first (:401-414) only logs and creates a default atom_metadata then `pass`es without calling Stage D; the actual Stage D invocation and an identical atom_metadata default block are repeated later (:480-499). The first block's work is largely dead/duplicated.",
      "recommendation": "Remove the first redundant run_stageD block; keep a single atom_metadata-default + Stage-D-call path.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-030",
      "location": "rna_predict/runners/pipeline_cli.py:9-12",
      "class": "design_defect",
      "severity": "low",
      "evidence": "pipeline_cli registers RNAConfig in the ConfigStore under name 'default' (:10) while also using config_name='default' with config_path='conf', where runners/conf/default.yaml also exists. Registering a structured config under the same primary name as a config file creates an ambiguous/conflicting Hydra composition for the entry point.",
      "recommendation": "Register the schema under a distinct name (e.g. 'rna_predict_config') and reference it from a defaults list, instead of colliding with the 'default' config file name.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-032",
      "location": "rna_predict/scripts/hypot_test_gen.py:8-39",
      "class": "design_defect",
      "severity": "low",
      "evidence": "remove_logger_lines and fix_leading_zeros are duplicated across at least three modules per the Stage-1 inventory: rna_predict/scripts/hypot_test_gen.py, scripts/automation/hypot_test_gen.py, and scripts/test_utils/hypot_test_gen.py (plus tests/common/mock_hypot_test_gen.py). Divergent copies of the same helper risk drift.",
      "recommendation": "Consolidate to one canonical implementation and import it everywhere.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-lm-lr-hardcoded",
      "location": "rna_predict/training/rna_lightning_module.py:1008-1012",
      "class": "design_defect",
      "severity": "low",
      "evidence": "configure_optimizers hardcodes Adam(lr=1e-3) and ignores any learning rate / optimizer settings in cfg, contradicting the otherwise config-driven design and preventing hyperparameter control during the 'Experimental' training mode.",
      "recommendation": "Read optimizer type and lr from cfg.training with sensible defaults.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-038",
      "location": "rna_predict/training/rna_lightning_module.py:368,374-377,459,480,540-541,585-598",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The training forward/training_step paths emit many unconditional print() statements (e.g. '[DEVICE-PATCH]' :368,:377; '[DEBUG][FORWARD]' :374,:459; '[NOISE-PRINT]' :480; '[TRAIN DEBUG]' :540-541; per-parameter grad-norm prints :585-598), independent of debug_logging, producing heavy stdout noise during normal training.",
      "recommendation": "Replace print() with logger.debug gated on self.debug_logging; remove the per-step backward()/grad-norm diagnostic from the hot path.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-angleloss-norm",
      "location": "rna_predict/utils/angle_loss.py:29-32",
      "class": "bug",
      "severity": "low",
      "evidence": "With a mask, loss = (per-element MSE over [B,L,num_angles]) summed and divided by mask.sum()+1e-8. mask.sum() counts valid (B,L) positions only, but the numerator sums over the num_angles feature axis too, so the mean is inflated by a factor of num_angles versus the true per-element mean. (The Lightning training_step at rna_lightning_module.py:573 multiplies the denominator by the feature count, so the two loss implementations disagree.)",
      "recommendation": "Divide by (mask.sum() * num_angles) + eps, or expand the mask and divide by the masked element count, to get a true per-element mean.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-039",
      "location": "rna_predict/utils/angle_loss.py:5-35",
      "class": "design_defect",
      "severity": "low",
      "evidence": "angle_loss is referenced only by tests/utils/test_angle_loss.py (full-repo grep for imports/usage). The production training loss is reimplemented inline in rna_lightning_module.training_step (MSE + mask, :564-576) rather than calling this utility, so the two angle-loss definitions can diverge and the util is unused in production.",
      "recommendation": "Either use angle_loss() in training_step or remove it; keep one angle-loss definition.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-040",
      "location": "rna_predict/utils/checkpointing.py:17-32",
      "class": "design_defect",
      "severity": "low",
      "evidence": "save_trainable_checkpoint prints the entire model state_dict key list and all named_parameters to stdout on every save (:17-23,:30-32). For real models this dumps thousands of lines per checkpoint save.",
      "recommendation": "Demote these dumps to logger.debug or remove them.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-devmgmt-struct-mutate",
      "location": "rna_predict/utils/device_management.py:118-124",
      "class": "design_defect",
      "severity": "low",
      "evidence": "handle_device_error mutates the Hydra config in place (cfg.device_management.force_components_to_cpu = [] / .append(component_path)). If cfg is a struct-flagged OmegaConf DictConfig (the norm for structured configs), adding a new key raises ConfigAttributeError; and the comment 'for future runs' is misleading since the config object is not persisted across process runs.",
      "recommendation": "Avoid mutating the config; track forced-CPU components in a local/runtime structure, or OmegaConf.set_struct(cfg, False) deliberately with documentation.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-041",
      "location": "rna_predict/utils/device_management.py:15-127",
      "class": "design_defect",
      "severity": "low",
      "evidence": "device_management's get_device_for_component/handle_device_error are used only by rna_predict/pipeline/stageB/torsion/torsionbert_inference.py (full-repo grep). The central orchestrators ignore this module: runners/full_pipeline.py and training/rna_lightning_module.py do ad-hoc cfg.device handling and define their own move_to_device (rna_lightning_module.py:1014), duplicating logic and leaving the conf/device_management config group largely unconsumed.",
      "recommendation": "Standardize device selection/movement on this module across stages, or remove it and its config group if not adopted.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-042",
      "location": "rna_predict/utils/rna_backbone_extraction.py:57",
      "class": "bug",
      "severity": "low",
      "evidence": "extract_pdb_backbone_coords contains the statement `line[17:20].strip()` (:57) which slices the residue-name field but discards the result (no assignment, no side effect). Dead code indicating an intended residue-name capture that was dropped.",
      "recommendation": "Remove the dead expression or assign and use the residue name.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-backbone-cif-index",
      "location": "rna_predict/utils/rna_backbone_extraction.py:57,84,91-106",
      "class": "bug",
      "severity": "low",
      "evidence": "extract_cif_backbone_coords detects the atom_site loop via lines[lines.index(line)+1] (line 84) — lines.index returns the FIRST matching line (wrong for repeated lines) and can IndexError if 'loop_' is the last line — and then parses fixed column positions fields[2]/[4]/[5]/[6..8], assuming a specific mmCIF column order that is not guaranteed. Also line 57 (`line[17:20].strip()`) in the PDB parser is a dead expression whose result is discarded (residue name never captured).",
      "recommendation": "Parse the _atom_site loop header to map column names to indices instead of assuming positions; iterate with enumerate instead of lines.index; remove or use the dead line[17:20] read.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-044",
      "location": "rna_predict/utils/scatter_utils.py:16-20",
      "class": "design_defect",
      "severity": "low",
      "evidence": "layernorm() returns torch.zeros_like(x) whenever the last dim is 1, with the inline comment 'Return zeros to ensure zero mean (test will pass the mean check)' (:19). The behavior is admittedly coded to satisfy a test rather than from a numerically-motivated convention, and silently discards the input for dim-1 features.",
      "recommendation": "Define a principled dim-1 behavior (e.g. return input unchanged or raise) and decouple it from test expectations.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-045",
      "location": "rna_predict/utils/scatter_utils.py:57-69,68-69,83-86",
      "class": "design_defect",
      "severity": "low",
      "evidence": "scatter_mean silently grows dim_size to max(index)+1 (:58-60) and clamps indices into range (:65), masking caller errors (e.g. wrong segment count) instead of failing. It also prints index contents and dim_size on every call (:68-69, fallback :83-86), noisy in a hot per-token path.",
      "recommendation": "Validate index range against dim_size and raise on violation; remove or gate the per-call prints behind a debug flag.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-047",
      "location": "rna_predict/utils/shape_utils.py:16-125",
      "class": "design_defect",
      "severity": "low",
      "evidence": "adjust_tensor_feature_dim (:40-55) and adjust_attention_bias (:92-117) silently zero-pad or slice tensors to coerce shapes, the same defect-masking pattern as the stageD tensor_fixes; mismatches are hidden rather than surfaced.",
      "recommendation": "Restrict silent coercion to known-safe cases and log/assert otherwise so genuine shape bugs are not hidden.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-048",
      "location": "rna_predict/utils/submission.py:99-103",
      "class": "design_defect",
      "severity": "low",
      "evidence": "coords_to_df copies the identical x/y/z values into every repeat column x_1..x_n,y_1..,z_1.. (:100-103), so all 'prediction_repeats' are byte-identical duplicates rather than distinct predictions. (This helper backs the dead predict_submission_original path, predict.py:331.)",
      "recommendation": "Accept per-repeat coordinates (or document that this helper intentionally duplicates a single prediction across columns).",
      "status": "survived"
    },
    {
      "id": "s2c6l0-resmap-meta-seqlen",
      "location": "rna_predict/utils/tensor_utils/residue_mapping.py:399-412,145-147",
      "class": "bug",
      "severity": "low",
      "evidence": "In derive_residue_atom_map Method 1, n_residues_meta is computed from max(residue_indices)+1 and passed to _derive_map_from_metadata, which builds a map of length n_residues_meta and then calls _validate_residue_atom_map iterating that length while indexing sequence_list[res_idx] (line 147). If the provided sequence is shorter than n_residues_meta (metadata implies more residues than the sequence), this raises IndexError in the warning path.",
      "recommendation": "Reconcile n_residues_meta with len(sequence_list) (raise a clear error on mismatch) and guard sequence_list indexing in _validate_residue_atom_map.",
      "status": "survived"
    },
    {
      "id": "s2c6l2-053",
      "location": "scripts/analysis/analyze_code.sh:274-282,315,320-324",
      "class": "bug",
      "severity": "low",
      "evidence": "coverage is installed into the uv-managed environment (`uv run pip install coverage`, :277) and other tools are invoked via `uv run`, but the coverage commands are called as bare `coverage run/report/xml/json` (:315,:320-324) and availability is checked with `command -v coverage` (:305) against the system PATH. If coverage lives only in the uv env, these bare invocations fail (the script tolerates it via `|| true`, silently skipping coverage).",
      "recommendation": "Invoke coverage consistently via `uv run coverage ...` (and check with `uv run coverage --version`).",
      "status": "survived"
    },
    {
      "id": "s2c6l0-analyze-undefvar",
      "location": "scripts/analysis/analyze_code.sh:404,420",
      "class": "bug",
      "severity": "low",
      "evidence": "Coverage reporting references $TARGET_BASENAME (lines 404 and 420), but that variable is never assigned anywhere in the script (the defined variable is BASENAME at line 288). Under `set -u` it would error; as-is it expands to empty, producing misleading messages like 'Coverage for  is ...'.",
      "recommendation": "Use the defined $BASENAME (or define TARGET_BASENAME) in the coverage messages.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-012",
      "location": "scripts/automation/batch_test_generator.py:31-43",
      "class": "bug",
      "severity": "low",
      "evidence": "main() is defined (lines 31-42) and prints a usage/processing flow, but there is no `if __name__ == '__main__': main()` guard, so executing `python batch_test_generator.py <folder>` does nothing. Additionally run_test_generation (lines 28-29) is a `pass` stub returning None, so in process_folder `result` is always falsy and it always prints 'Failed to generate tests for {file}' (lines 23-24). The file is labeled a stub, but the missing entrypoint makes even the stub path unreachable.",
      "recommendation": "Add the `if __name__ == '__main__': main()` guard; have run_test_generation return a truthy success indicator (or raise NotImplementedError) so the stub's control flow is meaningful.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-007",
      "location": "scripts/automation/commit_individual_files.sh:62-139",
      "class": "security",
      "severity": "low",
      "evidence": "The script iterates over data folders (RNA_NET, SPOT_RNA_PDB_dataset, bpRNA, kaggle), runs `git add \"$file\"` and `git commit` for every file found, then unconditionally `git push origin main`. There is no filtering/.gitignore enforcement or review step, so any secrets, credentials, or PII inadvertently present in those dataset directories would be committed and published to the remote automatically.",
      "recommendation": "Add an explicit allowlist/denylist and a dry-run/confirmation gate; never auto-push. Respect .gitignore and scan staged files for secrets before committing.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-016",
      "location": "scripts/automation/create_github_issues.py:11",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Usage docstring example references a foreign project: `python quick-fixes/automation-scripts/create_github_issues.py ImmortalDemonGod ProjectEquiSurv /Users/tomriddle1/ProjectEquiSurv/issue.json` (:11). This repo is RNA_PREDICT, the path quick-fixes/automation-scripts/ does not exist here, and ProjectEquiSurv is an unrelated repo — stale copied documentation.",
      "recommendation": "Update the usage example to this repository and a valid in-repo path.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-018",
      "location": "scripts/automation/github_automation/analyze_commit_log.py:8",
      "class": "design_defect",
      "severity": "low",
      "evidence": "LOG_FILE is hardcoded to 'all_commits.log' (:8) read from CWD, but github_automation.sh writes per-branch logs to $OUTPUT_DIR/logs/<branch>.log (github_automation.sh:246), never a combined 'all_commits.log'. The analyzer's input is thus orphaned from how logs are actually produced. Also imports Counter/defaultdict/datetime/timedelta (:2-3) that are unused.",
      "recommendation": "Accept the log path as an argv parameter (defaulting sensibly) and remove the unused imports.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-015",
      "location": "scripts/automation/github_automation/pull_requests/convert_prs_to_markdown.py:75-77,108",
      "class": "bug",
      "severity": "low",
      "evidence": "create_markdown_content does `pr['author'].get('name', ...)` etc.; GitHub returns a null author for PRs by deleted ('ghost') users, in which case pr['author'] is None and `.get` raises AttributeError, aborting conversion of that PR (no per-item try/except in main, lines 113-124). Separately, main reads `Path('pull_requests_full_pretty.json')` (line 108) from CWD, but the generator writes that file under a subdirectory `$OUTPUT_DIR/pull_requests/pull_requests_full_pretty.json` (github_automation.sh:226), so the expected input path does not match what the pipeline produces.",
      "recommendation": "Guard author access (`author = pr.get('author') or {}`); align the input path with the generator's output location (or accept it as an argument).",
      "status": "survived"
    },
    {
      "id": "s2c7l2-003",
      "location": "scripts/automation/hypot_test_gen.py:1-30",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Three divergent copies of hypot_test_gen.py exist: scripts/automation/hypot_test_gen.py and rna_predict/scripts/hypot_test_gen.py each define only the helper stubs fix_leading_zeros/remove_logger_lines, while scripts/test_utils/hypot_test_gen.py is the full 897-line generator. Likewise two batch_test_generator.py copies (automation stub vs test_utils real). The reorganize_scripts.sh `mv` (lines 10-13) was supposed to relocate these out of rna_predict/scripts/, yet rna_predict/scripts/hypot_test_gen.py still exists, leaving stale duplicates that confuse imports (see s2c7l2-001).",
      "recommendation": "Consolidate to a single canonical hypot_test_gen.py / batch_test_generator.py and remove the stale stub copies under scripts/automation/ and rna_predict/scripts/.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-013",
      "location": "scripts/coverage/show_coverage.py:14-24",
      "class": "bug",
      "severity": "low",
      "evidence": "run_command passes `check=True` only on the non-capture branch (line 17); when capture_output=True (line 15) it omits check, so subprocess.CalledProcessError is never raised on a nonzero exit and the except block at lines 19-24 (which references e.stdout/e.stderr) is dead for captured calls. Callers show_least_covered (line 268) and filter_coverage (line 346) then proceed to parse `result.stdout`, which on a failed `coverage report -m` is empty, silently yielding 'No files...' rather than surfacing the error.",
      "recommendation": "Pass check=True (and capture stderr) in both branches, or explicitly inspect result.returncode after capture and raise/report on failure.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-028",
      "location": "scripts/coverage/show_coverage.py:231",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Coverage scripts disagree on the memory-profiling plugin: show_coverage.py invokes pytest with `--memprof-top-n=10 --memprof-csv-file=...` (pytest-memprof) at :231, while run_failing_tests.sh uses `--memray --most-allocations=10 --stacks=5` (pytest-memray) at run_failing_tests.sh:383,405. run_command uses check=True (:17) so an absent plugin aborts the whole run before the coverage report. The two memory tooling choices are inconsistent across the coverage toolset.",
      "recommendation": "Standardize on one memory-profiling plugin across the coverage scripts and ensure it is declared in requirements-test.txt.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-027",
      "location": "scripts/coverage/show_coverage.py:54-191",
      "class": "design_defect",
      "severity": "low",
      "evidence": "show_coverage.py contains two parallel implementations of the same report logic. The helper set parse_coverage_report (:27), filter_coverage_report (:54), get_least_covered_report (:88) and parse_missing_lines (:147) are never called; main() instead uses the inline reimplementations show_least_covered (:265) and filter_coverage (:343). The unused functions are dead/duplicate code.",
      "recommendation": "Delete the unused helper functions or refactor main()'s inline logic to call them, keeping a single implementation.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-030",
      "location": "scripts/demo_stochastic_inference.py:62-68",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The uniqueness check hardcodes 5 repeats (`range(5)`, columns x_1..x_5/y_/z_ at :62-66 and the `len(unique_structs) == 5` assertion at :68), but predict_submission is called with no prediction_repeats (predict.py:219 default None → driven by config). If config sets repeats != 5, the x_5/y_5/z_5 columns won't exist (KeyError) or the success criterion is wrong. The demo silently assumes a config value it does not pass or read.",
      "recommendation": "Read the repeat count from cfg (or pass prediction_repeats=5 explicitly) and derive the column range and success threshold from that value.",
      "status": "survived"
    },
    {
      "id": "s2re1-006",
      "location": "scripts/inspect_checkpoint.py:10; scripts/inspect_pt_file.py:10; scripts/partial_checkpoint_full_pipeline_script.py:100",
      "class": "security",
      "severity": "low",
      "evidence": "Three developer/utility scripts call torch.load(...) (map_location='cpu' / default) with no weights_only=True against arbitrary user-supplied checkpoint/.pt paths. Same unsafe-pickle deserialization class as s2re1-002 but in tooling rather than the inference path; an operator pointing these at an untrusted .pt executes embedded pickle payloads.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c7l1-003",
      "location": "scripts/partial_checkpoint_full_pipeline_script.py:100",
      "class": "security",
      "severity": "low",
      "evidence": "checkpoint = torch.load(partial_ckpt_path) is called without weights_only=True. Here partial_ckpt_path is a self-created tempfile (line 88-91), so the immediate risk is low, but the default-pickle pattern is unsafe and would be a deserialization vector if the path were ever pointed at an externally produced checkpoint.",
      "recommendation": "Add weights_only=True (or map_location and a safe loader) to torch.load to harden the pattern even for self-produced checkpoints.",
      "status": "survived"
    },
    {
      "id": "s2re4-003",
      "location": "scripts/partial_checkpoint_full_pipeline_script.py:88",
      "class": "security",
      "severity": "low",
      "evidence": "partial_ckpt_path = tempfile.mktemp(suffix=\"_partial.ckpt\") uses tempfile.mktemp(), deprecated since Python 2.3 due to a TOCTOU race (the returned name can be created/symlinked by another process between mktemp() and the subsequent write at line 91 / read at line 100). Should use tempfile.mkstemp()/NamedTemporaryFile.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c7l2-029",
      "location": "scripts/reorganize_scripts.sh:10-13",
      "class": "design_defect",
      "severity": "low",
      "evidence": "This one-time migration `mv`s scripts out of rna_predict/scripts/ (e.g. hypot_test_gen.py at :13) but updates none of the moved files' internal import paths or PROJECT_ROOT computations, directly causing the broken import in scripts/test_utils/batch_test_generator.py (s2c7l2-001) and the wrong PROJECT_ROOT in scripts/run_all_pipeline.py (s2c7l2-009). It also left rna_predict/scripts/hypot_test_gen.py in place (still present), so the 'move' was partial and stale duplicates remain.",
      "recommendation": "After such moves, update intra-repo imports and path assumptions, and verify rna_predict/scripts/ no longer holds duplicated relocated files; treat this script as historical (it is non-idempotent and would fail re-running).",
      "status": "survived"
    },
    {
      "id": "s2c7l1-009",
      "location": "scripts/run_mutation_tests.sh:42-44",
      "class": "security",
      "severity": "low",
      "evidence": "CMD=\"mutatest -n $NUM_MUTATIONS -m $MUTATION_MODE -o $OUTPUT_FILE\" is later executed via unquoted `$CMD` expansion (`if ! $CMD 2>&1 | tee ...`). NUM_MUTATIONS/MUTATION_MODE come from unvalidated -n/-m CLI args and undergo word-splitting, allowing argument injection (e.g. extra mutatest flags) into the invoked command.",
      "recommendation": "Use an array (cmd=(mutatest -n \"$NUM_MUTATIONS\" -m \"$MUTATION_MODE\" -o \"$OUTPUT_FILE\")) and invoke \"${cmd[@]}\"; validate that NUM_MUTATIONS is an integer and MUTATION_MODE is in an allowed set.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-014",
      "location": "scripts/screen_finder_app/config.py:27-58",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "The documented example templates_config.json structure uses an action object keyed by `\"type\"` with values like 'text'/'clipboard' (e.g. `\"action\": {\"type\": \"click\", ...}`). However execute_action reads `action_config.get(\"action\", \"\")` (scripts/screen_finder_app/main.py:29) and the real templates/templates_config.json uses `\"action\": {\"action\": \"click\"}`. A config authored from the docstring example (key 'type') would always fall through to 'No recognized action' (main.py:57-58). The action-type names ('text' vs 'text_input') also disagree with execute_action's branches.",
      "recommendation": "Update the config.py docstring example to use the `action` key and the action-type names actually handled by execute_action (click, text_input, clipboard, double_click), or make execute_action accept both 'type' and 'action' keys.",
      "status": "survived"
    },
    {
      "id": "s2c7l0-016",
      "location": "scripts/screen_finder_app/gui_launcher.py:58,61,100,142",
      "class": "bug",
      "severity": "low",
      "evidence": "periodic_search_thread runs on a background daemon thread (started at gui_launcher.py:161) yet makes direct Dear PyGui calls from that thread — dpg.get_value('-INTERVAL-') (line 58) and multiple dpg.set_value('-STATUS-', ...) (lines 61,120,142) and execute_action side effects. The code's own comments (lines 46,51,251-253) acknowledge that direct DPG calls from non-main threads are unsafe and can crash/corrupt the GUI; the main render loop (lines 250-263) already polls status, making the threaded set_value calls both redundant and hazardous.",
      "recommendation": "Restrict DPG access to the main thread: have the worker thread update plain Python state (status_message, interval read once or via a thread-safe value) and let the render loop apply it, removing dpg.get_value/set_value calls from periodic_search_thread.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-008",
      "location": "scripts/screen_finder_app/main.py:25-58",
      "class": "security",
      "severity": "low",
      "evidence": "execute_action() drives pyautogui to click, double-click, type arbitrary text (action_config.get('text')), and send copy/paste hotkeys based entirely on templates_config.json (loaded in template_loader.py:30-32 with no validation of the action payload). If templates_config.json or a template image is tampered with, the tool will autonomously inject keystrokes/clicks into whatever application is focused, a local input-automation abuse vector.",
      "recommendation": "Validate and constrain action types/payloads against a strict schema, and treat templates_config.json as trusted input with restrictive file permissions; warn the user that text_input/clipboard actions execute arbitrary keystrokes.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-024",
      "location": "scripts/screen_finder_app/screenshot.py:17",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The package's two real entry points (main.py:60 main(), gui_launcher.py:36 periodic_search_thread) reimplement screen capture and template matching inline (main.py:84-101, gui_launcher.py:70-85) rather than calling the dedicated helpers: capture_all_monitors (screenshot.py:17), validate_and_match_template (template_matching.py:17), and select_region (region_selector.py:80) are never invoked by either entry point. These three modules are effectively orphaned/duplicate logic (e.g. screenshot.py returns BGR while main.py needs grayscale).",
      "recommendation": "Either route main.py/gui_launcher.py through these helpers (removing the inline duplicates) or delete the unused modules to avoid divergent matching implementations.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-023",
      "location": "scripts/screen_finder_app/template_matching.py:1",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Stale path header comments survive the reorganize move: template_matching.py:1 '# rna_predict/scripts/template_matching.py', logger.py:1, region_selector.py:1, screenshot.py:1 all still claim the old rna_predict/scripts/ location, whereas reorganize_scripts.sh:37 moved screen_finder_app to scripts/screen_finder_app/.",
      "recommendation": "Update or remove the path-header comments to reflect scripts/screen_finder_app/.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-022",
      "location": "scripts/screen_finder_app/template_matching.py:109-116",
      "class": "bug",
      "severity": "low",
      "evidence": "validate_and_match_template returns a tuple `(match_location_tuple, correlation_score)` (:79), but the __main__ self-test asserts `location == (200, 100)` (:116) against that 2-tuple-of-(tuple,float). The comparison is always False, so the example's assert raises AssertionError on a successful match. The comment 'Expected location: (200, 100)' (:115) reflects the same misunderstanding of the function's own return shape.",
      "recommendation": "Assert on location[0] (the coordinate tuple), e.g. `assert location[0] == (200, 100)`.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-004",
      "location": "scripts/test_utils/hypot_test_gen.py:25",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "Comment at :24 and PROMPT_TEMPLATE_FILE = Path(__file__).parent / 'prompt_template.md' (:25) assume prompt_template.md sits beside this script, but no such file exists in scripts/test_utils/ (only docs/examples/prompt_template.md exists, verified by find). load_text_prompt_template() therefore logs an error and returns '' (:34-38), so wrap_with_prompt produces an empty ''.format(...) result and the final test_wrapped_*.md is effectively blank.",
      "recommendation": "Ship a prompt_template.md alongside the script, or point PROMPT_TEMPLATE_FILE at the existing docs/examples/prompt_template.md.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-013",
      "location": "setup.py:32",
      "class": "doc_drift",
      "severity": "low",
      "evidence": "setup.py sets python_requires='>=3.8' (:32) while the toolchain targets Python 3.10 (mypy.ini per Stage-1 inventory configures Python 3.10). Mismatched declared interpreter floors between configs.",
      "recommendation": "Align python_requires with the actually supported/tested version (3.10) across setup.py and pyproject.toml.",
      "status": "survived"
    },
    {
      "id": "s2re4-008",
      "location": "test_download.py:1 and simple_test.py:1 (repo root)",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Two pytest files live at the repository root (test_download.py, simple_test.py) outside the collection scope: pytest.ini:4 sets 'testpaths = tests' and .vscode/settings.json points pytestArgs at ['tests'], so these root files are never collected/run. test_download.py is byte-for-byte identical (verified via diff, both 58 lines) to tests/test_download.py, making the root copy a stale orphan duplicate; simple_test.py is an orphan that uses unittest while the suite is pytest-based. Dead/uncollected test artifacts give a false impression of coverage and drift from the canonical copy.",
      "recommendation": "",
      "status": "survived"
    },
    {
      "id": "s2c7l2-026",
      "location": "tests/stageA/integration/conf/default.yaml:2",
      "class": "design_defect",
      "severity": "low",
      "evidence": "This integration-test config nests all Stage A params under a top-level `stageA:` key (:2), whereas the canonical conf/model/stageA.yaml is deliberately flat — its comment states 'Direct stageA configuration without double nesting' (rna_predict/conf/model/stageA.yaml:5) with params at top level. The test config also omits several keys present in the canonical file (checkpoint_zip_path, debug_logging, freeze_params, run_example, example_sequence, visualization.output_path), so the test exercises a differently-shaped config than production Stage A.",
      "recommendation": "Mirror the canonical (un-nested) stageA config shape and key set in the test fixture, or document why the test intentionally uses a divergent schema.",
      "status": "survived"
    },
    {
      "id": "s2c7l1-006",
      "location": "tests/stageA/integration/conf/default.yaml:7-8",
      "class": "security",
      "severity": "low",
      "evidence": "checkpoint_url: \"https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1\" points Stage A at a third-party personal Dropbox archive with no integrity verification (no checksum/signature). The downloaded archive is extracted and the resulting .pth is loaded by Stage A (the download/extract/torch.load code lives outside this file set, unverified here). A compromised or swapped link is a supply-chain vector that, combined with pickle-based checkpoint loading, can lead to code execution.",
      "recommendation": "Pin and verify a SHA-256 of the downloaded archive/checkpoint before extraction/loading, and host the artifact on a controlled, versioned location rather than a personal Dropbox share.",
      "status": "survived"
    },
    {
      "id": "s2c0l1-013",
      "location": ".env.example:1-3",
      "class": "security",
      "severity": "info",
      "evidence": "File documents required secrets ANTHROPIC_API_KEY and PERPLEXITY_API_KEY using placeholder values only ('your-api-key-here', 'pplx-abcde') — no real credentials are present. .gitignore:104 also excludes the real `.env`. This is correct handling; recorded as a positive/no-leak observation for the assigned file, not a defect.",
      "recommendation": "No action needed; continue keeping real secrets out of .env.example and ensure .env stays gitignored.",
      "status": "survived"
    },
    {
      "id": "s2c0l0-023",
      "location": ".env.example:6",
      "class": "doc_drift",
      "severity": "info",
      "evidence": "Example env defaults reference `MODEL=claude-3-7-sonnet-20250219` and recommend `claude-3-opus-20240229`. These are stale model identifiers for the (separate) Task Master tooling and do not affect the RNA pipeline; recorded as informational drift, not a pipeline defect.",
      "recommendation": "If Task Master tooling is retained, refresh the example to current model ids; otherwise remove the unrelated env template.",
      "status": "survived"
    },
    {
      "id": "s2c0l2-0043",
      "location": ".gitignore:182 (.vscode); .idea at :181",
      "class": "design_defect",
      "severity": "info",
      "evidence": ".gitignore ignores `.vscode` (line 183, from the Task Master block) and `.idea` (line 181), yet .vscode/settings.json is a tracked, inventoried config file (audit/01-understanding.md:70). As with package.json, the ignore rule contradicts the intent to version-control the VS Code workspace settings.",
      "recommendation": "Negate the tracked file (e.g. add `!.vscode/settings.json`) or remove the broad .vscode ignore.",
      "status": "survived"
    },
    {
      "id": "s2c1l0-rnaconfig-docstring-misplaced",
      "location": "rna_predict/conf/config_schema.py:1344-1349",
      "class": "doc_drift",
      "severity": "info",
      "evidence": "In RNAConfig the experiment_name field (lines 1345-1348) is declared before the intended class docstring string literal on line 1349 ('Root configuration for the entire RNA_PREDICT pipeline.'). Because a docstring must be the first statement, RNAConfig.__doc__ is None and the string is a no-op expression, so the class documentation is effectively lost.",
      "recommendation": "Move the docstring to be the first statement in the class body (above experiment_name).",
      "status": "survived"
    },
    {
      "id": "s2c5l1-stageD-debug-stdout-leak-005",
      "location": "rna_predict/pipeline/stageC/mp_nerf/rna/rna_folding.py:17",
      "class": "security",
      "severity": "info",
      "evidence": "Module import unconditionally prints the absolute source path via print(f\"!!!!!!!!!! CASCADE MODULE-LEVEL: rna_folding.py LOADED FROM: {__file__} !!!!!!!!!!\") at :17 (and another unconditional print at :180,:247). Across the assigned Stage D files there is pervasive unconditional/print-based dumping of full resolved configs, tensor shapes, sys.path and cwd (e.g. run_stageD.py:397-399 prints CWD/SCRIPT DIR/sys.path; diffusion_module.py:70-96 prints full cfg; bridging files print tensor metadata). These leak deployment filesystem layout and configuration to stdout regardless of the debug_logging flag, an information-disclosure / log-hygiene weakness rather than an exploitable vulnerability.",
      "recommendation": "Gate all diagnostic output behind the existing debug_logging flag and the logging module (no bare print), and remove the unconditional module-level path prints so absolute paths, sys.path, and full configs are not emitted in production runs.",
      "status": "survived"
    },
    {
      "id": "s2c5l1-mpnerf-filepath-parse-no-hardening-004",
      "location": "rna_predict/pipeline/stageC/mp_nerf/utils.py:298-321",
      "class": "security",
      "severity": "info",
      "evidence": "get_coords_from_file() dispatches on a caller-supplied file_path suffix to get_coords_from_pdb (:236-256) / get_coords_from_cif (:276-295), which feed the path directly into BioPython PDBParser/MMCIFParser.get_structure(). The path is used as-is with no canonicalization, allow-list, or size/complexity limits, and parse errors are re-wrapped with the full path echoed into the exception message (:242,:282). This is consistent with the intended purpose (loading user structure files), so there is no path-traversal escalation, but malformed/oversized structure files are an untrusted-input parsing surface with no resource bounds (DoS potential) when these helpers are exposed to externally provided files. No code-execution sink is present.",
      "recommendation": "If these loaders ever ingest untrusted/uploaded structures, add input validation (size limits, expected extension/content checks) and avoid echoing full filesystem paths in error strings surfaced to callers. Otherwise document that file_path must be a trusted, locally-controlled path.",
      "status": "survived"
    },
    {
      "id": "s2c6l0-valutils-dead-div",
      "location": "rna_predict/pipeline/stageD/stage_d_utils/validation_utils.py:24-35",
      "class": "design_defect",
      "severity": "info",
      "evidence": "validate_run_stageD_inputs computes `_ = n_atoms // n_residues if n_residues else None` (line 31) and discards it, and otherwise only raises when s_trunk is atom-level. z_trunk and s_inputs arguments are accepted but never validated, so the 'validation' is largely a no-op beyond one assertion.",
      "recommendation": "Remove the dead computation and either validate the other tensors (shape/level expectations) or trim the unused parameters.",
      "status": "survived"
    },
    {
      "id": "s2c7l2-025",
      "location": "scripts/screen_finder_app/py.typed:1",
      "class": "other",
      "severity": "info",
      "evidence": "py.typed is an empty PEP 561 inline-types marker (0 bytes, verified via wc -c), but scripts/screen_finder_app/ is vendored tooling under scripts/ and is not a distributed/installed package (no packaging entry references it). A py.typed marker has no effect outside an installed, importable distribution, so it is inert here.",
      "recommendation": "Remove py.typed from this non-packaged script directory, or package screen_finder_app properly if inline-type advertising is intended.",
      "status": "survived"
    }
  ]
}
```
