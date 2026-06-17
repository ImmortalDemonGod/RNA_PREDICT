# 04 — Goal + External Research

## Long-term goal candidates (plural by design; grounded in Stages 1–3)
### Candidate 1 — Provide a working sequence-to-3D RNA tertiary structure inference pipeline: given an RNA sequence (or CSV of sequences) produce per-residue/atom 3D coordinates output as CSV/PDB/.pt files, via the staged TorsionBERT (Stage B) -> MP-NeRF reconstruction (Stage C) chain exposed through RNAPredictor. _(grounded)_
- **Falsifiable success signals:**
  - Running the README-recommended command `uv run rna_predict/predict.py input_csv=... checkpoint_path=...` completes exit 0 and writes prediction_{i}.csv/.pdb/.pt and summary.csv to output_dir
  - RNAPredictor (rna_predict/predict.py:21, re-exported via interface.py) chains Stage B torsion prediction + Stage C reconstruction and returns atom coordinates for an arbitrary input sequence without hardcoded developer paths
  - The `rna_predict` console entry point resolves and runs instead of raising ModuleNotFoundError
  - Inference does not depend on a developer-local checkpoint path; a published/relative checkpoint loads successfully
- **Grounding:**
  - audit/01-understanding.md:6 (provisional intent: sequence-to-structure inference producing CSV/PDB/.pt is the apparent primary deliverable; README marks Inference as Functional)
  - README.md:7-11 (Project Status: Inference = ✅ Functional)
  - README.md:20-44 (recommended `uv run rna_predict/predict.py` command and CSV/PDB/.pt + summary.csv output files)
  - audit/01-understanding.md:28 (predict.py:440-543 primary README-recommended inference entry; batch_predict at :398 writes per-prediction CSV/PDB/.pt and summary.csv)
  - audit/03-execution.md:35-36 (predict.py runs: Hydra config loads but fails at HuggingFace tokenizer download then torch.load of hardcoded /Users/tomriddle1/... checkpoint — the inference goal is implemented but currently blocked by hardcoded path/network)
  - audit/03-execution.md:34 (`uv run rna_predict` => ModuleNotFoundError: No module named 'rna_predict.__main__'; console entry broken)
  - audit/02-static-audit.md s2c1l0-predict-yaml-hardcoded-ckpt (predict.yaml:13 checkpoint_path hardcoded to /Users/tomriddle1/... blocks inference off the developer machine)

### Candidate 2 — Package the pipeline for the Kaggle RNA 3D folding competition: ingest the competition dataset, run prediction over the test set, and emit a validated submission CSV via a dedicated Kaggle harness. _(grounded)_
- **Falsifiable success signals:**
  - kaggle/rna_predict.py in mode=predict runs data verification + full test-set processing and produces a submission that passes submission_validator checks
  - Submission generation does not depend on the hardcoded /Volumes/Totallynotaharddrive/... external-drive data path
  - interface.py demo writes submission.csv for an input sequence
  - submission_validator raises catchable exceptions (not sys.exit) so it is usable programmatically in the submission flow
- **Grounding:**
  - audit/01-understanding.md:6 (intent includes packaging for Kaggle RNA-folding competition submissions, rna_predict/kaggle/**)
  - audit/01-understanding.md:19 (kaggle/rna_predict.py:284-312 Kaggle harness: mode=predict runs data verification, sanity check, full test-set processing and submission validation)
  - audit/01-understanding.md:18 (interface.py writes submission.csv)
  - audit/02-static-audit.md s2c1l2-hardcoded-external-drive-data-path (kaggle/rna_predict.py:77 BASE_INPUT_ROOT_EXTERNAL_DRIVE = /Volumes/Totallynotaharddrive/...; integration test_real_rna_predict fails FileNotFoundError)
  - audit/03-execution.md:31,82 (Kaggle runner external-service/hardcoded-path; integration test_real_rna_predict fails because hardcoded CSV path does not exist)
  - audit/02-static-audit.md s2c1l0-submission-validator-sysexit (submission_validator.py:35 sys.exit terminates process instead of raising)
  - docs/pipeline/kaggle_info/kaggle_competition.md and M2_Plan.md (Kaggle competition planning docs, audit/01-understanding.md:117-118)

### Candidate 3 — Build a full AlphaFold3-inspired multi-stage RNA structure model (Stage A 2D adjacency/RFold -> Stage B torsion via TorsionBERT+Pairformer -> Stage C MP-NeRF reconstruction -> Stage D diffusion refinement) that is end-to-end trainable on RNA structural data using PyTorch Lightning. _(needs-human-confirm)_
- **Falsifiable success signals:**
  - `python -m rna_predict.training.train` imports and runs a Lightning training loop to completion without the hardcoded /Users/tomriddle1/... config_path and without the deepspeed/torch import failure
  - run_full_pipeline (runners/full_pipeline.py:334) executes the complete A->B->C->D chain producing output tensors
  - Stage D diffusion components are exercised by tests (currently 0% coverage) and import without the deepspeed AttributeError
  - Trained weights of all submodules are registered in state_dict (FeatureProcessor/AttentionComponents subclass nn.Module) and persist across checkpoint save/load
- **Grounding:**
  - audit/01-understanding.md:9 (four-stage A/B/C/D pipeline, each standalone-runnable, orchestrated by run_full_pipeline; training uses PyTorch Lightning train.py + rna_lightning_module.py)
  - README.md:7-11 (Training = 🚧 Experimental, Diffusion = 🧪 Research Prototype — full trainable model is aspirational, not yet functional)
  - audit/03-execution.md:38 (`python -m rna_predict.training.train --help` fails: import chain -> deepspeed AttributeError torch.library.custom_op; hardcoded @hydra.main config_path=/Users/tomriddle1/...)
  - audit/03-execution.md:33,52 (stageD diffusion components diffusion_module.py/diffusion_conditioning.py at 0% coverage; stageB pairwise ~2-3%)
  - audit/02-static-audit.md s2c3l0-001 (FeatureProcessor etc. are plain classes, parameters not registered in state_dict — training/checkpointing of those layers is broken)
  - audit/03-execution.md s2c4l0-pairformer-wrapper-predict-random (pairformer_wrapper.py:399-400 predict() uses torch.randn embeddings — pairwise branch not producing learned outputs)

### Candidate 4 — Serve as a research/experimentation platform and reference implementation for staged RNA 3D prediction methods (RFold secondary structure, TorsionBERT torsion angles, MP-NeRF forward kinematics, diffusion refinement, residue-atom bridging), documented for study and method comparison. _(speculative)_
- **Falsifiable success signals:**
  - Each stage is independently runnable via its own Hydra @main (run_stageA/stageB/stageC/run_stageD) for isolated experimentation
  - Extensive design/reference docs map implemented components to the literature (AF3, RFold, TorsionBERT, MP-NeRF)
  - benchmarks/benchmark.py compares naive vs optimized input-embedding implementations and reports latency/memory
- **Grounding:**
  - audit/01-understanding.md:9 (each stage runnable standalone via its own Hydra @main: run_stageA.py:153, stageB/main.py:518, stage_c_reconstruction.py:491, run_stageD.py:392)
  - audit/01-understanding.md:16 (benchmarks/benchmark.py:393-399 compares naive vs optimized input-embedding, measures latency/memory; referenced README.md:90)
  - audit/01-understanding.md:143-159 (docs/reference/advanced_methods/** AF3, diffusion, LoRA, isostericity, residue_atom_bridging design specs and reference literature)
  - audit/01-understanding.md:43 (role distribution: doc=141 — heavy documentation surface consistent with a study/reference repo)

### Candidate 5 — Maintain a heavily-tooled, CI/quality-gated Python research codebase: enforce coverage thresholds, linting/typing, mutation testing, containerization and AI-assisted task management as part of the engineering process. _(speculative)_
- **Falsifiable success signals:**
  - `make test` (ruff + mypy + pytest --cov) passes cleanly under CI on the supported Python (>=3.10, CI 3.11)
  - mutation testing (cosmic-ray.dev.toml / mutatest.ini) runs against a green test suite
  - Containerfile builds a runnable image (currently FROM python:3.7-slim, incompatible with required deps)
  - Task Master CLI (scripts/dev.js) resolves its modules and runs
- **Grounding:**
  - audit/03-execution.md:9-16 (Makefile test/lint targets, ruff/mypy, coverage xml/html; CI main.yml:50)
  - audit/03-execution.md:24-25 (cosmic-ray.dev.toml and mutatest.ini mutation-testing configs)
  - audit/03-execution.md:40 (`node scripts/dev.js list` fails ERR_MODULE_NOT_FOUND scripts/modules/commands.js — Task Master tooling incomplete)
  - audit/03-execution.md s2c0l0-002 (Containerfile:1 FROM python:3.7-slim incompatible with Python>=3.10 / torch wheels)
  - audit/01-understanding.md:9 (repo vendors a Node.js Task Master AI dev-task CLI and a screen-finder GUI app, largely unrelated to the RNA pipeline)
  - audit/02-static-audit.md Summary (538 findings; this is a derived audit observation, not a stated repo goal — tooling intent is inferred from config artifacts)

## External research (cross-checked; uncorroborated = recorded as unverified)
### Ideas that materially advance the goal
| Idea | Relevance | Sources |
| --- | --- | --- |
| Treat Stage A secondary-structure quality as the primary tunable lever for end-to-end accuracy, since multiple independent benchmarks identify SS quality (not the 3D module) as the dominant accuracy driver; invest in a stronger/pretrained SS prior feeding Stage B rather than scaling the reconstruction module. | RNA_PREDICT is already staged (Stage A SS -> Stage B torsion -> Stage C MP-NeRF). Corroborated evidence (3 sources) that SS quality is the bottleneck means improving Stage A yields the highest marginal return and validates the existing topology over a sequence-only design. | https://www.nature.com/articles/s42256-026-01223-x; https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715; https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full |
| Evaluate replacing or augmenting the Stage B TorsionBERT encoder with a larger structure-aware RNA language model (e.g. RiNALMo as used by DeepRNA-Twist, or ERNIE-RNA's base-pairing-aware embeddings) to improve torsion-angle quality while preserving the project's MSA-free, single-sequence inference design. | Stage B (rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py) is the upstream determinant of Stage C coordinates. The directional benefit of larger structure-aware RNA LMs is corroborated, though each tool's specific SOTA-over-TorsionBERT claim is unverified -- so this is a research experiment to A/B, not a guaranteed win. | https://academic.oup.com/bib/article/26/3/bbaf199/8124238; https://www.nature.com/articles/s41467-025-64972-0; https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full |
| Adopt the TorsionBERT-derived TB-MCQ torsion-quality scoring function as both a Stage B training signal and an intrinsic pipeline evaluation metric, giving a coordinate-free way to score torsion outputs before Stage C reconstruction. | TB-MCQ comes from the exact model RNA_PREDICT loads ('sayby/rna_torsionbert'), so it is directly applicable to the deployed Stage B and to gating/regression-testing torsion quality independent of the (lossy) MP-NeRF step. | https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663; https://huggingface.co/sayby/rna_torsionBERT |
| Add end-to-end coordinate supervision plus deep geometric restraints (inter-nucleotide distances/orientations) across the Stage B->C boundary, following the DRFold family, so Stage B is trained against final 3D error rather than torsion MAE alone. | DRFold attributes its large TM-score gains specifically to coordinate-supervised end-to-end learning combined with composite geometric potentials. RNA_PREDICT's frame->Cartesian decomposition (torsion frames -> MP-NeRF) is structurally compatible with backpropagating a coordinate loss through Stage C; tests/pipeline/test_stageC_training_gradient_flow.py indicates gradient flow is already a concern. | https://www.nature.com/articles/s41467-023-41303-9; https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659 |
| Scope Stage D diffusion realistically and evaluate SE(3) flow matching (RNA-FrameFlow) as a lower-noise alternative to the current AF3-inspired Gaussian diffusion, targeting the regimes where generative refinement actually helps (short RNAs, conformer diversity, torsion refinement) rather than long-RNA accuracy. | RNA_PREDICT Stage D (rna_predict/pipeline/stageD/diffusion/) is AF3-inspired; the AF3-for-RNA review shows diffusion improves conformer diversity and torsion quality but does not solve long RNA. Flow matching's superiority is unverified (single source) so this is a prototype experiment with a clearly bounded value proposition, not a committed rewrite. | https://pmc.ncbi.nlm.nih.gov/articles/PMC11213149/; https://journals.iucr.org/d/issues/2025/02/00/lie5001/; https://www.nature.com/articles/s42256-026-01223-x |
| Establish a standardized external benchmark harness (RNA-Puzzles, CASP15/CASP16, and a sequence-nonredundant set) that compares the full RNA_PREDICT pipeline against RhoFold+, DRfold2, and AlphaFold3 using the PLOS leaderboard methodology, with TM-score/RMSD reporting. | Provides the coverage denominator and external ceiling needed to know whether pipeline changes are real improvements. RhoFold+ and the DRFold family are corroborated high performers; the PLOS benchmark gives an objective, reproducible comparison protocol. | https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715; https://www.nature.com/articles/s41592-024-02487-0; https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659; https://journals.iucr.org/d/issues/2025/02/00/lie5001/ |
| Keep and strengthen the Stage B Pairformer/covariance branch to ingest evolutionary information when MSAs are available, since SS priors and evolutionary covariance are the two input classes shown to give the largest independent accuracy gains. | RNA_PREDICT already has a combined torsion+Pairformer Stage B (tests/stageB/test_combined_torsion_pairformer.py). The corroborated finding that covariance is a top independent gain justifies retaining/expanding this branch for the MSA-available regime while the TorsionBERT branch covers single-sequence inference. | https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full; https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715 |
| Document and rely on MP-NeRF's polymer-general internal-to-Cartesian algorithm as the validated, high-throughput backbone of Stage C, and exploit its 400-1200x parallel speedup to make Stage C cheap enough to sit inside a training loop or ensemble-generation loop. | Stage C (rna_predict/pipeline/stageC/mp_nerf/) is a direct, corroborated implementation of MP-NeRF (two independent source URLs + matching EleutherAI code lineage). The speed property is what makes coordinate-supervised training (idea 4) and ensemble sampling (idea 9) computationally feasible. | https://pubmed.ncbi.nlm.nih.gov/34709663/; https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.26768 |
| Make conformational-ensemble output a first-class pipeline capability: emit multiple plausible 3D conformers rather than a single structure, leveraging Stage D's generative sampling. | Multiple corroborated top methods (trRosettaRNA2, RhoFold+, AF3) deliberately produce multi-conformer ensembles, indicating ensembles are a recognized value-add for RNA. RNA_PREDICT's fast Stage C plus a generative Stage D can produce ensembles cheaply, differentiating it from single-structure predictors. | https://www.nature.com/articles/s42256-026-01223-x; https://www.nature.com/articles/s41592-024-02487-0; https://journals.iucr.org/d/issues/2025/02/00/lie5001/ |

### Sources
| Title | URL | Corroboration | Note |
| --- | --- | --- | --- |
| trRosettaRNA2: Predicting RNA 3D structure and conformers using a pre-trained secondary structure model and structure-aware attention | https://www.nature.com/articles/s42256-026-01223-x | corroborated | Central claim (staging a secondary-structure prior before 3D reconstruction is a winning design) is independently supported by the PLOS Comp Biol benchmark (pcbi.1012715) and the bioRxiv inputs ablation (2025.02.14.638364), both of which find SS quality is a primary accuracy driver. CASP16 'outperforms AF3' specific result is single-source and remains unverified, but the architectural thesis is corroborated and directly mirrors RNA_PREDICT Stage A->B->C. |
| RNA-TorsionBERT: leveraging language models for RNA 3D torsion angles prediction (Bioinformatics btaf004) | https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 | corroborated | Appears twice in the pool and is corroborated by the HuggingFace model card plus direct repo usage: 'sayby/rna_torsionbert' is loaded by rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py and torsionbert_inference.py. This is the exact Stage B model; angle targets and TB-MCQ metric are authoritative for this project. |
| sayby/rna_torsionBERT Model Card on HuggingFace | https://huggingface.co/sayby/rna_torsionBERT | corroborated | Corroborates the Bioinformatics paper and the repo integration (k-mer tokenization, weights). Confirmed in-repo by Stage B loader code; authoritative for the deployed Stage B weights. |
| DeepRNA-Twist: language-model-guided RNA torsion angle prediction with attention-inception network (Briefings in Bioinformatics bbaf199) | https://academic.oup.com/bib/article/26/3/bbaf199/8124238 | uncorroborated | The directional claim (larger RNA language-model embeddings improve torsion prediction) is supported by ERNIE-RNA and the bioRxiv inputs study. However the specific result that DeepRNA-Twist surpasses RNA-TorsionBERT on RNA-Puzzles/CASP-RNA/SPOT-RNA-1D is single-source and not independently verified here; recorded as unverified. |
| RhoFold+: Accurate RNA 3D structure prediction using a language model-based deep learning approach (Nature Methods 2024) | https://www.nature.com/articles/s41592-024-02487-0 | corroborated | Independently included and benchmarked in the PLOS Comp Biol six-method study (pcbi.1012715); its strong CASP15 performance is corroborated. Serves as an external single-chain ceiling for Stage B->C output comparison. |
| DRFold: Integrating end-to-end learning with deep geometrical potentials for ab initio RNA structure prediction (Nature Communications 2023) | https://www.nature.com/articles/s41467-023-41303-9 | corroborated | Independently benchmarked by the PLOS Comp Biol study (ranks among best) and extended by DRfold2. Its local-frame-rotation + geometric-restraint decomposition is corroborated as a real, high-performing design analogous to RNA_PREDICT Stage B (frames) -> Stage C (Cartesian). |
| DRfold2 is a deep learning-based tool that enables efficient and accurate RNA structure prediction (PLOS Biology) | https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659 | uncorroborated | Existence as a DRFold successor is plausible and consistent with DRFold, but the specific 'current practical performance ceiling' claim is single-source and not independently corroborated in the pool; recorded as unverified. |
| Has AlphaFold3 achieved success for RNA? (IUCr Acta Crystallographica D, Feb 2025) | https://journals.iucr.org/d/issues/2025/02/00/lie5001/ | corroborated | AF3's RNA limitations (long RNA, non-Watson-Crick pairs) are independently echoed by the PLOS Comp Biol benchmark and the bioRxiv inputs study, which both stress SS/MSA dependence. Diffusion-on-raw-coordinates description is consistent with AF3 documentation. Directly informs scope of RNA_PREDICT Stage D (rna_predict/pipeline/stageD/diffusion/). |
| MP-NeRF: A massively parallel method for accelerating protein structure reconstruction from internal coordinates (PubMed 34709663) | https://pubmed.ncbi.nlm.nih.gov/34709663/ | corroborated | Same paper as the Wiley JCC entry (two independent URLs) and directly implemented in-repo at rna_predict/pipeline/stageC/mp_nerf/ (massive_pnerf.py, rna/). Authoritative primary citation for Stage C; protein->RNA polymer generalization confirmed by the rna/ submodule. |
| MP-NeRF (Journal of Computational Chemistry 2022; code EleutherAI/mp_nerf) | https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.26768 | corroborated | Corroborates the PubMed entry for the same paper; code lineage (EleutherAI/mp_nerf) matches the vendored Stage C implementation including protein_utils/ and rna/ subpackages. |
| ERNIE-RNA: an RNA language model with structure-enhanced representations (Nature Communications 2025) | https://www.nature.com/articles/s41467-025-64972-0 | uncorroborated | The general thesis (structure/base-pairing-aware RNA LM pretraining improves representations without MSA) aligns with DeepRNA-Twist (RiNALMo) and the bioRxiv inputs study, but ERNIE-RNA's specific zero-shot SS and downstream SOTA numbers are single-source and not independently verified here; recorded as unverified. |
| RNA-FrameFlow: Flow Matching for de novo 3D RNA Backbone Design (ICML 2024 / PMC) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11213149/ | uncorroborated | Frame-based SE(3) representation of nucleotides is corroborated conceptually by DRFold (local frames) and AF3 frames, but flow-matching's specific superiority over diffusion for RNA refinement is single-source and unverified. Offered as a candidate Stage D upgrade path, not an established fact. |
| Systematic benchmarking of deep-learning methods for tertiary RNA structure prediction (PLOS Computational Biology 2025) | https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715 | corroborated | Authoritative multi-method leaderboard (DRfold, DeepFoldRNA, RhoFold, RoseTTAFoldNA, trRosettaRNA, AF3). Its key finding that SS quality and MSA depth drive accuracy is independently corroborated by the bioRxiv inputs ablation and trRosettaRNA2. Provides the external comparison denominator for the full pipeline. |
| On inputs to deep learning for RNA 3D structure prediction (bioRxiv 2025) | https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full | corroborated | Preprint (not peer-reviewed) but its central finding -- SS priors and evolutionary covariance give the largest independent gains and sequence-only models plateau -- is corroborated by the PLOS Comp Biol benchmark and trRosettaRNA2. Directly validates RNA_PREDICT's staged SS->torsion design and the Stage B Pairformer branch. |

## Machine-checkable object
```json
{
  "goal": {
    "candidates": [
      {
        "goal": "Provide a working sequence-to-3D RNA tertiary structure inference pipeline: given an RNA sequence (or CSV of sequences) produce per-residue/atom 3D coordinates output as CSV/PDB/.pt files, via the staged TorsionBERT (Stage B) -> MP-NeRF reconstruction (Stage C) chain exposed through RNAPredictor.",
        "success_signals": [
          "Running the README-recommended command `uv run rna_predict/predict.py input_csv=... checkpoint_path=...` completes exit 0 and writes prediction_{i}.csv/.pdb/.pt and summary.csv to output_dir",
          "RNAPredictor (rna_predict/predict.py:21, re-exported via interface.py) chains Stage B torsion prediction + Stage C reconstruction and returns atom coordinates for an arbitrary input sequence without hardcoded developer paths",
          "The `rna_predict` console entry point resolves and runs instead of raising ModuleNotFoundError",
          "Inference does not depend on a developer-local checkpoint path; a published/relative checkpoint loads successfully"
        ],
        "grounding": [
          "audit/01-understanding.md:6 (provisional intent: sequence-to-structure inference producing CSV/PDB/.pt is the apparent primary deliverable; README marks Inference as Functional)",
          "README.md:7-11 (Project Status: Inference = ✅ Functional)",
          "README.md:20-44 (recommended `uv run rna_predict/predict.py` command and CSV/PDB/.pt + summary.csv output files)",
          "audit/01-understanding.md:28 (predict.py:440-543 primary README-recommended inference entry; batch_predict at :398 writes per-prediction CSV/PDB/.pt and summary.csv)",
          "audit/03-execution.md:35-36 (predict.py runs: Hydra config loads but fails at HuggingFace tokenizer download then torch.load of hardcoded /Users/tomriddle1/... checkpoint — the inference goal is implemented but currently blocked by hardcoded path/network)",
          "audit/03-execution.md:34 (`uv run rna_predict` => ModuleNotFoundError: No module named 'rna_predict.__main__'; console entry broken)",
          "audit/02-static-audit.md s2c1l0-predict-yaml-hardcoded-ckpt (predict.yaml:13 checkpoint_path hardcoded to /Users/tomriddle1/... blocks inference off the developer machine)"
        ],
        "confidence": "grounded"
      },
      {
        "goal": "Package the pipeline for the Kaggle RNA 3D folding competition: ingest the competition dataset, run prediction over the test set, and emit a validated submission CSV via a dedicated Kaggle harness.",
        "success_signals": [
          "kaggle/rna_predict.py in mode=predict runs data verification + full test-set processing and produces a submission that passes submission_validator checks",
          "Submission generation does not depend on the hardcoded /Volumes/Totallynotaharddrive/... external-drive data path",
          "interface.py demo writes submission.csv for an input sequence",
          "submission_validator raises catchable exceptions (not sys.exit) so it is usable programmatically in the submission flow"
        ],
        "grounding": [
          "audit/01-understanding.md:6 (intent includes packaging for Kaggle RNA-folding competition submissions, rna_predict/kaggle/**)",
          "audit/01-understanding.md:19 (kaggle/rna_predict.py:284-312 Kaggle harness: mode=predict runs data verification, sanity check, full test-set processing and submission validation)",
          "audit/01-understanding.md:18 (interface.py writes submission.csv)",
          "audit/02-static-audit.md s2c1l2-hardcoded-external-drive-data-path (kaggle/rna_predict.py:77 BASE_INPUT_ROOT_EXTERNAL_DRIVE = /Volumes/Totallynotaharddrive/...; integration test_real_rna_predict fails FileNotFoundError)",
          "audit/03-execution.md:31,82 (Kaggle runner external-service/hardcoded-path; integration test_real_rna_predict fails because hardcoded CSV path does not exist)",
          "audit/02-static-audit.md s2c1l0-submission-validator-sysexit (submission_validator.py:35 sys.exit terminates process instead of raising)",
          "docs/pipeline/kaggle_info/kaggle_competition.md and M2_Plan.md (Kaggle competition planning docs, audit/01-understanding.md:117-118)"
        ],
        "confidence": "grounded"
      },
      {
        "goal": "Build a full AlphaFold3-inspired multi-stage RNA structure model (Stage A 2D adjacency/RFold -> Stage B torsion via TorsionBERT+Pairformer -> Stage C MP-NeRF reconstruction -> Stage D diffusion refinement) that is end-to-end trainable on RNA structural data using PyTorch Lightning.",
        "success_signals": [
          "`python -m rna_predict.training.train` imports and runs a Lightning training loop to completion without the hardcoded /Users/tomriddle1/... config_path and without the deepspeed/torch import failure",
          "run_full_pipeline (runners/full_pipeline.py:334) executes the complete A->B->C->D chain producing output tensors",
          "Stage D diffusion components are exercised by tests (currently 0% coverage) and import without the deepspeed AttributeError",
          "Trained weights of all submodules are registered in state_dict (FeatureProcessor/AttentionComponents subclass nn.Module) and persist across checkpoint save/load"
        ],
        "grounding": [
          "audit/01-understanding.md:9 (four-stage A/B/C/D pipeline, each standalone-runnable, orchestrated by run_full_pipeline; training uses PyTorch Lightning train.py + rna_lightning_module.py)",
          "README.md:7-11 (Training = 🚧 Experimental, Diffusion = 🧪 Research Prototype — full trainable model is aspirational, not yet functional)",
          "audit/03-execution.md:38 (`python -m rna_predict.training.train --help` fails: import chain -> deepspeed AttributeError torch.library.custom_op; hardcoded @hydra.main config_path=/Users/tomriddle1/...)",
          "audit/03-execution.md:33,52 (stageD diffusion components diffusion_module.py/diffusion_conditioning.py at 0% coverage; stageB pairwise ~2-3%)",
          "audit/02-static-audit.md s2c3l0-001 (FeatureProcessor etc. are plain classes, parameters not registered in state_dict — training/checkpointing of those layers is broken)",
          "audit/03-execution.md s2c4l0-pairformer-wrapper-predict-random (pairformer_wrapper.py:399-400 predict() uses torch.randn embeddings — pairwise branch not producing learned outputs)"
        ],
        "confidence": "needs-human-confirm"
      },
      {
        "goal": "Serve as a research/experimentation platform and reference implementation for staged RNA 3D prediction methods (RFold secondary structure, TorsionBERT torsion angles, MP-NeRF forward kinematics, diffusion refinement, residue-atom bridging), documented for study and method comparison.",
        "success_signals": [
          "Each stage is independently runnable via its own Hydra @main (run_stageA/stageB/stageC/run_stageD) for isolated experimentation",
          "Extensive design/reference docs map implemented components to the literature (AF3, RFold, TorsionBERT, MP-NeRF)",
          "benchmarks/benchmark.py compares naive vs optimized input-embedding implementations and reports latency/memory"
        ],
        "grounding": [
          "audit/01-understanding.md:9 (each stage runnable standalone via its own Hydra @main: run_stageA.py:153, stageB/main.py:518, stage_c_reconstruction.py:491, run_stageD.py:392)",
          "audit/01-understanding.md:16 (benchmarks/benchmark.py:393-399 compares naive vs optimized input-embedding, measures latency/memory; referenced README.md:90)",
          "audit/01-understanding.md:143-159 (docs/reference/advanced_methods/** AF3, diffusion, LoRA, isostericity, residue_atom_bridging design specs and reference literature)",
          "audit/01-understanding.md:43 (role distribution: doc=141 — heavy documentation surface consistent with a study/reference repo)"
        ],
        "confidence": "speculative"
      },
      {
        "goal": "Maintain a heavily-tooled, CI/quality-gated Python research codebase: enforce coverage thresholds, linting/typing, mutation testing, containerization and AI-assisted task management as part of the engineering process.",
        "success_signals": [
          "`make test` (ruff + mypy + pytest --cov) passes cleanly under CI on the supported Python (>=3.10, CI 3.11)",
          "mutation testing (cosmic-ray.dev.toml / mutatest.ini) runs against a green test suite",
          "Containerfile builds a runnable image (currently FROM python:3.7-slim, incompatible with required deps)",
          "Task Master CLI (scripts/dev.js) resolves its modules and runs"
        ],
        "grounding": [
          "audit/03-execution.md:9-16 (Makefile test/lint targets, ruff/mypy, coverage xml/html; CI main.yml:50)",
          "audit/03-execution.md:24-25 (cosmic-ray.dev.toml and mutatest.ini mutation-testing configs)",
          "audit/03-execution.md:40 (`node scripts/dev.js list` fails ERR_MODULE_NOT_FOUND scripts/modules/commands.js — Task Master tooling incomplete)",
          "audit/03-execution.md s2c0l0-002 (Containerfile:1 FROM python:3.7-slim incompatible with Python>=3.10 / torch wheels)",
          "audit/01-understanding.md:9 (repo vendors a Node.js Task Master AI dev-task CLI and a screen-finder GUI app, largely unrelated to the RNA pipeline)",
          "audit/02-static-audit.md Summary (538 findings; this is a derived audit observation, not a stated repo goal — tooling intent is inferred from config artifacts)"
        ],
        "confidence": "speculative"
      }
    ]
  },
  "research": {
    "sources": [
      {
        "title": "trRosettaRNA2: Predicting RNA 3D structure and conformers using a pre-trained secondary structure model and structure-aware attention",
        "url": "https://www.nature.com/articles/s42256-026-01223-x",
        "corroboration": "corroborated",
        "note": "Central claim (staging a secondary-structure prior before 3D reconstruction is a winning design) is independently supported by the PLOS Comp Biol benchmark (pcbi.1012715) and the bioRxiv inputs ablation (2025.02.14.638364), both of which find SS quality is a primary accuracy driver. CASP16 'outperforms AF3' specific result is single-source and remains unverified, but the architectural thesis is corroborated and directly mirrors RNA_PREDICT Stage A->B->C."
      },
      {
        "title": "RNA-TorsionBERT: leveraging language models for RNA 3D torsion angles prediction (Bioinformatics btaf004)",
        "url": "https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663",
        "corroboration": "corroborated",
        "note": "Appears twice in the pool and is corroborated by the HuggingFace model card plus direct repo usage: 'sayby/rna_torsionbert' is loaded by rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py and torsionbert_inference.py. This is the exact Stage B model; angle targets and TB-MCQ metric are authoritative for this project."
      },
      {
        "title": "sayby/rna_torsionBERT Model Card on HuggingFace",
        "url": "https://huggingface.co/sayby/rna_torsionBERT",
        "corroboration": "corroborated",
        "note": "Corroborates the Bioinformatics paper and the repo integration (k-mer tokenization, weights). Confirmed in-repo by Stage B loader code; authoritative for the deployed Stage B weights."
      },
      {
        "title": "DeepRNA-Twist: language-model-guided RNA torsion angle prediction with attention-inception network (Briefings in Bioinformatics bbaf199)",
        "url": "https://academic.oup.com/bib/article/26/3/bbaf199/8124238",
        "corroboration": "uncorroborated",
        "note": "The directional claim (larger RNA language-model embeddings improve torsion prediction) is supported by ERNIE-RNA and the bioRxiv inputs study. However the specific result that DeepRNA-Twist surpasses RNA-TorsionBERT on RNA-Puzzles/CASP-RNA/SPOT-RNA-1D is single-source and not independently verified here; recorded as unverified."
      },
      {
        "title": "RhoFold+: Accurate RNA 3D structure prediction using a language model-based deep learning approach (Nature Methods 2024)",
        "url": "https://www.nature.com/articles/s41592-024-02487-0",
        "corroboration": "corroborated",
        "note": "Independently included and benchmarked in the PLOS Comp Biol six-method study (pcbi.1012715); its strong CASP15 performance is corroborated. Serves as an external single-chain ceiling for Stage B->C output comparison."
      },
      {
        "title": "DRFold: Integrating end-to-end learning with deep geometrical potentials for ab initio RNA structure prediction (Nature Communications 2023)",
        "url": "https://www.nature.com/articles/s41467-023-41303-9",
        "corroboration": "corroborated",
        "note": "Independently benchmarked by the PLOS Comp Biol study (ranks among best) and extended by DRfold2. Its local-frame-rotation + geometric-restraint decomposition is corroborated as a real, high-performing design analogous to RNA_PREDICT Stage B (frames) -> Stage C (Cartesian)."
      },
      {
        "title": "DRfold2 is a deep learning-based tool that enables efficient and accurate RNA structure prediction (PLOS Biology)",
        "url": "https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659",
        "corroboration": "uncorroborated",
        "note": "Existence as a DRFold successor is plausible and consistent with DRFold, but the specific 'current practical performance ceiling' claim is single-source and not independently corroborated in the pool; recorded as unverified."
      },
      {
        "title": "Has AlphaFold3 achieved success for RNA? (IUCr Acta Crystallographica D, Feb 2025)",
        "url": "https://journals.iucr.org/d/issues/2025/02/00/lie5001/",
        "corroboration": "corroborated",
        "note": "AF3's RNA limitations (long RNA, non-Watson-Crick pairs) are independently echoed by the PLOS Comp Biol benchmark and the bioRxiv inputs study, which both stress SS/MSA dependence. Diffusion-on-raw-coordinates description is consistent with AF3 documentation. Directly informs scope of RNA_PREDICT Stage D (rna_predict/pipeline/stageD/diffusion/)."
      },
      {
        "title": "MP-NeRF: A massively parallel method for accelerating protein structure reconstruction from internal coordinates (PubMed 34709663)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/34709663/",
        "corroboration": "corroborated",
        "note": "Same paper as the Wiley JCC entry (two independent URLs) and directly implemented in-repo at rna_predict/pipeline/stageC/mp_nerf/ (massive_pnerf.py, rna/). Authoritative primary citation for Stage C; protein->RNA polymer generalization confirmed by the rna/ submodule."
      },
      {
        "title": "MP-NeRF (Journal of Computational Chemistry 2022; code EleutherAI/mp_nerf)",
        "url": "https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.26768",
        "corroboration": "corroborated",
        "note": "Corroborates the PubMed entry for the same paper; code lineage (EleutherAI/mp_nerf) matches the vendored Stage C implementation including protein_utils/ and rna/ subpackages."
      },
      {
        "title": "ERNIE-RNA: an RNA language model with structure-enhanced representations (Nature Communications 2025)",
        "url": "https://www.nature.com/articles/s41467-025-64972-0",
        "corroboration": "uncorroborated",
        "note": "The general thesis (structure/base-pairing-aware RNA LM pretraining improves representations without MSA) aligns with DeepRNA-Twist (RiNALMo) and the bioRxiv inputs study, but ERNIE-RNA's specific zero-shot SS and downstream SOTA numbers are single-source and not independently verified here; recorded as unverified."
      },
      {
        "title": "RNA-FrameFlow: Flow Matching for de novo 3D RNA Backbone Design (ICML 2024 / PMC)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11213149/",
        "corroboration": "uncorroborated",
        "note": "Frame-based SE(3) representation of nucleotides is corroborated conceptually by DRFold (local frames) and AF3 frames, but flow-matching's specific superiority over diffusion for RNA refinement is single-source and unverified. Offered as a candidate Stage D upgrade path, not an established fact."
      },
      {
        "title": "Systematic benchmarking of deep-learning methods for tertiary RNA structure prediction (PLOS Computational Biology 2025)",
        "url": "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715",
        "corroboration": "corroborated",
        "note": "Authoritative multi-method leaderboard (DRfold, DeepFoldRNA, RhoFold, RoseTTAFoldNA, trRosettaRNA, AF3). Its key finding that SS quality and MSA depth drive accuracy is independently corroborated by the bioRxiv inputs ablation and trRosettaRNA2. Provides the external comparison denominator for the full pipeline."
      },
      {
        "title": "On inputs to deep learning for RNA 3D structure prediction (bioRxiv 2025)",
        "url": "https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full",
        "corroboration": "corroborated",
        "note": "Preprint (not peer-reviewed) but its central finding -- SS priors and evolutionary covariance give the largest independent gains and sequence-only models plateau -- is corroborated by the PLOS Comp Biol benchmark and trRosettaRNA2. Directly validates RNA_PREDICT's staged SS->torsion design and the Stage B Pairformer branch."
      }
    ],
    "ideas": [
      {
        "idea": "Treat Stage A secondary-structure quality as the primary tunable lever for end-to-end accuracy, since multiple independent benchmarks identify SS quality (not the 3D module) as the dominant accuracy driver; invest in a stronger/pretrained SS prior feeding Stage B rather than scaling the reconstruction module.",
        "relevance": "RNA_PREDICT is already staged (Stage A SS -> Stage B torsion -> Stage C MP-NeRF). Corroborated evidence (3 sources) that SS quality is the bottleneck means improving Stage A yields the highest marginal return and validates the existing topology over a sequence-only design.",
        "sources": [
          "https://www.nature.com/articles/s42256-026-01223-x",
          "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715",
          "https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full"
        ]
      },
      {
        "idea": "Evaluate replacing or augmenting the Stage B TorsionBERT encoder with a larger structure-aware RNA language model (e.g. RiNALMo as used by DeepRNA-Twist, or ERNIE-RNA's base-pairing-aware embeddings) to improve torsion-angle quality while preserving the project's MSA-free, single-sequence inference design.",
        "relevance": "Stage B (rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py) is the upstream determinant of Stage C coordinates. The directional benefit of larger structure-aware RNA LMs is corroborated, though each tool's specific SOTA-over-TorsionBERT claim is unverified -- so this is a research experiment to A/B, not a guaranteed win.",
        "sources": [
          "https://academic.oup.com/bib/article/26/3/bbaf199/8124238",
          "https://www.nature.com/articles/s41467-025-64972-0",
          "https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full"
        ]
      },
      {
        "idea": "Adopt the TorsionBERT-derived TB-MCQ torsion-quality scoring function as both a Stage B training signal and an intrinsic pipeline evaluation metric, giving a coordinate-free way to score torsion outputs before Stage C reconstruction.",
        "relevance": "TB-MCQ comes from the exact model RNA_PREDICT loads ('sayby/rna_torsionbert'), so it is directly applicable to the deployed Stage B and to gating/regression-testing torsion quality independent of the (lossy) MP-NeRF step.",
        "sources": [
          "https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663",
          "https://huggingface.co/sayby/rna_torsionBERT"
        ]
      },
      {
        "idea": "Add end-to-end coordinate supervision plus deep geometric restraints (inter-nucleotide distances/orientations) across the Stage B->C boundary, following the DRFold family, so Stage B is trained against final 3D error rather than torsion MAE alone.",
        "relevance": "DRFold attributes its large TM-score gains specifically to coordinate-supervised end-to-end learning combined with composite geometric potentials. RNA_PREDICT's frame->Cartesian decomposition (torsion frames -> MP-NeRF) is structurally compatible with backpropagating a coordinate loss through Stage C; tests/pipeline/test_stageC_training_gradient_flow.py indicates gradient flow is already a concern.",
        "sources": [
          "https://www.nature.com/articles/s41467-023-41303-9",
          "https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659"
        ]
      },
      {
        "idea": "Scope Stage D diffusion realistically and evaluate SE(3) flow matching (RNA-FrameFlow) as a lower-noise alternative to the current AF3-inspired Gaussian diffusion, targeting the regimes where generative refinement actually helps (short RNAs, conformer diversity, torsion refinement) rather than long-RNA accuracy.",
        "relevance": "RNA_PREDICT Stage D (rna_predict/pipeline/stageD/diffusion/) is AF3-inspired; the AF3-for-RNA review shows diffusion improves conformer diversity and torsion quality but does not solve long RNA. Flow matching's superiority is unverified (single source) so this is a prototype experiment with a clearly bounded value proposition, not a committed rewrite.",
        "sources": [
          "https://pmc.ncbi.nlm.nih.gov/articles/PMC11213149/",
          "https://journals.iucr.org/d/issues/2025/02/00/lie5001/",
          "https://www.nature.com/articles/s42256-026-01223-x"
        ]
      },
      {
        "idea": "Establish a standardized external benchmark harness (RNA-Puzzles, CASP15/CASP16, and a sequence-nonredundant set) that compares the full RNA_PREDICT pipeline against RhoFold+, DRfold2, and AlphaFold3 using the PLOS leaderboard methodology, with TM-score/RMSD reporting.",
        "relevance": "Provides the coverage denominator and external ceiling needed to know whether pipeline changes are real improvements. RhoFold+ and the DRFold family are corroborated high performers; the PLOS benchmark gives an objective, reproducible comparison protocol.",
        "sources": [
          "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715",
          "https://www.nature.com/articles/s41592-024-02487-0",
          "https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003659",
          "https://journals.iucr.org/d/issues/2025/02/00/lie5001/"
        ]
      },
      {
        "idea": "Keep and strengthen the Stage B Pairformer/covariance branch to ingest evolutionary information when MSAs are available, since SS priors and evolutionary covariance are the two input classes shown to give the largest independent accuracy gains.",
        "relevance": "RNA_PREDICT already has a combined torsion+Pairformer Stage B (tests/stageB/test_combined_torsion_pairformer.py). The corroborated finding that covariance is a top independent gain justifies retaining/expanding this branch for the MSA-available regime while the TorsionBERT branch covers single-sequence inference.",
        "sources": [
          "https://www.biorxiv.org/content/10.1101/2025.02.14.638364v1.full",
          "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012715"
        ]
      },
      {
        "idea": "Document and rely on MP-NeRF's polymer-general internal-to-Cartesian algorithm as the validated, high-throughput backbone of Stage C, and exploit its 400-1200x parallel speedup to make Stage C cheap enough to sit inside a training loop or ensemble-generation loop.",
        "relevance": "Stage C (rna_predict/pipeline/stageC/mp_nerf/) is a direct, corroborated implementation of MP-NeRF (two independent source URLs + matching EleutherAI code lineage). The speed property is what makes coordinate-supervised training (idea 4) and ensemble sampling (idea 9) computationally feasible.",
        "sources": [
          "https://pubmed.ncbi.nlm.nih.gov/34709663/",
          "https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.26768"
        ]
      },
      {
        "idea": "Make conformational-ensemble output a first-class pipeline capability: emit multiple plausible 3D conformers rather than a single structure, leveraging Stage D's generative sampling.",
        "relevance": "Multiple corroborated top methods (trRosettaRNA2, RhoFold+, AF3) deliberately produce multi-conformer ensembles, indicating ensembles are a recognized value-add for RNA. RNA_PREDICT's fast Stage C plus a generative Stage D can produce ensembles cheaply, differentiating it from single-structure predictors.",
        "sources": [
          "https://www.nature.com/articles/s42256-026-01223-x",
          "https://www.nature.com/articles/s41592-024-02487-0",
          "https://journals.iucr.org/d/issues/2025/02/00/lie5001/"
        ]
      }
    ]
  }
}
```
