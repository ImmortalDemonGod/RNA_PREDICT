# Pull Request #72: 📝 Add docstrings to `feat/phase1-step2-direct-angle-loss`

## Status
- State: MERGED
- Created: 2025-05-05
- Updated: 2025-05-06
- Closed: 2025-05-06
- Merged: 2025-05-06

## Changes
- Additions: 1031
- Deletions: 328
- Changed Files: 44

## Author
- Name: N/A
- Login: app/coderabbitai
- Bot: Yes

## Assignees
- ImmortalDemonGod

## Description
Docstrings generation was requested by @ImmortalDemonGod.

* https://github.com/ImmortalDemonGod/RNA_PREDICT/pull/71#issuecomment-2852448410

The following files were modified:

* `rna_predict/dataset/collate.py`
* `rna_predict/dataset/dataset_loader.py`
* `rna_predict/dataset/loader.py`
* `rna_predict/dataset/preprocessing/angle_utils.py`
* `rna_predict/dataset/preprocessing/angles.py`
* `rna_predict/dataset/preprocessing/compute_ground_truth_angles.py`
* `rna_predict/dataset/tmp_tests/test_angle_extraction_perf.py`
* `rna_predict/dataset/tmp_tests/test_compute_ground_truth_angles.py`
* `rna_predict/dataset/tmp_tests/test_dssr_installation.py`
* `rna_predict/dataset/tmp_tests/test_extract_angles.py`
* `rna_predict/dataset/tmp_tests/test_train_with_angle_supervision.py`
* `rna_predict/pipeline/merger/simple_latent_merger.py`
* `rna_predict/pipeline/stageA/adjacency/RFold_code.py`
* `rna_predict/pipeline/stageA/adjacency/rfold_predictor.py`
* `rna_predict/pipeline/stageA/run_stageA.py`
* `rna_predict/pipeline/stageB/main.py`
* `rna_predict/pipeline/stageB/pairwise/dummy_pairformer.py`
* `rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py`
* `rna_predict/pipeline/stageB/torsion/dummy_torsion_model.py`
* `rna_predict/pipeline/stageB/torsion/torsion_bert_predictor.py`
* `rna_predict/pipeline/stageB/torsion/torsionbert_inference.py`
* `rna_predict/pipeline/stageC/stage_c_reconstruction.py`
* `rna__atom_bridge.py`
* `rna__utils.py`
* `rna_predict/pipeline/stageD/diffusion/generator.py`
* `rna_predict/pipeline/stageD/diffusion/protenix_diffusion_manager.py`
* `rna_predict/pipeline/stageD/run_stageD.py`
* `rna_predict/pipeline/stageD/stage_d_utils/feature_utils.py`
* `rna_predict/predict.py`
* `rna_predict/training/rna_lightning_module.py`
* `rna_predict/training/train.py`
* `scripts/test_utils/batch_test_generator.py`
* `tests/integration/test_pipeline_dimensions.py`
* `tests/stageB/test_main_comprehensive.py`
* `tests/stageB/torsion/test_stageB_torsionbert_predictor_comprehensive.py`
* `tests/stageB/torsion/test_torsionbert.py`
* `tests/stageC/test_stage_c_reconstruction.py`
* `tests/stageC/test_stage_c_reconstruction_comprehensive.py`
* `tests/stageD/unit/diffusion/test_generator.py`
* `tests/test_args_namespace.py`
* `tests/test_config.py`
* `tests/test_debug_logging.py`
* `tests/test_dnabert_mps.py`
* `tests/test_rfold_model.py`


These files were kept as they were

* `tests/common/test_batch_test_generator.py`
* `tests/test_rfold_model_fix.py`




These file types are not supported

* `rna_predict/conf/default.yaml`
* `rna_predict/conf/model/stageA.yaml`
* `rna_predict/conf/model/stageB_pairformer.yaml`
* `rna_predict/conf/model/stageB_torsion.yaml`
* `rna_predict/conf/model/stageC.yaml`
* `rna_predict/conf/model/stageD.yaml`
* `rna_predict/conf/model/stageD_diffusion.yaml`




ℹ️ Note

CodeRabbit cannot perform edits on its own pull requests yet.

## Comments

### Comment by coderabbitai
- Created: 2025-05-05
- Author Association: NONE

> [!IMPORTANT]
> ## Review skipped
> 
> Bot user detected.
> 
> To trigger a single review, invoke the `@coderabbitai review` command.
> 
> You can disable this status message by setting the `reviews.review_status` to `false` in the CodeRabbit configuration file.

---


🪧 Tips

### Chat

There are 3 ways to chat with [CodeRabbit](https://coderabbit.ai?utm_source=oss&utm_medium=github&utm_campaign=ImmortalDemonGod/RNA_PREDICT&utm_content=72):

- Review comments: Directly reply to a review comment made by CodeRabbit. Example:
  - `I pushed a fix in commit , please review it.`
  - `Generate unit testing code for this file.`
  - `Open a follow-up GitHub issue for this discussion.`
- Files and specific lines of code (under the "Files changed" tab): Tag `@coderabbitai` in a new review comment at the desired location with your query. Examples:
  - `@coderabbitai generate unit testing code for this file.`
  -	`@coderabbitai modularize this function.`
- PR comments: Tag `@coderabbitai` in a new PR comment to ask questions about the PR branch. For the best results, please provide a very specific query, as very limited context is provided in this mode. Examples:
  - `@coderabbitai gather interesting stats about this repository and render them as a table. Additionally, render a pie chart showing the language distribution in the codebase.`
  - `@coderabbitai read src/utils.ts and generate unit testing code.`
  - `@coderabbitai read the files in the src/scheduler package and generate a class diagram using mermaid and a README in the markdown format.`
  - `@coderabbitai help me debug CodeRabbit configuration file.`

### Support

Need help? Join our [Discord community](https://discord.gg/coderabbit) for assistance with any issues or questions.

Note: Be mindful of the bot's finite context window. It's strongly recommended to break down tasks such as reading entire modules into smaller chunks. For a focused discussion, use review comments to chat about specific files and their changes, instead of using the PR comments.

### CodeRabbit Commands (Invoked using PR comments)

- `@coderabbitai pause` to pause the reviews on a PR.
- `@coderabbitai resume` to resume the paused reviews.
- `@coderabbitai review` to trigger an incremental review. This is useful when automatic reviews are disabled for the repository.
- `@coderabbitai full review` to do a full review from scratch and review all the files again.
- `@coderabbitai summary` to regenerate the summary of the PR.
- `@coderabbitai generate sequence diagram` to generate a sequence diagram of the changes in this PR.
- `@coderabbitai resolve` resolve all the CodeRabbit review comments.
- `@coderabbitai configuration` to show the current CodeRabbit configuration for the repository.
- `@coderabbitai help` to get help.

### Other keywords and placeholders

- Add `@coderabbitai ignore` anywhere in the PR description to prevent this PR from being reviewed.
- Add `@coderabbitai summary` to generate the high-level summary at a specific location in the PR description.
- Add `@coderabbitai` anywhere in the PR title to generate the title automatically.

### CodeRabbit Configuration File (`.coderabbit.yaml`)

- You can programmatically configure CodeRabbit by adding a `.coderabbit.yaml` file to the root of your repository.
- Please see the [configuration documentation](https://docs.coderabbit.ai/guides/configure-coderabbit) for more information.
- If your editor has YAML language server enabled, you can add the path at the top of this file to enable auto-completion and validation: `# yaml-language-server: $schema=https://coderabbit.ai/integrations/schema.v2.json`

### Documentation and Community

- Visit our [Documentation](https://docs.coderabbit.ai) for detailed information on how to use CodeRabbit.
- Join our [Discord Community](http://discord.gg/coderabbit) to get help, request features, and share feedback.
- Follow us on [X/Twitter](https://twitter.com/coderabbitai) for updates and announcements.

---
