#!/usr/bin/env bash
###############################################################################
# reorg_tests.sh
#
# Reorganize the test files into a single coherent structure:
#   tests/common/
#   tests/integration/
#   tests/performance/
#   tests/e2e/
#   tests/stageA/{unit, integration, e2e}
#   tests/stageB/{unit, integration, e2e}
#   tests/stageC/{unit, integration, e2e}
#   tests/stageD/{unit, integration, e2e}
#
# This script is idempotent; run it multiple times as needed.
# Some 'mv' commands won't apply if files have already been moved.
#
# Usage:
#   bash scripts/reorg_tests.sh
###############################################################################

set -euo pipefail

TESTS_DIR="tests"

echo "Ensuring final directory structure..."
mkdir -p "${TESTS_DIR}/common"
mkdir -p "${TESTS_DIR}/integration"
mkdir -p "${TESTS_DIR}/performance"
mkdir -p "${TESTS_DIR}/e2e"

for stg in stageA stageB stageC stageD; do
  mkdir -p "${TESTS_DIR}/${stg}/unit"
  mkdir -p "${TESTS_DIR}/${stg}/integration"
  mkdir -p "${TESTS_DIR}/${stg}/e2e"
done

echo "Moving top-level tests into final structure..."

###############################################################################
# Move 'common' tests (truly stage-agnostic or developer scripts) to tests/common
###############################################################################
# If the 'common' directory already exists with tests inside it, we'll keep it as is.
# We'll move any leftover "test_*.py" that belong in common or script utilities.

# Directly move items from tests/common if they're in the wrong place:
# (No direct rename needed if they're already in tests/common, but we handle leftover top-level files.)

# Move test_dummy, test_batch_test_generator, etc. to tests/common
if [ -f "${TESTS_DIR}/test_dummy.py" ]; then
  mv "${TESTS_DIR}/test_dummy.py" "${TESTS_DIR}/common/"
fi
if [ -f "${TESTS_DIR}/test_batch_test_generator.py" ]; then
  mv "${TESTS_DIR}/test_batch_test_generator.py" "${TESTS_DIR}/common/"
fi
if [ -f "${TESTS_DIR}/test_fix_leading_zeros.py" ]; then
  mv "${TESTS_DIR}/test_fix_leading_zeros.py" "${TESTS_DIR}/common/"
fi
if [ -f "${TESTS_DIR}/test_remove_logger_lines.py" ]; then
  mv "${TESTS_DIR}/test_remove_logger_lines.py" "${TESTS_DIR}/common/"
fi
if [ -f "${TESTS_DIR}/test_scatter_utils.py" ]; then
  mv "${TESTS_DIR}/test_scatter_utils.py" "${TESTS_DIR}/common/"
fi
if [ -f "${TESTS_DIR}/test_input_embd_utils.py" ]; then
  mv "${TESTS_DIR}/test_input_embd_utils.py" "${TESTS_DIR}/common/"
fi
if [ -f "${TESTS_DIR}/test_ml_utils_consolidated.py" ]; then
  mv "${TESTS_DIR}/test_ml_utils_consolidated.py" "${TESTS_DIR}/common/"
fi

###############################################################################
# Move integration tests (cross-stage) to tests/integration
###############################################################################
if [ -f "${TESTS_DIR}/test_main_integration.py" ]; then
  mv "${TESTS_DIR}/test_main_integration.py" "${TESTS_DIR}/integration/"
fi
if [ -f "${TESTS_DIR}/test_pipeline_integration.py" ]; then
  mv "${TESTS_DIR}/test_pipeline_integration.py" "${TESTS_DIR}/integration/"
fi

###############################################################################
# Move performance tests to tests/performance
###############################################################################
if [ -f "${TESTS_DIR}/test_benchmark.py" ]; then
  mv "${TESTS_DIR}/test_benchmark.py" "${TESTS_DIR}/performance/"
fi
if [ -f "${TESTS_DIR}/test_benchmark_suite.py" ]; then
  mv "${TESTS_DIR}/test_benchmark_suite.py" "${TESTS_DIR}/performance/"
fi
if [ -f "${TESTS_DIR}/test_performance.py" ]; then
  mv "${TESTS_DIR}/test_performance.py" "${TESTS_DIR}/performance/"
fi

###############################################################################
# Move e2e tests to tests/e2e (or stageX/e2e if specific)
###############################################################################
if [ -f "${TESTS_DIR}/test_stageD_demo.py" ]; then
  mv "${TESTS_DIR}/test_stageD_demo.py" "${TESTS_DIR}/stageD/e2e/"
fi
if [ -f "${TESTS_DIR}/test_stageD_diffusion.py" ]; then
  mv "${TESTS_DIR}/test_stageD_diffusion.py" "${TESTS_DIR}/stageD/e2e/"
fi
# If there's a test_end_to_end_stageA_to_D.py or similar:
if [ -f "${TESTS_DIR}/test_end_to_end_stageA_to_D.py" ]; then
  mv "${TESTS_DIR}/test_end_to_end_stageA_to_D.py" "${TESTS_DIR}/e2e/"
fi

###############################################################################
# Move leftover mp_nerf_tests => stageC/unit
###############################################################################
if [ -d "${TESTS_DIR}/mp_nerf_tests" ]; then
  if [ -f "${TESTS_DIR}/mp_nerf_tests/test_main.py" ]; then
    mv "${TESTS_DIR}/mp_nerf_tests/test_main.py" "${TESTS_DIR}/stageC/unit/"
  fi
  if [ -f "${TESTS_DIR}/mp_nerf_tests/test_ml_utils.py" ]; then
    mv "${TESTS_DIR}/mp_nerf_tests/test_ml_utils.py" "${TESTS_DIR}/stageC/unit/"
  fi
  rmdir --ignore-fail-on-non-empty "${TESTS_DIR}/mp_nerf_tests" || true
fi

###############################################################################
# Move stageA tests
###############################################################################
# Already found that there's tests/stageA with subfolders. We'll just place leftover "test_stageA*.py" there
if [ -f "${TESTS_DIR}/test_stageA.py" ]; then
  mv "${TESTS_DIR}/test_stageA.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_RFold_code.py" ]; then
  mv "${TESTS_DIR}/test_RFold_code.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_ref_space_uid_patch.py" ]; then
  mv "${TESTS_DIR}/test_ref_space_uid_patch.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_refactored_RFold_code.py" ]; then
  mv "${TESTS_DIR}/test_refactored_RFold_code.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_run_stageA.py" ]; then
  mv "${TESTS_DIR}/test_run_stageA.py" "${TESTS_DIR}/stageA/integration/"
fi
if [ -f "${TESTS_DIR}/test_token_feature_shape.py" ]; then
  mv "${TESTS_DIR}/test_token_feature_shape.py" "${TESTS_DIR}/stageA/unit/"
fi

###############################################################################
# Move stageB tests
###############################################################################
if [ -f "${TESTS_DIR}/test_stageB_torsionbert.py" ]; then
  mv "${TESTS_DIR}/test_stageB_torsionbert.py" "${TESTS_DIR}/stageB/unit/"
fi

###############################################################################
# Move stageC tests
###############################################################################
if [ -f "${TESTS_DIR}/test_final_kb_rna.py" ]; then
  mv "${TESTS_DIR}/test_final_kb_rna.py" "${TESTS_DIR}/stageC/unit/"
fi
if [ -f "${TESTS_DIR}/test_kb_proteins.py" ]; then
  mv "${TESTS_DIR}/test_kb_proteins.py" "${TESTS_DIR}/stageC/unit/"
fi
if [ -f "${TESTS_DIR}/test_rna_refactored.py" ]; then
  mv "${TESTS_DIR}/test_rna_refactored.py" "${TESTS_DIR}/stageC/unit/"
fi

###############################################################################
# Move stageD tests
###############################################################################
if [ -f "${TESTS_DIR}/test_run_stageD_diffusion.py" ]; then
  mv "${TESTS_DIR}/test_run_stageD_diffusion.py" "${TESTS_DIR}/stageD/integration/"
fi
if [ -f "${TESTS_DIR}/test_atom_trunk_small_natom.py" ]; then
  mv "${TESTS_DIR}/test_atom_trunk_small_natom.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_generator.py" ]; then
  mv "${TESTS_DIR}/test_generator.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_protenix_diffusion_manager.py" ]; then
  mv "${TESTS_DIR}/test_protenix_diffusion_manager.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_protenix_integration.py" ]; then
  mv "${TESTS_DIR}/test_protenix_integration.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_shape_mismatch.py" ]; then
  mv "${TESTS_DIR}/test_shape_mismatch.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_shape_mismatch_atom_tokens.py" ]; then
  mv "${TESTS_DIR}/test_shape_mismatch_atom_tokens.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_single_sample_expansion.py" ]; then
  mv "${TESTS_DIR}/test_single_sample_expansion.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_single_sample_shape_expansion.py" ]; then
  mv "${TESTS_DIR}/test_single_sample_shape_expansion.py" "${TESTS_DIR}/stageD/unit/"
fi
if [ -f "${TESTS_DIR}/test_stageD_shape_mismatch.py" ]; then
  mv "${TESTS_DIR}/test_stageD_shape_mismatch.py" "${TESTS_DIR}/stageD/unit/"
fi

###############################################################################
# Additional leftover test files that might be in the top-level tests folder
###############################################################################
# e.g. test_atom_attention_encoder.py, test_atom_encoder.py, etc.
# We'll guess they belong to stageA or stageC. You can adjust as needed.

if [ -f "${TESTS_DIR}/test_atom_attention_encoder.py" ]; then
  # Possibly stageA or stageD? Let's assume stageA
  mv "${TESTS_DIR}/test_atom_attention_encoder.py" "${TESTS_DIR}/stageA/unit/"
fi

if [ -f "${TESTS_DIR}/test_atom_encoder.py" ]; then
  # If it references pipeline/stageA code or illusions. We'll guess stageA.
  mv "${TESTS_DIR}/test_atom_encoder.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_atom_transformer.py" ]; then
  mv "${TESTS_DIR}/test_atom_transformer.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_attention_and_checkpointing.py" ]; then
  mv "${TESTS_DIR}/test_attention_and_checkpointing.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_block_sparse.py" ]; then
  mv "${TESTS_DIR}/test_block_sparse.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_embedders.py" ]; then
  mv "${TESTS_DIR}/test_embedders.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_transformer.py" ]; then
  mv "${TESTS_DIR}/test_transformer.py" "${TESTS_DIR}/stageA/unit/"
fi
if [ -f "${TESTS_DIR}/test_dataset_loader.py" ]; then
  mv "${TESTS_DIR}/test_dataset_loader.py" "${TESTS_DIR}/common/"
fi

# Clean up any leftover empty directories (like 'unit/', etc.)
echo "Cleaning up any empty or leftover directories..."

# We'll attempt to remove the old 'unit' folder if empty
rmdir --ignore-fail-on-non-empty "${TESTS_DIR}/unit" 2>/dev/null || true
rmdir --ignore-fail-on-non-empty "${TESTS_DIR}/mp_nerf_tests" 2>/dev/null || true

# Also remove any __pycache__ folders
find "${TESTS_DIR}" -type d -name "__pycache__" -exec rm -rf {} +

echo "Reorganization complete! The tests should now be organized as follows:

tests/
  ├─ common/
  ├─ e2e/
  ├─ integration/
  ├─ performance/
  ├─ stageA/ (unit, integration, e2e)
  ├─ stageB/ (unit, integration, e2e)
  ├─ stageC/ (unit, integration, e2e)
  └─ stageD/ (unit, integration, e2e)

Enjoy a cleaner test suite!"