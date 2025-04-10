# Progressive Test Coverage System

This document explains the progressive test coverage system implemented in `rna_predict/scripts/run_failing_tests.sh`, which aligns test coverage goals with the Kaggle competition timeline.

## Overview

The progressive test coverage system:

1. Automatically adjusts coverage requirements based on the current Kaggle competition phase
2. Implements a gradual day-by-day increase in coverage goals within each phase
3. Sets higher standards for critical modules compared to utility modules
4. Provides clear visibility into upcoming milestones and coverage goals
5. Supports both overall and module-specific coverage testing

## Kaggle Competition Timeline

The system is aligned with the Stanford RNA 3D Folding Competition timeline:

| Phase | Date Range | Description |
|-------|------------|-------------|
| **Exploration** | Feb 27 - Mar 27, 2025 | Initial exploration and model development |
| **Development** | Mar 28 - Apr 22, 2025 | Core implementation and refinement |
| **Optimization** | Apr 23 - May 15, 2025 | Performance tuning after leaderboard refresh |
| **Final Submission** | May 16 - May 29, 2025 | Final preparations for submission |

## Coverage Goals by Phase

The coverage goals increase progressively through each phase, with a gradual day-by-day increase within each phase:

| Phase | Starting Coverage | Target Coverage | Critical Modules | Standard Modules | Utility Modules |
|-------|------------------|-----------------|------------------|------------------|----------------|
| Exploration | 80% | 80% | 85% | 75% | 70% |
| Development | 80% | 85% | 90% | 85% | 75% |
| Optimization | 85% | 90% | 95% | 90% | 80% |
| Final Submission | 90% | 95% | 98% | 95% | 85% |

Rather than jumping immediately to the target coverage when a new phase begins, the system calculates a daily incremental increase based on:

1. The current day within the phase
2. The total duration of the phase
3. The starting coverage and target coverage for the phase

This ensures a smooth, gradual progression that gives the team time to adapt and improve test coverage at a sustainable pace.

## Module Categories

Modules are categorized based on their importance to the RNA prediction pipeline:

### Critical Modules
- `rna_predict.pipeline.stageB` - Torsion angle prediction (TorsionBERT)
- `rna_predict.pipeline.stageD.diffusion` - Diffusion-based refinement

### Standard Modules
- `rna_predict.pipeline.stageA` - RNA 2D structure prediction
- `rna_predict.pipeline.stageC` - Forward kinematics
- `rna_predict.pipeline.stageD.tensor_fixes` - Shape handling utilities

### Utility Modules
- `rna_predict.utils` - General utilities
- `rna_predict.scripts` - Scripts and tools
- `rna_predict.dataset` - Dataset handling

## Usage

### Standard Usage

Run all tests with the phase-appropriate overall coverage goal:

```bash
./rna_predict/scripts/run_failing_tests.sh
```

### Module-Specific Testing

Run tests with different coverage thresholds for each module category:

```bash
./rna_predict/scripts/run_failing_tests.sh --module-specific
```

## Configuration

The system uses a JSON configuration file (`.coverage_config.json`) to store:

1. Phase dates and coverage goals
2. Module categorization
3. Current coverage status and base coverage

This file is automatically created if it doesn't exist, but can be manually edited to adjust goals or module categorization.

## Requirements

- `jq` command-line JSON processor (for parsing the configuration)
- `bc` command-line calculator (for calculating coverage goals)
- `pytest` with the following plugins:
  - `pytest-xdist` (for parallel testing)
  - `pytest-cov` (for coverage reporting)
  - `pytest-memray` (for memory profiling)
  - `pytest-timeout` (for test timeouts)

## Benefits for Kaggle Competition

1. **Focus on Critical Components**: Higher coverage for core prediction modules ensures reliability where it matters most
2. **Gradual Progression**: Realistic coverage goals that increase at a sustainable pace
3. **Milestone Awareness**: Keeps the team aware of upcoming competition phases and deadlines
4. **Adaptability**: Can be adjusted if competition dates change or if you need to prioritize different modules

## Implementation Details

The script:

1. Determines the current competition phase based on the date
2. Calculates the appropriate coverage goal based on days into the phase
3. Applies phase transition smoothing to prevent abrupt jumps between phases
4. Caps daily coverage increases to ensure sustainable progress
5. Shows days remaining until the next milestone
6. Runs tests with the appropriate coverage threshold
7. Generates detailed coverage reports
8. Updates the stored coverage goal and last run date for the next run

For module-specific testing, it applies different thresholds to different module categories based on their criticality to the competition goals.

### Advanced Features

#### Phase Transition Smoothing

When transitioning between phases, the system checks if:
1. We're within the first 3 days of a new phase
2. The current coverage is significantly below the target of the previous phase

If both conditions are met, it adjusts the starting point to be closer to the actual coverage, preventing unrealistic jumps in requirements.

#### Maximum Daily Increase Cap

To prevent large jumps in coverage requirements after periods of inactivity:
1. The system tracks the last time tests were run
2. It caps the maximum daily increase to 0.5% per day
3. For longer gaps between runs, it allows a proportional increase (e.g., 1.5% after 3 days)

#### Configuration Backup and Recovery

To prevent data loss and ensure continuity:
1. The system creates a backup of the configuration file before each run
2. It validates the JSON structure and restores from backup if corrupted
3. It falls back to creating a new configuration if no valid backup exists
