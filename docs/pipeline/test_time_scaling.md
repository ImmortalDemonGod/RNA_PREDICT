Test Time Scaling
=================

This document outlines the test time scaling for the RNA 3D prediction pipeline.

Test Cases:
-----------
    • Small RNA (20-30 nucleotides)
    • Medium RNA (50-100 nucleotides)
    • Large RNA (150-200 nucleotides)
    • Very Large RNA (300+ nucleotides)

Metrics to Measure:
------------------
    • Total pipeline runtime
    • Time per major stage
    • Memory usage peaks
    • Disk space requirements
    • CPU/GPU utilization

Testing Protocol:
----------------
    1. Run each test case 3 times to get average metrics
    2. Record system specifications:
        • CPU model and cores
        • RAM capacity
        • GPU model (if applicable)
        • Storage type (SSD/HDD)
    3. Monitor resource usage during runs
    4. Document any bottlenecks or failures

Expected Scaling:
----------------
    • Runtime should scale approximately O(n²) with sequence length
    • Memory usage expected to scale linearly O(n)
    • Storage requirements scale linearly with sequence length

Optimization Targets:
-------------------
    • Identify stages that don't scale well
    • Find opportunities for parallelization
    • Reduce memory footprint where possible
    • Optimize I/O operations 