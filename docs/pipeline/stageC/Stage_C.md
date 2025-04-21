🧙🏾‍♂️ **Integrated Stage C: Forward Kinematics Comprehensive Guide (RNA Pipeline) 🚀**

---

### 🧬 Motivation and Key Concepts

**Why Forward Kinematics?**

- RNA structure is best represented using internal coordinates: bond lengths, bond angles, and torsion angles (**α, β, γ, δ, ε, ζ, χ, sugar pucker**).
- Predicting torsion angles (**Stage B**) is more straightforward and biologically relevant compared to directly predicting Cartesian coordinates.
- Forward Kinematics (**FK**) translates these angles into accurate 3D atom positions, ensuring physically consistent and valid structures.

**🔄 Invariance & Efficiency**

- Torsion angles are rotation/translation-invariant, thus RNA’s intrinsic “folding instructions.”
- FK efficiently manages geometric constraints through sequential rotations around bonds with fixed reference geometry.

**📌 Core Steps**

1. Initialize the first residue in a canonical orientation.
2. For each subsequent residue, apply torsion angles (from Stage B) using known bond geometry.
3. Position atoms through rotations around bond axes.
4. Explicitly manage sugar puckering variations (C3′-endo, pseudorotation).
5. Output robust 3D coordinates for each residue’s heavy atoms (optionally including base atoms).

---

### 📚 Extended Theoretical Foundations

#### 🛠️ Kinematic Analogy (Robotics & RNA)

- RNA backbone analogous to robotic joint chains: torsion angles represent joints.
- Sequential rotations reconstruct RNA's 3D backbone using known bond lengths and angles.

#### 📐 Mathematics: Rotation Matrices & Homogeneous Transformations

**Rotation Matrices:** (SO(3), orthonormal)

\[
R_z(\theta) = \begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

**Homogeneous Transformations (4×4):** Efficient combination of rotations and translations.

\[
T = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\]

- Enables chained transformations along the RNA backbone (similar to Denavit–Hartenberg method in robotics).

#### 📏 Numerical Stability & Reference Geometry

- Small numerical errors accumulate through sequential transformations. Regular orthonormalization recommended.
- Natural Extension Reference Frame (**NeRF**) method strongly recommended for numerical stability and efficiency.
- Standard RNA geometry from parameter sets (**AMBER, 3DNA/DSSR**).
- Flexible modeling of sugar puckers (C3′-endo, predicted pseudorotation, or refined via minimization).

**Key References:**
- Richardson et al. (2008), Murray et al. (2003), 3DNA/DSSR documentation.

---

### 📈 Data Flow in RNA Multi-Stage Pipeline

1. **Stage A:** Extract 2D adjacency/base pairs from raw sequence.
2. **Stage B:** Predict torsion angles (**α, β, γ, δ, ε, ζ, χ**).
3. **Stage C (Current)**:
   - **Input:** Predicted torsion angles, reference geometry.
   - **Output:** 3D atom coordinates (heavy atoms).
4. **Stage D (Optional):** AF3-like or diffusion-based refinement in Cartesian or angle space.

---

### 💻 Detailed Pseudocode Implementation

```python
def forward_kinematics(torsion_angles, sequence, reference_geometry, ring_pucker_model=None):
    N = len(sequence)
    coords = alloc_coord_array(N)

    coords[0] = place_first_residue(torsion_angles[0], sequence[0], reference_geometry)

    for i in range(1, N):
        anchor_positions = get_anchor_positions(coords[i-1], sequence[i-1])
        alpha, beta, gamma, delta, epsilon, zeta, chi = torsion_angles[i]

        coords[i] = build_residue(
            anchor_positions,
            (alpha, beta, gamma, delta, epsilon, zeta, chi),
            sequence[i],
            reference_geometry
        )

        if ring_pucker_model is not None:
            coords[i] = refine_sugar_pucker(coords[i], ring_pucker_model[i])

    coords = final_refinement(coords)
    return coords
```

#### 🔎 Step-by-Step Local Frame Construction (Detailed)

- **Local Frame Construction**:
    - Identify three previously placed atoms (A, B, C).
    - Define local axes at atom C:
        ```
        x_axis = normalize(B - C)
        temp   = normalize(A - B)
        z_axis = normalize(cross(x_axis, temp))
        y_axis = cross(z_axis, x_axis)
        ```
    - Place atom D in this local frame using bond length, angle, and torsion angle:
        ```
        D_local = (d * cos(theta), d * sin(theta), 0)
        rotate around x_axis by torsion phi
        ```
    - Transform back to global coordinates.

#### ⚙️ Implementation Notes

- **Anchor Atoms:** Usually O3′(i-1); consistency with Stage B indexing critical.
- **Applying Torsions:** Each torsion is a rotation around a local bond axis. NeRF preferred.
- **Sugar Ring Closure:** Use ideal geometry or refine ring via minimization if flexible puckering.
- **Complexity:** Linear (O(N)) complexity with nucleotide count.

---

### 🧪 Validation, Error Metrics, and Constraints

- **RMSD Validation:**
    - Build RNA structure from known torsions (PDB).
    - Accept RMSD <0.5 Å.
- **Geometric Checks:** Verify bond lengths/angles (use **MolProbity Suite**).
- **Steric Clash Checks:** Validate absence of severe clashes; possibly minimize (MD engines).
- **Ring Closure & Constraints:** Use mini inverse-kinematics for sugar ring closure.
- **Numerical Stability:** Regularly recompute internal coordinates to verify against input torsion angles for drift checks.
- **Energy Scoring:** Optional short minimization to relieve minor steric strain.

---

## 🔧 Configuration & Execution (Hydra)

Stage C (reconstruction) is configured using Hydra. Parameters are defined in YAML files and can be overridden via the command line when running the stage's entry point.

**Entry Point:** `rna_predict.pipeline.stageC.stage_c_reconstruction`

**Configuration Files:**

* **Main Stage C settings:** `rna_predict/conf/model/stageC.yaml` (controls method selection, device, reconstruction options)
* **(Optional) MP-NeRF Model settings:** Referenced within `stageC.yaml` (e.g., `defaults: - mp_nerf_model: default_rna`). See `rna_predict/conf/mp_nerf_model/` for specific model variants.

These are loaded via the main `rna_predict/conf/default.yaml`.

### Key Configuration Parameters (`stageC.yaml`)

```yaml
# rna_predict/conf/model/stageC.yaml
defaults:
  - mp_nerf_model: default_rna # Selects which MP-NeRF config to load from mp_nerf_model/
  - _self_

stageC:
  method: "mp_nerf"            # "mp_nerf" or "legacy" (fallback)
  device: "cpu"                # "cpu" or "cuda"
  angle_representation: "cartesian" # Expected input format ('cartesian' or 'dihedral')
  do_ring_closure: false       # Apply ring closure constraints
  place_bases: true            # Reconstruct base atoms
  sugar_pucker: "C3'-endo"     # Default sugar pucker conformation
  # ... add other stageC specific parameters ...
```

*Note: MP-NeRF specific hyperparameters (layers, dimensions) are typically defined in files within `rna_predict/conf/mp_nerf_model/`, selected by the `defaults` list above.*

### Command-Line Overrides

Override parameters using dot notation:

* Use the legacy reconstruction method:
    ```bash
    python -m rna_predict.pipeline.stageC.stage_c_reconstruction stageC.method=legacy
    ```
* Run on CUDA:
    ```bash
    python -m rna_predict.pipeline.stageC.stage_c_reconstruction stageC.device=cuda
    ```
* Enable ring closure:
    ```bash
    python -m rna_predict.pipeline.stageC.stage_c_reconstruction stageC.do_ring_closure=true
    ```
* Disable base placement:
    ```bash
    python -m rna_predict.pipeline.stageC.stage_c_reconstruction stageC.place_bases=false
    ```
* Select a different MP-NeRF model configuration (assuming `rna_predict/conf/mp_nerf_model/fast_model.yaml` exists):
    ```bash
    python -m rna_predict.pipeline.stageC.stage_c_reconstruction mp_nerf_model=fast_model
    ```

### HPC Execution

For High Performance Computing (HPC) environments, see the [HPC Integration Guide](../integration/hydra_integration/hpc_overrides.md) for SLURM and GridEngine examples.

**Basic HPC Example:**
```bash
python -m rna_predict.pipeline.stageC.stage_c_reconstruction \
    stageC.device=cuda \
    stageC.method=mp_nerf \
    +hpc_cluster=slurm \
    hydra.launcher.gpus=2
```

### Typed Configuration (Optional)

Check `rna_predict/conf/config_schema.py` for potential typed dataclasses (e.g., `StageCConfig`) that provide structure and validation for the configuration.

---

### 📖 Detailed References & Acknowledgments

- **Murray et al. (2003)**: RNA backbone is rotameric (PNAS).
- **Richardson et al. (2008)**: RNA backbone suite nomenclature (RNA).
- **3DNA/DSSR**: Standard RNA geometry tools ([x3dna.org](http://x3dna.org)).
- **MolProbity Suite (Suitename)**: RNA rotamer and sugar pucker validation.
- User documentation: `Multi_Stage_RNA3D_Pipeline_Technical_Architecture&Implementation_Plan.md`, `torsion_angles.md`.

---

### 🎯 Comprehensive Conclusion

🧙🏾‍♂️ **Stage C** provides a mathematically rigorous and computationally efficient approach to converting predicted torsion angles into accurate RNA 3D structures. By leveraging rotation matrices, homogeneous transformations, and the NeRF method, FK ensures physically consistent and biologically valid atomic coordinates.

✅ **Recommended Next Steps:**

- Implement and rigorously test the provided `forward_kinematics` pseudocode.

- Validate against PDB structures.

- Integrate the validated method into Stage B for a complete RNA structural pipeline.

✨ **Additional Enhancements:**

- Include visual diagrams illustrating rotations and local reference frames.

- Incorporate MkDocs admonitions (`!!! note`) for clarity.

- Auto-generate Table of Contents for enhanced readability.