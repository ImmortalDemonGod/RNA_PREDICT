import torch


class StageCReconstruction:
    """
    Original fallback approach for Stage C (legacy).
    You can keep it if you want a backup method.
    """

    def __call__(self, torsion_angles: torch.Tensor):
        N = torsion_angles.size(0)
        coords = torch.zeros((N * 3, 3))
        return {"coords": coords, "atom_count": coords.size(0)}


def run_stageC_rna_mpnerf(
    sequence: str,
    predicted_torsions: torch.Tensor,
    device="cpu",
    do_ring_closure=False,
    place_bases=True,
):
    """
    Main RNA Stage C function that calls mp-nerf-based methods to build
    backbone + optional ring closure + base placement. Returns a coords dict.
    """
    from rna_predict.pipeline.stageC.mp_nerf.rna import (
        build_scaffolds_rna_from_torsions,
        handle_mods,
        place_rna_bases,
        rna_fold,
        skip_missing_atoms,
    )

    # 1) Build scaffolds from torsions
    scaffolds = build_scaffolds_rna_from_torsions(
        sequence, predicted_torsions, device=device
    )

    # 2) Optionally handle partial/missing data
    scaffolds = skip_missing_atoms(sequence, scaffolds)
    scaffolds = handle_mods(sequence, scaffolds)

    # 3) Fold backbone
    coords_bb = rna_fold(scaffolds, device=device, do_ring_closure=do_ring_closure)

    # 4) Place bases if desired
    if place_bases:
        coords_full = place_rna_bases(
            coords_bb, sequence, scaffolds["angles_mask"], device=device
        )
    else:
        coords_full = coords_bb

    total_atoms = coords_full.shape[0] * coords_full.shape[1]
    return {"coords": coords_full, "atom_count": total_atoms}


def run_stageC(
    sequence: str, torsion_angles: torch.Tensor, method="mp_nerf", device="cpu"
):
    """
    Unified entry point for Stage C. If method="mp_nerf", we call run_stageC_rna_mpnerf.
    Otherwise, fallback to the old StageCReconstruction.
    """
    if method == "mp_nerf":
        return run_stageC_rna_mpnerf(sequence, torsion_angles, device=device)
    else:
        stageC = StageCReconstruction()
        return stageC(torsion_angles)


if __name__ == "__main__":
    sample_seq = "ACGU"
    # Dummy torsions (L=4, dims=7 for alpha..zeta, chi)
    dummy_torsions = torch.zeros((len(sample_seq), 7))
    out = run_stageC(sample_seq, dummy_torsions, method="mp_nerf", device="cpu")
    print("RNA coords shape:", out["coords"].shape, " total atoms:", out["atom_count"])
