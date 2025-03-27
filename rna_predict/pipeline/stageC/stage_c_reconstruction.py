import torch


class StageCReconstruction:
    """
    Legacy fallback approach for Stage C. Used if method != 'mp_nerf'.
    Returns trivial coords.
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
    sugar_pucker="C3'-endo",
):
    """
    Main RNA Stage C function. We build scaffolds referencing final_kb_rna,
    fold the backbone, optionally place bases, and do ring closure if desired.
    """
    from rna_predict.pipeline.stageC.mp_nerf.rna import (
        build_scaffolds_rna_from_torsions,
        handle_mods,
        place_rna_bases,
        rna_fold,
        skip_missing_atoms,
    )

    # 1) Build scaffolds from predicted torsions
    scaffolds = build_scaffolds_rna_from_torsions(
        seq=sequence,
        torsions=predicted_torsions,
        device=device,
        sugar_pucker=sugar_pucker,
    )

    # 2) Potentially skip missing atoms or handle special modifications
    scaffolds = skip_missing_atoms(sequence, scaffolds)
    scaffolds = handle_mods(sequence, scaffolds)

    # 3) Fold backbone with mp_nerf approach
    coords_bb = rna_fold(scaffolds, device=device, do_ring_closure=do_ring_closure)

    # 4) Optionally place base atoms
    if place_bases:
        coords_full = place_rna_bases(
            coords_bb, sequence, scaffolds["angles_mask"], device=device
        )
    else:
        coords_full = coords_bb

    # final
    total_atoms = coords_full.shape[0] * coords_full.shape[1]
    return {"coords": coords_full, "atom_count": total_atoms}


def run_stageC(
    sequence: str,
    torsion_angles: torch.Tensor,
    method="mp_nerf",
    device="cpu",
    do_ring_closure=False,
    place_bases=True,
    sugar_pucker="C3'-endo",
):
    """
    Unified Stage C entrypoint. If method=="mp_nerf", uses final approach referencing final_kb_rna,
    else fallback to the trivial StageCReconstruction.
    """
    if method == "mp_nerf":
        return run_stageC_rna_mpnerf(
            sequence=sequence,
            predicted_torsions=torsion_angles,
            device=device,
            do_ring_closure=do_ring_closure,
            place_bases=place_bases,
            sugar_pucker=sugar_pucker,
        )
    else:
        stageC = StageCReconstruction()
        return stageC(torsion_angles)


if __name__ == "__main__":
    # example usage
    sample_seq = "ACGU"
    dummy_torsions = torch.zeros((len(sample_seq), 7))  # alpha..zeta, chi in degrees
    out = run_stageC(
        sequence=sample_seq,
        torsion_angles=dummy_torsions,
        method="mp_nerf",
        device="cpu",
        do_ring_closure=False,
        place_bases=True,
        sugar_pucker="C3'-endo",
    )
    print("RNA coords shape:", out["coords"].shape, " total atoms:", out["atom_count"])
