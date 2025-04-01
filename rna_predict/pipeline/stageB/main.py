import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
from rna_predict.pipeline.stageC.stage_c_reconstruction import StageCReconstruction


def run_pipeline(sequence: str):
    """
    A simpler demonstration that runs Stage A -> TorsionBERT -> Stage C, 
    but does not unify TorsionBERT + Pairformer in one call.
    """
    stageA = StageARFoldPredictor(config={})
    adjacency = stageA.predict_adjacency(sequence)
    adjacency_t = torch.from_numpy(adjacency).float()
    print(f"[Stage A] adjacency shape = {adjacency_t.shape}")

    # Stage B: TorsionBERT only (angles)
    stageB = StageBTorsionBertPredictor(
        model_name_or_path="sayby/rna_torsionbert",
        device="cpu",
        angle_mode="degrees",
        num_angles=7,
        max_length=512,
    )
    outB = stageB(sequence, adjacency_t)
    torsion_angles = outB["torsion_angles"]
    print(f"[Stage B] angles shape = {torsion_angles.shape}")

    # Stage C: Generate dummy 3D coords
    stageC = StageCReconstruction()
    outC = stageC(torsion_angles)
    coords = outC["coords"]
    print(f"[Stage C] coords shape = {coords.shape}, #atoms = {outC['atom_count']}")


# --- NEW FUNCTION: run_stageB_combined ---
def run_stageB_combined(
    sequence: str,
    adjacency_matrix: torch.Tensor,
    torsion_bert_model: StageBTorsionBertPredictor,
    pairformer_model: PairformerWrapper,
    device: str = "cpu",
    init_z_from_adjacency: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Runs TorsionBERT + Pairformer in a single pass (unified Stage B).

    Args:
      sequence: RNA sequence, e.g., "ACGUACGU"
      adjacency_matrix: shape [N, N] from Stage A
      torsion_bert_model: TorsionBertPredictor instance
      pairformer_model: PairformerWrapper instance
      device: "cpu" or "cuda"
      init_z_from_adjacency: if True, initialize pair embeddings with adjacency

    Returns:
      {
        "torsion_angles": (N, #angles or 2*#angles),
        "s_embeddings":   (N, c_s),
        "z_embeddings":   (N, N, c_z)
      }
    """
    pairformer_model.to(device)
    torsion_bert_model.model.to(device)

    # 1) TorsionBERT -> torsion angles
    torsion_out = torsion_bert_model(sequence, adjacency=adjacency_matrix)
    torsion_angles = torsion_out["torsion_angles"].to(device)
    N = torsion_angles.size(0)

    adjacency_matrix = adjacency_matrix.to(device)

    # 2) Prepare single (s) and pair (z) embeddings
    c_s = pairformer_model.c_s
    c_z = pairformer_model.c_z
    init_s = torch.randn((1, N, c_s), device=device, requires_grad=True)

    if init_z_from_adjacency:
        # Expand adjacency to shape (1, N, N, c_z)
        init_z = adjacency_matrix.unsqueeze(-1).expand(-1, -1, c_z).unsqueeze(0).clone()
        init_z += 0.01 * torch.randn_like(init_z)
        pair_mask = (adjacency_matrix > 0).float().unsqueeze(0)
    else:
        init_z = torch.randn((1, N, N, c_z), device=device, requires_grad=True)
        pair_mask = torch.ones((1, N, N), device=device)

    # 3) Forward pass in Pairformer
    s_up, z_up = pairformer_model(init_s, init_z, pair_mask)

    return {
        "torsion_angles": torsion_angles,
        "s_embeddings": s_up.squeeze(0),  # shape [N, c_s]
        "z_embeddings": z_up.squeeze(0),  # shape [N, N, c_z]
    }


# --- DEMO: Minimal Gradient Flow Test ---
def demo_gradient_flow_test():
    """
    Demonstrates a small forward/backward pass across:
      - TorsionBERT
      - Pairformer
      - A final MLP or Linear
    Confirms each obtains non-None grads.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running gradient flow test on device: {device}")

    # 1) Instantiate TorsionBERT & Pairformer
    try:
        torsion_model = StageBTorsionBertPredictor(model_name_or_path="sayby/rna_torsionbert", device=device)
    except Exception as e:
        print(f"[Warning] Could not load 'sayby/rna_torsionbert'. Using dummy path. Error: {e}")
        torsion_model = StageBTorsionBertPredictor(model_name_or_path="dummy_invalid_path", device=device)

    pairformer_model = PairformerWrapper(n_blocks=2, c_z=32, c_s=64, dropout=0.1).to(device)

    # 2) Fake data
    seq = "ACGUACGU"  # length N=8
    adjacency_mat = torch.ones((8, 8), device=device)  # or random

    # 3) Forward pass with the new run_stageB_combined
    outB = run_stageB_combined(
        sequence=seq,
        adjacency_matrix=adjacency_mat,
        torsion_bert_model=torsion_model,
        pairformer_model=pairformer_model,
        device=device,
        init_z_from_adjacency=False
    )
    s_emb = outB["s_embeddings"]     # [N, c_s], e.g., [8, 64]
    torsion_angles = outB["torsion_angles"]  # [N, angles]
    z_emb = outB["z_embeddings"]     # [N, N, c_z], e.g., [8, 8, 32]

    # 4) Final linear layers
    # Merge or separately process s_emb, torsion_angles, z_emb
    final_head_s = torch.nn.Linear(64, 3).to(device) # from s_emb dimension
    final_head_angles = torch.nn.Linear(torsion_angles.shape[-1], 3).to(device)
    final_head_z = torch.nn.Linear(32, 3).to(device)

    coords_pred_s = final_head_s(s_emb)
    coords_pred_angles = final_head_angles(torsion_angles)
    z_pooled = z_emb.mean(dim=1)  # shape [N, 32]
    coords_pred_z = final_head_z(z_pooled)

    coords_pred = coords_pred_s + coords_pred_angles + coords_pred_z
    target = torch.zeros_like(coords_pred)

    # 5) Loss and backward
    loss = F.mse_loss(coords_pred, target)
    print(f"Loss: {loss.item()}")
    torsion_model.model.zero_grad()
    pairformer_model.zero_grad()
    final_head_s.zero_grad()
    final_head_angles.zero_grad()
    final_head_z.zero_grad()

    loss.backward()

    # 6) Inspect gradients
    print("\n--- Grad Check ---")
    for name, param in final_head_s.named_parameters():
        if param.grad is not None:
            print(f"  [final_head_s] {name}, grad sum={param.grad.abs().sum().item():.4e}")
    
    for name, param in final_head_angles.named_parameters():
        if param.grad is not None:
            print(f"  [final_head_angles] {name}, grad sum={param.grad.abs().sum().item():.4e}")
            
    for name, param in final_head_z.named_parameters():
        if param.grad is not None:
            print(f"  [final_head_z] {name}, grad sum={param.grad.abs().sum().item():.4e}")
            
    for name, param in pairformer_model.named_parameters():
        if param.grad is not None:
            print(f"  [pairformer] {name}, grad sum={param.grad.abs().sum().item():.4e}")
            
    # For TorsionBERT, we only check a few params due to potential size
    torsion_params = list(torsion_model.model.named_parameters())
    if torsion_params:
        for name, param in torsion_params[:3]:  # Show first few
            if param.grad is not None:
                print(f"  [torsion_bert] {name}, grad sum={param.grad.abs().sum().item():.4e}")


def main():
    """
    Original main remains or can call the gradient flow test:
    """
    sample_seq = "ACGUACGU"
    run_pipeline(sample_seq)   # The old pipeline demonstration
    print("\n=== Now Running Gradient Flow Test ===")
    demo_gradient_flow_test()


if __name__ == "__main__":
    main()
