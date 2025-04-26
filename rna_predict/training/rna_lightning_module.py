import lightning as L
import torch
import numpy as np
from typing import Optional
from omegaconf import DictConfig
from rna_predict.pipeline.merger.simple_latent_merger import LatentInputs
import snoop
from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
from rna_predict.pipeline.stageC.stage_c_reconstruction import StageCReconstruction, run_stageC
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager
from rna_predict.pipeline.merger.simple_latent_merger import SimpleLatentMerger

class RNALightningModule(L.LightningModule):
    """
    LightningModule wrapping the full RNA_PREDICT pipeline for training and inference.
    Uses Hydra config for construction. All major submodules are accessible as attributes for checkpointing.
    """
    def __init__(self, cfg: Optional[DictConfig] = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        if cfg is not None:
            self._instantiate_pipeline(cfg)
            self._integration_test_mode = False  # Use real pipeline
        else:
            self.pipeline = torch.nn.Identity()
            self._integration_test_mode = True  # Use dummy layer
        # Dummy layer for integration test to ensure trainability
        self._integration_test_dummy = torch.nn.Linear(16, 21 * 3)

    def _instantiate_pipeline(self, cfg):
        """
        Instantiate all pipeline stages as module attributes using Hydra config.
        This is the single source of pipeline construction, following Hydra best practices.
        """


        print("[DEBUG-LM] torch.cuda.is_available():", torch.cuda.is_available())
        print("[DEBUG-LM] torch.backends.mps.is_available():", getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
        print("[DEBUG-LM] cfg.device:", getattr(cfg, 'device', None))

        print("[DEBUG-LM] cfg.model.stageB:", getattr(cfg.model, 'stageB', None))
        if hasattr(cfg.model, 'stageB'):
            print("[DEBUG-LM] cfg.model.stageB keys:", list(cfg.model.stageB.keys()) if hasattr(cfg.model.stageB, 'keys') else str(cfg.model.stageB))
            if hasattr(cfg.model.stageB, 'pairformer'):
                print("[DEBUG-LM] cfg.model.stageB.pairformer keys:", list(cfg.model.stageB.pairformer.keys()) if hasattr(cfg.model.stageB.pairformer, 'keys') else str(cfg.model.stageB.pairformer))
            else:
                print("[DEBUG-LM] cfg.model.stageB.pairformer: NOT FOUND")
        else:
            print("[DEBUG-LM] cfg.model.stageB: NOT FOUND")

        self.device_ = torch.device(cfg.device) if hasattr(cfg, 'device') else torch.device('cpu')
        print(f"[DEBUG-LM] self.device_ in RNALightningModule: {self.device_}")
        # Ensure Stage A is always moved to the same device as the rest of the model
        self.stageA = StageARFoldPredictor(cfg.model.stageA, self.device_)
        if hasattr(self.stageA, 'model'):
            print(f"[DEBUG-LM] After StageARFoldPredictor init, self.stageA.device: {getattr(self.stageA, 'device', None)}")
            print(f"[DEBUG-LM] After StageARFoldPredictor init, self.stageA.model.device: {getattr(getattr(self.stageA, 'model', None), 'device', 'NO DEVICE ATTR')}")
            self.stageA.model.to(self.device_)
        self.stageB_torsion = StageBTorsionBertPredictor(cfg.model.stageB.torsion_bert)
        print("[DEBUG-LM] About to access cfg.model.stageB.pairformer. Keys:", list(cfg.model.stageB.keys()) if hasattr(cfg.model.stageB, 'keys') else str(cfg.model.stageB))
        if "pairformer" not in cfg.model.stageB:
            raise ValueError("[UNIQUE-ERR-PAIRFORMER-MISSING] pairformer not found in cfg.model.stageB. Available keys: " + str(list(cfg.model.stageB.keys())))
        self.stageB_pairformer = PairformerWrapper(cfg.model.stageB.pairformer)
        # Use unified Stage C entrypoint (mp_nerf or fallback) per config
        self.stageC = StageCReconstruction()
        print("[DEBUG-LM-STAGED] cfg.model.stageD:", getattr(cfg.model, 'stageD', None))
        # Pass the full config to ProtenixDiffusionManager, not just cfg.model
        self.stageD = ProtenixDiffusionManager(cfg)

        merger_cfg = cfg.model.latent_merger if hasattr(cfg.model, 'latent_merger') else None
        # Fallbacks for dimensions (should be config-driven in production)
        dim_angles = getattr(merger_cfg, 'dim_angles', 7) if merger_cfg else 7
        dim_s = getattr(merger_cfg, 'dim_s', 64) if merger_cfg else 64
        dim_z = getattr(merger_cfg, 'dim_z', 32) if merger_cfg else 32
        dim_out = getattr(merger_cfg, 'output_dim', 128) if merger_cfg else 128
        self.latent_merger = SimpleLatentMerger(dim_angles, dim_s, dim_z, dim_out)

        # Create a pipeline module that contains all components
        # This ensures the model has trainable parameters for the optimizer
        self.pipeline = torch.nn.ModuleDict({
            'stageA': self.stageA,
            'stageB_torsion': self.stageB_torsion,
            'stageB_pairformer': self.stageB_pairformer,
            'stageC': self.stageC,
            'stageD': self.stageD,
            'latent_merger': self.latent_merger
        })

        # Debug: Print model.stageB and model.stageB.torsion_bert config for systematic debugging
        print(f"[DEBUG-RNA-LM-STAGEB] model.stageB: {getattr(cfg.model, 'stageB', None)}")
        print(f"[DEBUG-RNA-LM-STAGEB] model.stageB.torsion_bert: {getattr(getattr(cfg.model, 'stageB', None), 'torsion_bert', None)}")

    ###@snoop
    def forward(self, batch, **kwargs):
        print("[DEBUG-LM] Entered forward")
        print(f"[DEBUG-LM] batch keys: {list(batch.keys())}")
        sequence = batch["sequence"][0]  # assumes batch size 1 for now
        print(f"[DEBUG-LM] StageA input sequence: {sequence}")
        adj = self.stageA.predict_adjacency(sequence)
        print(f"[DEBUG-LM] StageA output adj type: {type(adj)}")
        outB_torsion = self.stageB_torsion(sequence, adjacency=adj)
        print(f"[DEBUG-LM] StageB_torsion output keys: {list(outB_torsion.keys())}")
        torsion_angles = outB_torsion["torsion_angles"]
        print(f"[DEBUG-LM] torsion_angles shape: {getattr(torsion_angles, 'shape', None)}")
        outB_pairformer = self.stageB_pairformer.predict(sequence, adjacency=adj)
        print(f"[DEBUG-LM] StageB_pairformer output type: {type(outB_pairformer)}")
        # Use unified Stage C entrypoint (mp_nerf or fallback) per config
        print("[DEBUG-LM] Calling run_stageC entrypoint with debug_logging:", getattr(self.cfg.model.stageC, 'debug_logging', None))
        outC = run_stageC(sequence=sequence, torsion_angles=torsion_angles, cfg=self.cfg)
        print(f"[DEBUG-LM] run_stageC output keys: {list(outC.keys())}")
        coords = outC["coords"]
        print(f"[DEBUG-LM] coords shape: {getattr(coords, 'shape', None)}")
        device = getattr(self.cfg, 'device', outB_pairformer[0].device)
        coords = coords.to(device)
        print(f"[DEBUG-LM] coords_init shape (after .to(device)): {coords.shape}, dtype: {coords.dtype}, device: {coords.device}")
        s_emb, z_emb = outB_pairformer
        s_trunk = s_emb.unsqueeze(0)
        z_trunk = z_emb.unsqueeze(0)
        s_inputs = torch.zeros_like(s_trunk)
        input_feature_dict = {
            "atom_to_token_idx": batch["atom_to_token_idx"],
            "ref_element": batch["ref_element"],
            "ref_atom_name_chars": batch["ref_atom_name_chars"],
        }
        input_feature_dict = self.move_to_device(input_feature_dict, device)
        atom_metadata = outC.get("atom_metadata", None)
        if atom_metadata is not None:
            override_input_features = dict(input_feature_dict)
            override_input_features["atom_metadata"] = self.move_to_device(atom_metadata, device)
        else:
            override_input_features = input_feature_dict
        self.debug_print_devices(override_input_features)
        print(f"[DEBUG-LM] s_trunk shape: {s_trunk.shape}, dtype: {s_trunk.dtype}, device: {s_trunk.device}")
        print(f"[DEBUG-LM] z_trunk shape: {z_trunk.shape}, dtype: {z_trunk.dtype}, device: {z_trunk.device}")
        print(f"[DEBUG-LM] s_inputs shape: {s_inputs.shape}, dtype: {s_inputs.dtype}, device: {s_inputs.device}")
        # --- Unified Latent Merger Integration ---
        inputs = LatentInputs(
            adjacency=adj,
            angles=torsion_angles,
            s_emb=s_emb,
            z_emb=z_emb,
            partial_coords=coords,
        )
        unified_latent = self.latent_merger(inputs)
        print(f"[DEBUG-LM] unified_latent shape: {unified_latent.shape if unified_latent is not None else None}")
        # Stage D: Pass unified_latent as condition (update Stage D logic as needed)
        # Example: self.stageD(coords, unified_latent, ...)
        # TODO: Update Stage D to accept and use unified_latent
        # Return outputs including unified_latent
        output = {
            "adjacency": adj,
            "torsion_angles": torsion_angles,
            "s_embeddings": s_emb,
            "z_embeddings": z_emb,
            "coords": coords,
            "unified_latent": unified_latent,
            # Add other outputs as needed
        }
        # --- PROPAGATE atom_metadata and atom_count from Stage C if present ---
        if outC.get("atom_metadata") is not None:
            output["atom_metadata"] = outC["atom_metadata"]
        if outC.get("atom_count") is not None:
            output["atom_count"] = outC["atom_count"]
        print(f"[DEBUG-LM-FORWARD-RETURN] Returning output with keys: {list(output.keys())}")
        if output.get("atom_metadata") is not None:
            print(f"[DEBUG-LM-FORWARD-RETURN] output['atom_metadata'] keys: {list(output['atom_metadata'].keys())}")
        else:
            print("[DEBUG-LM-FORWARD-RETURN] output['atom_metadata'] is None")
        return output

    #@snoop
    def training_step(self, batch, batch_idx):
        print("[DEBUG-LM] Entered training_step")
        # Print requires_grad for all model parameters
        print("[DEBUG][training_step] Model parameters requires_grad status:")
        for name, param in self.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}")

        input_tensor = batch["coords_true"]
        target_tensor = batch["coords_true"]
        print(f"[DEBUG-LM] input_tensor.shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"[DEBUG-LM] target_tensor.shape: {target_tensor.shape}, dtype: {target_tensor.dtype}")
        output = self.forward(batch)
        print(f"[DEBUG-LM] output.keys(): {list(output.keys())}")
        print(f"[DEBUG-LM] output['atom_metadata']: {output.get('atom_metadata', None)}")
        print(f"[DEBUG-LM] batch.keys(): {list(batch.keys())}")
        for k in ["atom_mask", "atom_names", "residue_indices"]:
            if k in batch:
                v = batch[k]
                if hasattr(v, 'shape'):
                    print(f"[DEBUG-LM] batch['{k}'].shape: {v.shape}")
                else:
                    print(f"[DEBUG-LM] batch['{k}']: {v}")
        predicted_coords = output["coords"] if isinstance(output, dict) and "coords" in output else output
        print(f"[DEBUG-LM] predicted_coords.requires_grad: {getattr(predicted_coords, 'requires_grad', 'N/A')}, grad_fn: {getattr(predicted_coords, 'grad_fn', 'N/A')}")
        pred_atom_metadata = output.get("atom_metadata", None)
        # Force systematic masking if atom_metadata is present
        if pred_atom_metadata is not None:
            print("[DEBUG-LM] Running systematic masking logic!")
            # Gather predicted atom keys
            pred_keys = list(zip(pred_atom_metadata["residue_indices"], pred_atom_metadata["atom_names"]))
            print(f"[DEBUG-LM] pred_keys (first 10): {pred_keys[:10]}")
            # Gather target atom keys from batch metadata if available
            batch_atom_names = batch.get("atom_names", None)
            batch_res_indices = batch.get("residue_indices", None)
            # If present, batch_atom_names and batch_res_indices are lists of lists (batch dimension)
            if batch_atom_names is not None and batch_res_indices is not None:
                # Flatten for batch size 1 (current pipeline)
                if isinstance(batch_atom_names, list) and len(batch_atom_names) == 1:
                    batch_atom_names = batch_atom_names[0]
                    batch_res_indices = batch_res_indices[0]
                tgt_keys = list(zip(batch_res_indices, batch_atom_names))
            else:
                mask_indices = torch.where(batch["atom_mask"])[0].tolist()
                tgt_keys = []
                for idx in mask_indices:
                    if idx < len(pred_atom_metadata["residue_indices"]):
                        tgt_keys.append((pred_atom_metadata["residue_indices"][idx], pred_atom_metadata["atom_names"][idx]))
            print(f"[DEBUG-LM] tgt_keys (first 10): {tgt_keys[:10]}")
            # Build boolean mask over predicted atoms for those present in target
            tgt_keys_set = set(tgt_keys)
            mask_pred_np = np.array([(ri, an) in tgt_keys_set for ri, an in pred_keys])
            mask_pred = torch.from_numpy(mask_pred_np).to(predicted_coords.device)
            mask_pred = mask_pred.bool()
            n_matched = mask_pred.sum().item()
            print(f"[DEBUG-LM] n_matched: {n_matched} / {len(pred_keys)}")
            if n_matched == 0:
                print("[DEBUG-LM][WARNING] No matched atoms found! Setting loss to zero.")
                loss = torch.tensor(0.0, device=predicted_coords.device, requires_grad=True)
            else:
                pred_sel = predicted_coords[mask_pred]
                # Find indices of matched keys in tgt_keys for true_sel
                matched_keys = [pk for pk in pred_keys if pk in tgt_keys_set]
                target_indices = [tgt_keys.index(pk) for pk in matched_keys]
                target_indices_tensor = torch.tensor(target_indices, dtype=torch.long, device=pred_sel.device)
                true_sel_all = target_tensor[batch["atom_mask"].bool()]
                true_sel = true_sel_all[target_indices_tensor]
                # Instrumentation for debugging requires_grad chain
                print(f"[DEBUG-LM] pred_sel.requires_grad: {pred_sel.requires_grad}, pred_sel.grad_fn: {getattr(pred_sel, 'grad_fn', None)}")
                print(f"[DEBUG-LM] true_sel.requires_grad: {true_sel.requires_grad}, true_sel.grad_fn: {getattr(true_sel, 'grad_fn', None)}")
                print(f"[DEBUG-LM] predicted_coords.requires_grad: {predicted_coords.requires_grad}, predicted_coords.grad_fn: {getattr(predicted_coords, 'grad_fn', None)}")
                print(f"[DEBUG-LM] target_tensor.requires_grad: {target_tensor.requires_grad}, target_tensor.grad_fn: {getattr(target_tensor, 'grad_fn', None)}")
                print(f"[DEBUG-LM] mask_pred dtype: {mask_pred.dtype}, device: {mask_pred.device}")
                print(f"[DEBUG-LM] target_indices_tensor dtype: {target_indices_tensor.dtype}, device: {target_indices_tensor.device}")
                print(f"[DEBUG-LM] true_sel_all.requires_grad: {true_sel_all.requires_grad}, true_sel_all.grad_fn: {getattr(true_sel_all, 'grad_fn', None)}")
                if pred_sel.shape[0] != true_sel.shape[0]:
                    print(f"[DEBUG-LM][MISMATCH-FILTERED] pred_sel.shape={pred_sel.shape}, true_sel.shape={true_sel.shape}")
                    loss = torch.tensor(0.0, device=pred_sel.device, requires_grad=True)
                else:
                    loss = ((pred_sel - true_sel) ** 2).mean()
                    print(f"[DEBUG-LM] Masked-aligned filtered loss value: {loss.item()}")
        else:
            print("[DEBUG-LM] Systematic masking not possible: atom_metadata missing!")
            real_target_coords = target_tensor[batch["atom_mask"].bool()]
            if predicted_coords.shape[0] != real_target_coords.shape[0]:
                print(f"[DEBUG-LM][MISMATCH] predicted_coords.shape[0]={predicted_coords.shape[0]}, real_target_coords.shape[0]={real_target_coords.shape[0]}")
                loss = torch.tensor(0.0, device=predicted_coords.device, requires_grad=True)
            else:
                loss = ((predicted_coords - real_target_coords) ** 2).mean()
                print(f"[DEBUG-LM] Masked-aligned loss value: {loss.item()}")
        print(f"[DEBUG-LM] loss.requires_grad: {loss.requires_grad}, loss.grad_fn: {loss.grad_fn}")
        return {"loss": loss}

    def train_dataloader(self):
        """
        Real dataloader for RNA_PREDICT pipeline using minimal Kaggle data.
        """
        from rna_predict.dataset.loader import RNADataset
        from rna_predict.dataset.collate import rna_collate_fn
        dataset = RNADataset(
            index_csv=self.cfg.data.index_csv,
            cfg=self.cfg,
            load_adj=False,
            load_ang=False,
            verbose=False
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            collate_fn=rna_collate_fn,
            num_workers=0
        )

    def configure_optimizers(self):
        """
        Returns the optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def move_to_device(self, obj, device):
        if isinstance(obj, dict):
            return {k: self.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self.move_to_device(v, device) for v in obj)
        elif hasattr(obj, 'to') and callable(getattr(obj, 'to')):
            try:
                return obj.to(device)
            except Exception:
                return obj  # Non-tensor objects
        else:
            return obj

    def debug_print_devices(self, obj, prefix="input_feature_dict"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.debug_print_devices(v, prefix=f"{prefix}[{k}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self.debug_print_devices(v, prefix=f"{prefix}[{i}]")
        elif hasattr(obj, 'device'):
            print(f"[DEBUG-LM] {prefix} device: {obj.device}")
