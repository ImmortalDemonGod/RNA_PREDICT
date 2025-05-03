import torch
from rna_predict.pipeline.stageB.pairwise.pairformer import MSABlock, MSAModule
from rna_predict.conf.config_schema import MSAConfig

def test_msa_block():
    print("Testing MSABlock initialization...")
    try:
        cfg = MSAConfig(
            c_m=8,
            c=8,
            c_z=8,
            dropout=0.1,
            pair_dropout=0.25
        )
        mb = MSABlock(cfg=cfg)
        print("MSABlock initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing MSABlock: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_msa_module():
    print("Testing MSAModule initialization...")
    try:
        minimal_cfg = MSAConfig(
            n_blocks=1,
            c_m=8,
            c=8,
            c_z=16,
            dropout=0.0,
            c_s_inputs=8,
            enable=False,
            blocks_per_ckpt=1,
            input_feature_dims={"msa": 32, "has_deletion": 1, "deletion_value": 1},
            pair_dropout=0.25
        )
        mm = MSAModule(minimal_cfg)
        print("MSAModule initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing MSAModule: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_msa_block()
    test_msa_module()
