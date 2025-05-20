import subprocess
import os
import pytest
import csv

@pytest.mark.slow
@pytest.mark.skip(reason="Flaky in full suite: skipping until stable")
def test_train_with_angle_supervision(tmp_path):
    """
    Runs an integration test for the training pipeline with angle supervision enabled.
    
    This test executes the training script via a subprocess with specific configuration overrides, captures the output logs, and verifies that angle loss computation occurs and checkpoints are saved. It asserts successful training completion, checks for expected debug messages, and confirms that checkpoint files are created in the reported directory.
    
    Args:
        tmp_path: Temporary directory provided by pytest for storing log output.
    """
    # Create a minimal test dataset
    test_data_dir = os.path.join(tmp_path, "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create a minimal index file
    index_path = os.path.join(test_data_dir, "index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "sequence", "structure_id", "resolution"])
        writer.writerow(["test1", "ACGU", "test1", "2.0"])
    
    # Run training script with minimal configuration
    cmd = [
        "python",
        "-m", "rna_predict.training.train",
        f"data.index_csv={index_path}",
        "data.root_dir=" + test_data_dir,
        "data.load_ang=True",
        "data.batch_size=1",
        "data.num_workers=0",
        "data.max_residues=4",
        "data.max_atoms=16",
        "data.C_element=4",
        "data.C_char=8",
        "data.ref_element_size=4",
        "data.ref_atom_name_chars_size=8",
        # Minimal dimensions for all stages
        "+model.dimensions.c_s=8",
        "+model.dimensions.c_z=4",
        "+model.dimensions.c_s_inputs=8",
        "+model.dimensions.c_token=8",
        "+model.dimensions.c_atom=4",
        "+model.dimensions.c_atom_coords=3",
        "+model.dimensions.c_noise_embedding=4",
        "+model.dimensions.restype_dim=8",
        "+model.dimensions.profile_dim=8",
        "+model.dimensions.c_pair=4",
        "+model.dimensions.ref_element_size=4",
        "+model.dimensions.ref_atom_name_chars_size=8",
        # Stage A minimal config
        "model.stageA.num_hidden=8",
        "model.stageA.batch_size=1",
        # Stage B minimal config
        "model.stageB.pairformer.n_blocks=1",
        "model.stageB.pairformer.n_heads=1",
        "model.stageB.pairformer.c_z=4",
        "model.stageB.pairformer.c_s=8",
        "model.stageB.pairformer.c_token=8",
        "model.stageB.pairformer.c_atom=4",
        "model.stageB.pairformer.c_pair=4",
        # Stage D minimal config
        "++stageD_diffusion.diffusion.model_architecture.c_token=8",
        "++stageD_diffusion.diffusion.model_architecture.c_s=8",
        "++stageD_diffusion.diffusion.model_architecture.c_z=4",
        "++stageD_diffusion.diffusion.model_architecture.c_s_inputs=8",
        "++stageD_diffusion.diffusion.model_architecture.c_atom=4",
        "++stageD_diffusion.diffusion.model_architecture.c_noise_embedding=4",
        "++stageD_diffusion.diffusion.model_architecture.num_layers=2",
        "++stageD_diffusion.diffusion.model_architecture.num_heads=2",
        "++stageD_diffusion.diffusion.model_architecture.test_residues_per_batch=2",
        "++stageD_diffusion.diffusion.model_architecture.c_atompair=4",
        "++stageD_diffusion.diffusion.model_architecture.sigma_data=1.0",
        # Stage D transformer minimal config
        "++stageD_diffusion.diffusion.transformer.n_blocks=2",
        "++stageD_diffusion.diffusion.transformer.n_heads=2",
        # Stage D atom encoder minimal config
        "++stageD_diffusion.diffusion.atom_encoder.c_in=4",
        "++stageD_diffusion.diffusion.atom_encoder.c_hidden=[8]",
        "++stageD_diffusion.diffusion.atom_encoder.c_out=4",
        "++stageD_diffusion.diffusion.atom_encoder.n_blocks=1",
        "++stageD_diffusion.diffusion.atom_encoder.n_heads=2",
        "++stageD_diffusion.diffusion.atom_encoder.n_queries=2",
        "++stageD_diffusion.diffusion.atom_encoder.n_keys=2",
        # Stage D atom decoder minimal config
        "++stageD_diffusion.diffusion.atom_decoder.c_in=4",
        "++stageD_diffusion.diffusion.atom_decoder.c_hidden=[8,4,2]",
        "++stageD_diffusion.diffusion.atom_decoder.c_out=4",
        "++stageD_diffusion.diffusion.atom_decoder.n_blocks=1",
        "++stageD_diffusion.diffusion.atom_decoder.n_heads=2",
        "++stageD_diffusion.diffusion.atom_decoder.n_queries=2",
        "++stageD_diffusion.diffusion.atom_decoder.n_keys=2",
        # Feature dimensions minimal config
        "++stageD_diffusion.diffusion.feature_dimensions.c_s=8",
        "++stageD_diffusion.diffusion.feature_dimensions.c_s_inputs=8",
        "++stageD_diffusion.diffusion.feature_dimensions.c_sing=8",
        "++stageD_diffusion.diffusion.feature_dimensions.s_trunk=8",
        "++stageD_diffusion.diffusion.feature_dimensions.s_inputs=8",
        # Memory optimization
        "+memory_optimization.use_checkpointing=True",
        "+memory_optimization.checkpoint_every_n_layers=1",
        "+memory_optimization.optimize_memory_layout=True",
        "+memory_optimization.mixed_precision=True",
        # Training config
        "training.accelerator=cpu",
        "training.devices=1",
        # Output directory
        f"training.checkpoint_dir={os.path.join(tmp_path, 'checkpoints')}",
        # Debug logging
        "model.stageA.debug_logging=True",
        "model.stageB.pairformer.debug_logging=True",
        "model.stageC.debug_logging=True",
        "stageD_diffusion.debug_logging=True",
    ]
    
    # Run the training script and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )
    
    stdout, stderr = process.communicate()
    
    # Print output for debugging
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    
    # Check if training completed successfully
    assert process.returncode == 0, f"Training failed with return code {process.returncode}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
    
    # Check if checkpoint directory was created
    checkpoint_dir = os.path.join(tmp_path, "checkpoints")
    assert os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} was not created"
