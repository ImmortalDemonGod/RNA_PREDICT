import torch
from rna_predict.pipeline.stageD.diffusion.generator import TrainingNoiseSampler, InferenceNoiseScheduler

# Test TrainingNoiseSampler
print("Testing TrainingNoiseSampler...")
sampler = TrainingNoiseSampler()
try:
    # This should work with a device
    noise_levels = sampler(size=torch.Size([2, 3]), device=torch.device("cpu"))
    print(f"Success! Shape: {noise_levels.shape}")
    
    # This should fail without a device
    try:
        noise_levels = sampler(size=torch.Size([2, 3]))
        print("ERROR: Should have failed without device")
    except TypeError as e:
        print(f"Expected error without device: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Test InferenceNoiseScheduler
print("\nTesting InferenceNoiseScheduler...")
scheduler = InferenceNoiseScheduler()
try:
    # This should work with a device
    schedule = scheduler(device=torch.device("cpu"))
    print(f"Success! Shape: {schedule.shape}")
    
    # This should fail without a device
    try:
        schedule = scheduler()
        print("ERROR: Should have failed without device")
    except TypeError as e:
        print(f"Expected error without device: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
