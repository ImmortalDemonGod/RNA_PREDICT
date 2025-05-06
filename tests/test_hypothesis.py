import torch
from rna_predict.pipeline.stageD.diffusion.bridging.sequence_utils import extract_sequence

# Create a tensor similar to the one in the failing test
tensor = torch.tensor([65, 85, 71, 67])  # ASCII for "AUGC"
input_features = {"sequence": tensor}

try:
    result = extract_sequence(None, input_features, {})
    print(f"Success! Result: {result}")
except ValueError as e:
    print(f"Error: {e}")
    
# Print the string representation of the tensor values
print("String representation of tensor values:")
for val in tensor.tolist():
    print(f"Value: {val}, String: '{str(val)}', Length: {len(str(val))}")
