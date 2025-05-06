import torch

# Create a tensor similar to the one in the failing test
tensor = torch.tensor([65, 85, 71, 67])  # ASCII for "AUGC"

# Print the string representation of the tensor values
print("String representation of tensor values:")
for val in tensor.tolist():
    print(f"Value: {val}, String: '{str(val)}', Length: {len(str(val))}")
    
# Check if all strings have length 1
all_len_1 = all(len(str(val)) == 1 for val in tensor.tolist())
print(f"All strings have length 1: {all_len_1}")
