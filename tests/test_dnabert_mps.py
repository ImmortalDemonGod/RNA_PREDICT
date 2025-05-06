import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sayby/rna_torsionbert"

def print_param_devices(model):
    for name, param in model.named_parameters():
        print(f"Param: {name}, device: {param.device}")

def main():
    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model class:", model.__class__)
    print("Model config:", model.config)
    print("Before to(device):")
    print_param_devices(model)
    try:
        model = model.to(device)
    except Exception as e:
        print("Exception during model.to(device):", e)
    print("After to(device):")
    print_param_devices(model)
    print("Model class after to(device):", model.__class__)
    print("Model config after to(device):", model.config)
    # Try moving submodules
    for name, module in model.named_modules():
        try:
            module.to(device)
        except Exception as e:
            print(f"Could not move submodule {name} to {device}: {e}")
    print("After submodule to(device):")
    print_param_devices(model)

if __name__ == "__main__":
    main()
