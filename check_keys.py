import torch
state = torch.load('models/mlp_best.pth', weights_only=True, map_location='cpu')
print("=== MLP Saved Keys ===")
for i, k in enumerate(state.keys()):
    print(f"{i}: {k}")
