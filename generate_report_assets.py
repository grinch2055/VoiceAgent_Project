# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("?? Generating Report Assets...")

# --- ASSET 1: CNN Feature Maps (Fiche Synthese CNN) ---
print("1. Generating CNN Feature Maps...")

# Define LeNet (Must match your training script exactly)
class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Load CNN Model
cnn_model = LeNet()
try:
    cnn_model.load_state_dict(torch.load("cnn_lenet.pth", weights_only=True))
    cnn_model.eval()
    
    # Hook setup
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())
    
    cnn_model.net[0].register_forward_hook(hook_fn)
    
    # Forward pass with dummy data (replace with test image later)
    dummy = torch.randn(1, 1, 128, 100)
    with torch.no_grad():
        cnn_model(dummy)
    
    # Save Feature Map visualization
    if activations:
        feats = activations[0][0, 0, :, :].numpy()  # Take 1st batch, 1st channel
        plt.figure(figsize=(6, 4))
        plt.imshow(feats, cmap='viridis')
        plt.title('CNN Layer 1: Activation Map (Feature Map)')
        plt.colorbar()
        plt.savefig("report/figures/cnn_feature_map.png")
        plt.close()
        print("   ? Saved: report/figures/cnn_feature_map.png")

except FileNotFoundError:
    print("   ?? cnn_lenet.pth not found. Skipping CNN visuals.")

# --- ASSET 2: MLP Confusion Matrix (Fiche Synthese MLP) ---
print("2. Generating Confusion Matrix...")
try:
    # Load MLP model
    class MLP_Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(24, 64)
            self.fc2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 4)
        def forward(self, x): return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    mlp_model = MLP_Custom()
    mlp_model.load_state_dict(torch.load("mlp_best.pth", weights_only=True))
    mlp_model.eval()
    
    # Load test data
    X, y = torch.load("data/placeholder/part1.pt", weights_only=True)
    X_test, y_test = X[400:], y[400:]
    
    preds = mlp_model(X_test).argmax(dim=1)
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('MLP Classification Confusion Matrix')
    plt.savefig("report/figures/mlp_confusion_matrix.png")
    plt.close()
    print("   ? Saved: report/figures/mlp_confusion_matrix.png")

except FileNotFoundError:
    print("   ?? mlp_best.pth not found. Skipping MLP visuals.")

print("?? Done. Check 'report/figures/' for your report images.")
