# -*- coding: utf-8 -*-
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)

# Dataset
DATA = "data/raw_voice/part1.pt"
X, y = torch.load(DATA, weights_only=True)
X_tr, X_te, y_tr, y_te = train_test_split(X.numpy(), y.numpy(), test_size=0.2, stratify=y.numpy(), random_state=42)

class TabDS(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_dl = DataLoader(TabDS(X_tr, y_tr), batch_size=16, shuffle=True)
test_dl = DataLoader(TabDS(X_te, y_te), batch_size=16)

# Modèles (fiche MLP §3.1 & §3.2)
class MLP_Custom(nn.Module):
    def __init__(self): super().__init__()
        self.fc1=nn.Linear(24,64); self.fc2=nn.Linear(64,32); self.out=nn.Linear(32,4)
    def forward(self,x): return self.out(nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x)))))

def init_xavier(m):
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

# Training
model = MLP_Custom().to(device); model.apply(init_xavier)
weights = 1.0 / (torch.bincount(torch.tensor(y_tr, dtype=torch.long)).float() + 1e-6)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
opt = optim.Adam(model.parameters(), lr=1e-3)

print("📦 Training MLP (stratified + weighted loss)...")
for ep in range(30):
    model.train()
    for xb, yb in train_dl:
        opt.zero_grad()
        loss = criterion(model(xb.to(device)), yb.to(device))
        loss.backward(); opt.step()

# Eval & Save
model.eval()
yt, yp = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        yt.extend(yb.tolist()); yp.extend(model(xb.to(device)).argmax(1).tolist())
print(classification_report(yt, yp, zero_division=0))

cm = confusion_matrix(yt, yp, labels=[0,1,2,3])
plt.figure(); plt.imshow(cm, cmap='Blues'); plt.colorbar()
plt.xticks(range(4),["ouvrir","fermer","rechercher","arreter"], rotation=45)
plt.yticks(range(4),["ouvrir","fermer","rechercher","arreter"])
plt.tight_layout(); plt.savefig("report/figures/mlp_confusion_matrix.png")
torch.save(model.state_dict(), "models/mlp_best.pth")
print("✅ Part I complete.")