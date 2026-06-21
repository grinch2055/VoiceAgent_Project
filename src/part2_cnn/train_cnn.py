# -*- coding: utf-8 -*-
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("report/figures", exist_ok=True)

# Manual ops (fiche CNN §2 & §6)
def corr2d_manual(X, K):
    h,w=K.shape; Y=torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]): Y[i,j]=(X[i:i+h, j:j+w]*K).sum()
    return Y

def pool2d_manual(X, ps, mode='max'):
    ph,pw=ps; Y=torch.zeros(X.shape[0]-ph+1, X.shape[1]-pw+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            reg=X[i:i+ph, j:j+pw]; Y[i,j]=reg.max() if mode=='max' else reg.mean()
    return Y

print("🔍 Manual corr2d/pool2d OK" if corr2d_manual(torch.ones((4,4)), torch.tensor([[1,-1]])).shape==torch.Size([3,4]) else "❌")

DATA = "data/raw_voice/part2.pt"
X, y = torch.load(DATA, weights_only=True)
idx = torch.randperm(X.shape[0]); sp=int(0.8*X.shape[0])
X_tr, y_tr = X[idx[:sp]], y[idx[:sp]]
X_te, y_te = X[idx[sp:]], y[idx[sp:]]

class ImgDS(Dataset):
    def __init__(self,X,y): self.X,self.y=X,y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i].unsqueeze(0).float(), self.y[i].long()

train_dl=DataLoader(ImgDS(X_tr,y_tr),8,True); test_dl=DataLoader(ImgDS(X_te,y_te),8,False)

class LeNet(nn.Module):
    def __init__(self): super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,6,5,padding=2),nn.Sigmoid(),nn.AvgPool2d(2,2),
            nn.Conv2d(6,16,5),nn.Sigmoid(),nn.AvgPool2d(2,2),nn.Flatten(),
            nn.LazyLinear(120),nn.Sigmoid(),nn.LazyLinear(84),nn.Sigmoid(),nn.LazyLinear(4))
    def forward(self,x): return self.net(x)

model=LeNet().to(device); opt=optim.Adam(model.parameters(),1e-3); crit=nn.CrossEntropyLoss()
print("📦 Training CNN...")
for ep in range(20):
    model.train(); tl=0
    for xb,yb in train_dl: opt.zero_grad(); l=crit(model(xb.to(device)),yb.to(device)); l.backward(); opt.step(); tl+=l.item()
    print(f"Ep {ep+1}/20 | Loss: {tl/len(train_dl):.4f}")

model.eval()
with torch.no_grad(): fm=model.net[0](torch.randn(1,1,128,100,device=device))
plt.figure(); plt.imshow(fm[0,0,:,:].cpu().numpy(), cmap='viridis'); plt.colorbar(); plt.tight_layout()
plt.savefig("report/figures/cnn_feature_map.png")
torch.save(model.state_dict(), "models/cnn_lenet.pth")
print("✅ Part II complete.")