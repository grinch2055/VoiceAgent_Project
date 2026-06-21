# -*- coding: utf-8 -*-
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)

DATA = "data/raw_voice/part3.pt"
src, tgt = torch.load(DATA, weights_only=True)
class SeqDS(Dataset):
    def __init__(self,s,t): self.s,self.t=s.long(),t.long()
    def __len__(self): return len(self.s)
    def __getitem__(self,i): return self.s[i], self.t[i]
dl = DataLoader(SeqDS(src,tgt), 8, True)

class Enc(nn.Module):
    def __init__(self,v,e,h): super().__init__(); self.emb=nn.Embedding(v,e); self.gru=nn.GRU(e,h,batch_first=True)
    def forward(self,x): return self.gru(self.emb(x))[1]
class Dec(nn.Module):
    def __init__(self,v,e,h,o): super().__init__(); self.emb=nn.Embedding(v,e); self.gru=nn.GRU(e,h,batch_first=True); self.fc=nn.Linear(h,o)
    def forward(self,x,h): o,h=self.gru(self.emb(x),h); return self.fc(o.squeeze(1)), h
class Seq2Seq(nn.Module):
    def __init__(self): super().__init__(); self.enc=Enc(50,32,64); self.dec=Dec(10,32,64,10)
    def forward(self,src,tgt_in):
        h=self.enc(src); outs=[]
        for t in range(tgt_in.size(1)): p,h=self.dec(tgt_in[:,t:t+1],h); outs.append(p)
        return torch.stack(outs,1), h

model=Seq2Seq().to(device); opt=optim.Adam(model.parameters(),1e-3); crit=nn.CrossEntropyLoss()
tf_ratio=0.7
print("📦 Training Seq2Seq (BPTT + Clipping + Teacher Forcing)...")
for ep in range(20):
    model.train(); tl=0
    for s,t in dl:
        s,t=s.to(device),t.to(device)
        t_in=torch.cat([torch.ones(s.size(0),1,dtype=torch.long,device=device), t[:,:-1]],1)
        opt.zero_grad(); logits,_=model(s,t_in)
        loss=crit(logits.reshape(-1,logits.size(-1)), t.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()
    print(f"Ep {ep+1}/20 | Loss: {tl/len(dl):.4f}")

ppl=math.exp(tl/len(dl)); print(f"📊 Perplexity: {ppl:.2f}")
torch.save(model.state_dict(), "models/seq2seq_gru.pth")
print("✅ Part III complete.")