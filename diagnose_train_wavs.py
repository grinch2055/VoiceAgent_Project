# -*- coding: utf-8 -*-
import os, glob, torch, numpy as np, librosa
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP Model (properly indented)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)
    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

# CNN Model (properly indented)
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Flatten(), nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(), nn.LazyLinear(4)
        )
    def forward(self, x):
        return self.net(x)

# Load models
mlp = MLP().to(device)
mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
mlp.eval()

cnn = LeNet().to(device)
cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
cnn.eval()

LABELS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}
wavs = sorted(glob.glob("data/raw_voice/*.wav"))[:4]

print("🔍 Testing on training WAVs...")
for f in wavs:
    y, sr = librosa.load(f, sr=16000)
    
    # MLP Features (24D exact)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    feat = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
    
    # CNN Spectrogram (128x100 exact)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    if mel.shape[1] < 100:
        mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :100]
    
    # Extract label from filename (e.g., ouvrir_0.wav -> 0)
    basename = os.path.basename(f)
    try:
        true_label = int(basename.split("_")[-1].replace(".wav", "")) // 10
    except:
        true_label = -1
    
    with torch.no_grad():
        mlp_p = mlp(torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
        cnn_p = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)).argmax().item()
    
    print(f"{basename} | True: ~{true_label} | MLP: {LABELS[mlp_p]} | CNN: {LABELS[cnn_p]}")

print("\n✅ Diagnostic complete.")