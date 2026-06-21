# -*- coding: utf-8 -*-
import torch, numpy as np, librosa, glob, os
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# 1. Model Definitions (EXACT match with training)
class MLP_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)
    def forward(self, x): return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2,2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2,2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(), nn.LazyLinear(84), nn.Sigmoid(), nn.LazyLinear(4)
        )
    def forward(self, x): return self.net(x)

# 2. Load Trained Weights
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()

    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("✅ Models loaded successfully.\n")
except Exception as e:
    print(f"❌ Model loading failed: {e}\nRun training scripts first."); exit()

LABELS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}
wav_files = sorted(glob.glob("data/raw_voice/*.wav"))

if not wav_files:
    print("⚠️ No .wav files found in data/raw_voice/")
else:
    print(f"🎧 Testing {min(3, len(wav_files))} samples from dataset:\n" + "-"*40)
    for f in wav_files[:3]:
        y, sr = librosa.load(f, sr=16000)
        
        # MLP Features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        mlp_feat = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
        
        # CNN Spectrogram
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
        else: mel = mel[:, :100]
        
        # Inference
        with torch.no_grad():
            mlp_pred = mlp(torch.tensor(mlp_feat, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
            cnn_pred = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)).argmax().item()
            
        status = "✅ AGREE" if mlp_pred == cnn_pred else "⚠️ DISAGREE"
        print(f"File: {os.path.basename(f)}")
        print(f"  MLP -> {LABELS[mlp_pred]} | CNN -> {LABELS[cnn_pred]} | {status}")
        print("-" * 40)
    print("\n🎉 Offline test complete. Models are functional.")