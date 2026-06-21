# -*- coding: utf-8 -*-
import os, glob, torch, librosa, numpy as np, torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PART I: MLP ----------------
class MLP_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)
    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

# ---------------- PART II: CNN ----------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(), nn.LazyLinear(84), nn.Sigmoid(), nn.LazyLinear(4)
        )
    def forward(self, x):
        return self.net(x)

# Load models
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()
    
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("? Models loaded successfully.")
except Exception as e:
    print(f"? Error loading models: {e}")
    print("?? Ensure you have run the training scripts first.")
    exit()

INTENT_MAP = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

# Find a test wav file
wav_files = glob.glob("data/raw_voice/*.wav")
if not wav_files:
    print("? No voice recordings found in data/raw_voice/. Please record your voice first.")
else:
    test_wav = wav_files[0]
    print(f"\n?? Testing with real recording: {os.path.basename(test_wav)}")
    
    # Extract audio features
    y, sr = librosa.load(test_wav, sr=16000)
    
    # MLP Features (13 MFCC + ZCR + Centroid + RMS + padding to 24)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    mlp_feat = np.concatenate([mfccs, [zcr, cent, rms] + [0.0]*8])
    
    # CNN Spectrogram (128x100)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    if mel.shape[1] < 100:
        mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :100]
    
    # Inference
    with torch.no_grad():
        mlp_tensor = torch.tensor(mlp_feat, dtype=torch.float32).unsqueeze(0).to(device)
        mlp_out = mlp(mlp_tensor)
        mlp_pred = mlp_out.argmax().item()
        
        cnn_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        cnn_out = cnn(cnn_tensor)
        cnn_pred = cnn_out.argmax().item()
    
    # Results
    intent = INTENT_MAP.get(mlp_pred, "inconnu")
    target = os.path.basename(test_wav).split("_")[0]
    
    # Calculate probability
    probs = torch.nn.functional.softmax(mlp_out, dim=1)
    max_prob = probs[0, mlp_pred].item()
    
    print(f"?? MLP predicts: {intent} (confidence: {max_prob:.2%})")
    print(f"?? CNN predicts: {INTENT_MAP.get(cnn_pred, 'inconnu')}")
    print(f"?? Action mapper would execute: {intent.upper()} {target.upper()}")
    
    if mlp_pred == cnn_pred:
        print("\n? Pipeline test complete: Models agree on intent!")
    else:
        print("\n?? Pipeline test complete: Models disagree (normal with small datasets).")
