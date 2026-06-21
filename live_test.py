# -*- coding: utf-8 -*-
import torch, librosa, numpy as np, sounddevice as sd, time, torch.nn as nn

# 1. Model Definitions (must exactly match your training scripts)
class MLP_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)
    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2,2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2,2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(4)
        )
    def forward(self, x):
        return self.net(x)

# 2. Configuration
FS = 16000
DUR = 2.0
SAMPLES = int(FS * DUR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTENT_MAP = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

# 3. Load Trained Models
print("?? Loading models...")
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()

    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("? Models loaded successfully.")
except Exception as e:
    print(f"? Failed to load models: {e}")
    print("?? Run training scripts first to generate .pth files.")
    exit()

# 4. Real-time Inference Loop
print("\n?? Live Voice Test Started")
print("Speak clearly for 2 seconds when prompted.")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        input("?? Press ENTER to record a 2-second command...")
        print("??? Recording...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32')
        sd.wait()

        y = audio.squeeze()
        
        # MLP Feature Extraction
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms = np.mean(librosa.feature.rms(y=y))
        mlp_feat = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])

        # CNN Feature Extraction
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100:
            mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
        else:
            mel = mel[:, :100]

        # Inference
        with torch.no_grad():
            mlp_out = mlp(torch.tensor(mlp_feat, dtype=torch.float32).unsqueeze(0).to(device))
            mlp_pred = mlp_out.argmax().item()
            mlp_conf = torch.nn.functional.softmax(mlp_out, dim=1)[0, mlp_pred].item()

            cnn_out = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
            cnn_pred = cnn_out.argmax().item()

        print(f"\n?? Results:")
        print(f"   MLP: {INTENT_MAP[mlp_pred]} ({mlp_conf:.1%})")
        print(f"   CNN: {INTENT_MAP[cnn_pred]}")
        
        if mlp_pred == cnn_pred:
            print(f"? AGREEMENT -> Intent: {INTENT_MAP[mlp_pred].upper()}")
        else:
            print("?? Models disagree. Try speaking more clearly or record again.")
        print("-" * 40)

except KeyboardInterrupt:
    print("\n?? Test stopped. Good job!")
