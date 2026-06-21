# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd
import torch.nn as nn

# 1. Model Definitions (EXACT match with training)
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
            nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Flatten(), nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(), nn.LazyLinear(4)
        )
    def forward(self, x):
        return self.net(x)

# 2. Configuration
FS, DUR = 16000, 2.0
SAMPLES = int(FS * DUR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTENT_MAP = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}
ACTION_MAP = {
    "ouvrir": lambda t: print(f"   [DEMO] Would open: {t}"),
    "fermer": lambda t: print(f"   [DEMO] Would close: {t}"),
    "rechercher": lambda t: print(f"   [DEMO] Would search: {t}"),
    "arreter": lambda t: print(f"   [DEMO] Would stop system")
}

# 3. Load Trained Models
print("📦 Loading trained models...")
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()

    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("✅ Models loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    print("⚠️ Run training scripts first to generate .pth files.")
    sys.exit(1)

# 4. Microphone Selection
print("🎤 Available microphones:")
input_devs = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
if not input_devs:
    print("❌ No input devices found. Check Windows Sound settings.")
    sys.exit(1)
for i in input_devs:
    print(f"  ID {i:2d} | {sd.query_devices(i)['name']}")

try:
    dev_str = input("\nEnter microphone ID to use (e.g., 1, 5, 9) or press ENTER for default: ").strip()
    mic_id = int(dev_str) if dev_str else input_devs[0]
except:
    mic_id = input_devs[0]

# Quick validation test
print(f"\n🧪 Testing microphone ID {mic_id}... Speak for 2 seconds.")
try:
    test_audio = sd.rec(int(2*FS), samplerate=FS, channels=1, dtype='float32', device=mic_id)
    sd.wait()
    rms = float(np.sqrt(np.mean(test_audio**2)))
    if rms < 0.02:
        print(f"⚠️ Volume too low (RMS={rms:.4f}). Check Windows Sound > Recording > Levels.")
        sys.exit(1)
    print(f"✅ Microphone OK (RMS={rms:.3f}).\n")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# 5. Live Inference Loop
print("🎙️ LIVE VOICE TEST STARTED")
print("Say one of: 'ouvrir chrome', 'fermer notepad', 'rechercher meteo', 'arreter'")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        input("⏸️ Press ENTER to record a 2-second command...")
        print("🎙️ Recording...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32', device=mic_id)
        sd.wait()
        y = audio.squeeze()

        # MLP Features (24D)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms_feat = np.mean(librosa.feature.rms(y=y))
        mlp_feat = np.concatenate([mfcc, [zcr, cent, rms_feat] + [0.0]*8])

        # CNN Spectrogram (128x100)
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

        # Results
        intent = INTENT_MAP[mlp_pred]
        print(f"\n📊 Results:")
        print(f"   MLP: {intent} ({mlp_conf:.1%})")
        print(f"   CNN: {INTENT_MAP[cnn_pred]}")

        if mlp_pred == cnn_pred:
            print(f"✅ AGREEMENT -> Intent: {intent.upper()}")
            # Safe demo action (no real app launch)
            target = "chrome" if "ouvrir" in intent else "notepad" if "fermer" in intent else "météo" if "rechercher" in intent else "system"
            ACTION_MAP[intent](target)
        else:
            print("⚠️ Models disagree. Try speaking more clearly or record again.")
        print("-" * 50)

except KeyboardInterrupt:
    print("\n👋 Test stopped. Good job!")