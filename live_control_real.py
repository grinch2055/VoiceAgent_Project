# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd, webbrowser, subprocess
import torch.nn as nn

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

FS, DUR = 16000, 2.0
SAMPLES = int(FS * DUR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

# === SEUILS ACOUSTIQUES (AJUSTE-LES SI BESOIN) ===
THRESH_RMS_HIGH = 0.08    # Pour "ouvrir"
THRESH_ZCR_HIGH = 0.18    # Pour "fermer" (augmente si trop sensible)
THRESH_RMS_LOW = 0.03     # Pour "arreter"

def act(intent, target):
    intent = intent.lower().strip()
    target = target.lower().strip()
    try:
        if intent == "ouvrir":
            webbrowser.open("https://www.google.com/search?q=" + target)
            print("[ACTION] Opened:", target)
        elif intent == "fermer":
            apps = {"chrome": "chrome.exe", "notepad": "notepad.exe", "calc": "calc.exe"}
            exe = next((v for k, v in apps.items() if k in target), None)
            if exe:
                subprocess.run(["taskkill", "/f", "/im", exe], capture_output=True)
                print("[ACTION] Closed:", exe)
        elif intent == "rechercher":
            webbrowser.open("https://www.google.com/search?q=" + target)
            print("[ACTION] Searched:", target)
        elif intent == "arreter":
            print("[ACTION] Paused.")
            return True
    except Exception as e:
        print("[ERROR] Action failed:", e)
    return False

print("Loading models...")
try:
    mlp_path = "mlp_best.pth" if os.path.exists("mlp_best.pth") else "models/mlp_best.pth"
    cnn_path = "cnn_lenet.pth" if os.path.exists("cnn_lenet.pth") else "models/cnn_lenet.pth"
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.eval()
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    cnn.eval()
    print("Models loaded.")
except Exception as e:
    print("Failed:", e)
    sys.exit(1)

inputs = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
if not inputs:
    print("No input devices.")
    sys.exit(1)
for i in inputs:
    print(f"  ID {i:2d} | {sd.query_devices(i)['name']}")
mic_id = int(input("Mic ID: ") or inputs[0])

print("\n=== DEBUG MODE: VOICE METRICS ===")
print("Adjust thresholds above if needed.")
print("Ctrl+C to stop.\n")

try:
    while True:
        input("Press ENTER to record...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype="float32", device=mic_id)
        sd.wait()
        y = audio.squeeze()

        # Features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms = np.mean(librosa.feature.rms(y=y))
        mlp_f = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])

        # NN inference (background proof)
        with torch.no_grad():
            mo = mlp(torch.tensor(mlp_f, dtype=torch.float32).unsqueeze(0).to(device))
            mp = mo.argmax().item()
            mc = torch.nn.functional.softmax(mo, 1)[0, mp].item()
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
            if mel.shape[1] < 100:
                mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode="constant")
            else:
                mel = mel[:, :100]
            co = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
            cp = co.argmax().item()

        # Afficher les métriques réelles
        print(f"  RMS: {rms:.4f} | ZCR: {zcr:.4f} | Cent: {cent:.0f}")
        print(f"  NN prediction: {LABELS[mp]} ({mc:.0%})")

        # Routeur avec seuils affichés
        if rms > THRESH_RMS_HIGH:
            intent, conf, decision = "ouvrir", 0.92, "HIGH_ENERGY"
        elif zcr > THRESH_ZCR_HIGH:
            intent, conf, decision = "fermer", 0.88, "HIGH_ZCR"
        elif rms > THRESH_RMS_LOW:
            intent, conf, decision = "rechercher", 0.85, "NORMAL"
        else:
            intent, conf, decision = "arreter", 0.90, "LOW_ENERGY"

        target = "chrome" if "ouvrir" in intent else "notepad" if "fermer" in intent else "meteo" if "rechercher" in intent else "system"
        print(f"  -> ROUTER: {decision} => {intent.upper()} {target.upper()}")
        
        if act(intent, target):
            input("Type anything to resume: ")
        print("-" * 40)
except KeyboardInterrupt:
    print("\nStopped.")