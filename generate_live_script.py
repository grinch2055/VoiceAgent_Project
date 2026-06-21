# -*- coding: utf-8 -*-
script_content = '''# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd, webbrowser, subprocess
import torch.nn as nn

# ========== MODÈLES (MATCH EXACT WITH SAVED WEIGHTS) ==========
# MLP: nn.Sequential (keys: 0,2,4 for Linear layers)
def create_mlp():
    return nn.Sequential(
        nn.Linear(24, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 4)
    )

# CNN: LeNet exact
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

# ========== CONFIG ==========
FS, DUR = 16000, 2.0
SAMPLES = int(FS * DUR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

# ========== ACTIONS ==========
def act(intent, target):
    intent = intent.lower().strip()
    target = target.lower().strip()
    try:
        if intent == "ouvrir":
            webbrowser.open("https://www.google.com/search?q=" + target)
            print(f"?? Opened: {target}")
        elif intent == "fermer":
            apps = {"chrome": "chrome.exe", "notepad": "notepad.exe", "calc": "calc.exe"}
            exe = next((v for k, v in apps.items() if k in target), None)
            if exe:
                subprocess.run(["taskkill", "/f", "/im", exe], capture_output=True)
                print(f"?? Closed {exe}")
        elif intent == "rechercher":
            webbrowser.open("https://www.google.com/search?q=" + target)
            print(f"?? Searched: {target}")
        elif intent == "arreter":
            print("?? Paused.")
            return True
    except Exception as e:
        print(f"? Action failed: {e}")
    return False

# ========== LOAD MODELS ==========
print("?? Loading models...")
try:
    mlp = create_mlp().to(device)
    mlp.load_state_dict(torch.load("models/mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()
    
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("models/cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("? Models loaded successfully.")
except Exception as e:
    print(f"? Failed: {e}")
    sys.exit(1)

# ========== MIC SELECTION ==========
inputs = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
if not inputs:
    print("? No input devices found.")
    sys.exit(1)
for i in inputs:
    print(f"  ID {i:2d} | {sd.query_devices(i)[\"name\"]}")
mic_id = int(input("?? Mic ID: ") or inputs[0])
print(f"? Ready. Speak clearly. Ctrl+C to stop.\\n")

# ========== MAIN LOOP ==========
try:
    while True:
        input("?? Press ENTER to record...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype="float32", device=mic_id)
        sd.wait()
        y = audio.squeeze()
        
        # MLP features (24D)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms = np.mean(librosa.feature.rms(y=y))
        mlp_f = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
        
        # CNN features (128x100)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100:
            mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode="constant")
        else:
            mel = mel[:, :100]
        
        # Inference
        with torch.no_grad():
            mo = mlp(torch.tensor(mlp_f, dtype=torch.float32).unsqueeze(0).to(device))
            mp = mo.argmax().item()
            mc = torch.nn.functional.softmax(mo, 1)[0, mp].item()
            co = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
            cp = co.argmax().item()
            cc = torch.nn.functional.softmax(co, 1)[0, cp].item()
        
        # Decision by max confidence (fallback for small dataset)
        if mc > cc:
            intent, conf, decision = LABELS[mp], mc, "MLP"
        else:
            intent, conf, decision = LABELS[cp], cc, "CNN"
        
        target = "chrome" if "ouvrir" in intent else "notepad" if "fermer" in intent else "meteo" if "rechercher" in intent else "system"
        
        if conf > 0.55:
            print(f"? DECISION ({decision}): {intent.upper()} {target.upper()} ({conf:.1%})")
            if act(intent, target):
                input("?? Type anything to resume: ")
        else:
            print(f"?? Low confidence ({conf:.1%}). Retry.")
except KeyboardInterrupt:
    print("\\n?? Stopped.")
'''

with open("live_control_real.py", "w", encoding="utf-8") as f:
    f.write(script_content)
print("? live_control_real.py generated successfully.")
