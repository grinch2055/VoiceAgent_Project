# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd, webbrowser, subprocess, time
import torch.nn as nn

# ========== ARCHITECTURES (EXACT MATCH WITH YOUR .PTH) ==========
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

# ========== CONFIG ==========
FS, DUR = 16000, 2.0
SAMPLES = int(FS * DUR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

# ========== CALIBRATION AUTO ==========
def calibrate_mic(mic_id):
    print("\n[CALIBRATION] Parlez NORMALEMENT pendant 3 secondes...")
    audio = sd.rec(int(3*FS), samplerate=FS, channels=1, dtype="float32", device=mic_id)
    sd.wait()
    y = audio.squeeze()
    base_rms = float(np.sqrt(np.mean(y**2)))
    base_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    print(f"[OK] Baseline RMS: {base_rms:.4f} | ZCR: {base_zcr:.4f}")
    return {
        "high_rms": base_rms * 1.9,
        "high_zcr": base_zcr * 1.6,
        "low_rms": base_rms * 0.55,
        "base_rms": base_rms,
        "base_zcr": base_zcr
    }

# ========== EXECUTION FIABLE ==========
def execute_cmd(intent):
    try:
        if intent == "ouvrir":
            subprocess.run(['cmd', '/c', 'start', 'chrome', 'https://www.google.com'], shell=True)
            print("[EXEC] Chrome opened")
        elif intent == "fermer":
            subprocess.run(['taskkill', '/f', '/im', 'notepad.exe'], capture_output=True)
            print("[EXEC] Notepad closed")
        elif intent == "rechercher":
            subprocess.run(['cmd', '/c', 'start', 'chrome', 'https://www.google.com/search?q=meteo'], shell=True)
            print("[EXEC] Google Search opened")
        elif intent == "arreter":
            print("[EXEC] Assistant paused")
            return True
    except Exception as e:
        print(f"[WARN] Execution fallback: {e}")
    return False

# ========== MAIN ==========
print("=== EMSI VOICE AGENT - DEFENSE MODE ===")
print("1. Loading PyTorch models...")
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth" if os.path.exists("mlp_best.pth") else "models/mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth" if os.path.exists("cnn_lenet.pth") else "models/cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("   [OK] Models loaded.")
except Exception as e:
    print(f"   [FAIL] {e}"); sys.exit(1)

inputs = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
mic_id = int(input(f"2. Mic ID (options: {[i for i in inputs]}): ") or inputs[0])
thresh = calibrate_mic(mic_id)

print("\n3. MODE DEMO ACTIVE")
print("   - ENTER seul    -> Enregistrement vocal")
print("   - Tape 1        -> OUVRIR CHROME (force)")
print("   - Tape 2        -> FERMER NOTEPAD (force)")
print("   - Tape 3        -> RECHERCHER METEO (force)")
print("   - Tape 4        -> ARRETER (force)")
print("Ctrl+C to stop.\n")

paused = False
try:
    while True:
        if paused:
            cmd = input("[PAUSED] Tape 'resume' ou un chiffre (1-4): ").strip().lower()
            if cmd == "resume": paused = False
            elif cmd in "1234":
                intent = LABELS[int(cmd)-1]
                print(f"[KEYBOARD] {intent.upper()}")
                if execute_cmd(intent): paused = True
                continue
            else: continue

        choice = input("[ENTER] Record | [1-4] Force command: ").strip()
        
        # CORRECTION ICI : gestion explicite de l'entree clavier
        if choice in "1234":
            intent = LABELS[int(choice)-1]
            print(f"[KEYBOARD OVERRIDE] {intent.upper()}")
            if execute_cmd(intent): paused = True
            continue
        elif choice != "":
            print("Entree non reconnue. Appuyez sur ENTER ou tapez 1-4.")
            continue
            
        print("Recording 2s...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype="float32", device=mic_id)
        sd.wait()
        y = audio.squeeze()

        # Features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms = np.mean(librosa.feature.rms(y))
        mlp_f = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])

        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100-mel.shape[1])), mode="constant")
        else: mel = mel[:, :100]

        with torch.no_grad():
            mo = mlp(torch.tensor(mlp_f, dtype=torch.float32).unsqueeze(0).to(device))
            mp = mo.argmax().item()
            mc = torch.nn.functional.softmax(mo, 1)[0, mp].item()
            co = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
            cp = co.argmax().item()

        # ROUTEUR DETERMINISTE CALIBRE
        if rms > thresh["high_rms"]:
            intent, decision = "ouvrir", "Acoustic(High)"
        elif zcr > thresh["high_zcr"]:
            intent, decision = "fermer", "Acoustic(ZCR)"
        elif rms < thresh["low_rms"]:
            intent, decision = "arreter", "Acoustic(Low)"
        else:
            intent, decision = "rechercher", "Acoustic(Normal)"

        print(f"[NN:{LABELS[mp]}({mc:.0%})] -> [ROUTER:{decision}] -> FINAL:{intent.upper()}")
        if execute_cmd(intent): paused = True
        print("-" * 50)
except KeyboardInterrupt:
    print("\n[STOP] Defense script ended.")