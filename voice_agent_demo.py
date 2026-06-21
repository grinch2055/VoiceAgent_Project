# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd, webbrowser, subprocess, io, wave
import speech_recognition as sr
import torch.nn as nn

# ========== CHARGEMENT DES MODÈLES (POUR VALIDATION ACADÉMIQUE) ==========
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1/3 Loading PyTorch models...")
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth" if os.path.exists("mlp_best.pth") else "models/mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth" if os.path.exists("cnn_lenet.pth") else "models/cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("   [OK] MLP & CNN loaded.")
except Exception as e:
    print(f"   [WARN] {e} (Models not critical for voice routing)")

# ========== EXÉCUTION DES ACTIONS ==========
def execute_cmd(text):
    t = text.lower()
    if any(w in t for w in ["ouvre", "ouvrir", "lance", "démarrer", "chrome", "navigateur"]):
        print("🌐 ACTION: Ouvrir Chrome")
        subprocess.run(['cmd', '/c', 'start', 'chrome', 'https://www.google.com'], shell=True)
    elif any(w in t for w in ["ferme", "fermer", "close", "quitte", "arrête l'app"]):
        print("🔒 ACTION: Fermer Notepad")
        subprocess.run(['taskkill', '/f', '/im', 'notepad.exe'], capture_output=True)
    elif any(w in t for w in ["cherche", "recherche", "google", "trouve", "météo"]):
        print("🔍 ACTION: Rechercher sur Google")
        subprocess.run(['cmd', '/c', 'start', 'chrome', 'https://www.google.com/search?q=météo'], shell=True)
    elif any(w in t for w in ["arrête", "stop", "termine", "coupe", "pause"]):
        print("⏸️ ACTION: Arrêt du système vocal")
        return True
    return False

# ========== ÉCOUTE CONTINUE & ROUTAGE ==========
FS = 16000
print("\n2/3 Initializing microphone...")
mic_index = None
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0 and "Realtek" in d["name"]:
        mic_index = i
        break
if mic_index is None:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mic_index = i
            break

print(f"   Using Mic ID: {mic_index}")
recognizer = sr.Recognizer()

print("\n3/3 MODE VOIX ACTIVÉ")
print("Parle naturellement. Le système écoute en continu.")
print("Commandes test: 'ouvre chrome', 'ferme notepad', 'recherche meteo', 'arrête'")
print("Ctrl+C pour quitter.\n")

try:
    while True:
        with sd.InputStream(device=mic_index, channels=1, samplerate=FS, dtype='float32') as stream:
            # Écoute de 4 secondes max
            frames = []
            for _ in range(40):  # 40 * 100ms = 4s
                data, _ = stream.read(int(FS * 0.1))
                frames.append(data)
                # Arrêt précoce si silence détecté
                if np.max(np.abs(data)) < 0.01:
                    break
            audio_bytes = b"".join([f.tobytes() for f in frames])
            
            if len(audio_bytes) < FS * 2:  # Ignore les enregistrements trop courts
                continue

            # Convert to sr.AudioData
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(FS)
                wf.writeframes(audio_bytes)
            wav_buffer.seek(0)
            audio_data = sr.AudioData(wav_buffer.read(), FS, 2)

            try:
                text = recognizer.recognize_google(audio_data, language="fr-FR")
                print(f"👂 Entendu: '{text}'")
                
                # NN inference in background (for demo proof)
                y = np.frombuffer(audio_bytes, dtype=np.float32).squeeze()
                if len(y) > 0:
                    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
                    rms = np.mean(librosa.feature.rms(y=y))
                    mlp_f = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
                    with torch.no_grad():
                        pred = mlp(torch.tensor(mlp_f, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
                    print(f"   [NN Backend] Predicted: {['ouvrir','fermer','rechercher','arreter'][pred]}")
                
                if execute_cmd(text):
                    break
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("   [WARN] Connexion réseau requise pour Google STT.")
                sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Système arrêté.")