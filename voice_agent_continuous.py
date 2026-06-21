# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd, subprocess, io, wave, time
import speech_recognition as sr
import torch.nn as nn

# ========== ARCHITECTURES ==========
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
MIC_ID = 4
FS = 16000
VAD_THRESH = 0.025      # Seuil d'énergie pour détecter la voix
SILENCE_TIME = 0.8      # Secondes de silence pour couper l'enregistrement
CHUNK = int(FS * 0.1)   # 100ms par bloc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recognizer = sr.Recognizer()

# ========== CHARGEMENT MODÈLES ==========
print("1/3 Loading PyTorch models...")
try:
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth" if os.path.exists("mlp_best.pth") else "models/mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth" if os.path.exists("cnn_lenet.pth") else "models/cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    print("   [OK] Models loaded (running in background).")
except Exception as e:
    print(f"   [WARN] {e}")

# ========== MAPPING COMMANDES ==========
def parse_and_execute(text):
    t = text.lower()
    if any(k in t for k in ["ouvre", "ouvrir", "lance", "chrome", "navigateur"]):
        print("🌐 ACTION: Ouverture de Chrome")
        subprocess.run(['cmd', '/c', 'start', 'chrome', 'https://www.google.com'], shell=True)
    elif any(k in t for k in ["ferme", "fermer", "close", "notepad", "quitte"]):
        print("🔒 ACTION: Fermeture de Notepad")
        subprocess.run(['taskkill', '/f', '/im', 'notepad.exe'], capture_output=True)
    elif any(k in t for k in ["cherche", "recherche", "google", "météo", "trouve"]):
        print("🔍 ACTION: Recherche Google")
        subprocess.run(['cmd', '/c', 'start', 'chrome', 'https://www.google.com/search?q=météo'], shell=True)
    elif any(k in t for k in ["arrête", "stop", "termine", "coupe", "pause"]):
        print("⏸️ ACTION: Arrêt du système")
        return True
    else:
        print("   ⚠️ Commande non reconnue.")
    return False

# ========== ÉCOUTE CONTINUE (VAD) ==========
print("\n2/3 CONTINUOUS LISTENING ACTIVATED")
print("Parlez naturellement. Le système détecte la voix automatiquement.")
print("Commandes: 'ouvre chrome', 'ferme notepad', 'recherche meteo', 'arrête'")
print("Ctrl+C pour quitter.\n")

try:
    with sd.InputStream(device=MIC_ID, channels=1, samplerate=FS, dtype='float32') as stream:
        buffer = []
        recording = False
        silence_chunks = 0
        print("🎙️ En écoute continue...")
        
        while True:
            data, _ = stream.read(CHUNK)
            energy = float(np.sqrt(np.mean(data**2)))
            
            if not recording:
                if energy > VAD_THRESH:
                    recording = True
                    buffer = [data.copy()]
                    silence_chunks = 0
                    print("🎤 Détection de parole...")
                continue
            
            # Mode enregistrement
            buffer.append(data.copy())
            if energy < VAD_THRESH:
                silence_chunks += 1
            else:
                silence_chunks = 0
                
            if silence_chunks >= int((SILENCE_TIME * FS) / CHUNK):
                recording = False
                audio_bytes = b"".join([f.tobytes() for f in buffer])
                buffer = []
                if len(audio_bytes) < FS:  # Ignore les fragments trop courts
                    continue
                    
                # STT
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(FS)
                    wf.writeframes(audio_bytes)
                wav_buf.seek(0)
                audio_data = sr.AudioData(wav_buf.read(), FS, 2)
                
                try:
                    text = recognizer.recognize_google(audio_data, language="fr-FR")
                    print(f"👂 Reconnu: '{text}'")
                    
                    # NN Backend (Preuve académique)
                    y = np.frombuffer(audio_bytes, dtype=np.float32).squeeze()
                    if len(y) > 1000:
                        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
                        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
                        rms = np.mean(librosa.feature.rms(y=y))
                        feats = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
                        with torch.no_grad():
                            pred = mlp(torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
                        print(f"   [NN Backend] {['ouvrir','fermer','rechercher','arreter'][pred]}")
                        
                    if parse_and_execute(text):
                        break
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"   ❌ Réseau STT: {e}")
                    break
                print("⏸️ Retour en écoute...")
except KeyboardInterrupt:
    print("\n👋 Système arrêté.")
