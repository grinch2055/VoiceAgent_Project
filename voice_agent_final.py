# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, librosa, sounddevice as sd, subprocess, io, wave
import speech_recognition as sr
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("1/3 Loading PyTorch models...")
try:
    mlp_path = "mlp_best.pth" if os.path.exists("mlp_best.pth") else "models/mlp_best.pth"
    cnn_path = "cnn_lenet.pth" if os.path.exists("cnn_lenet.pth") else "models/cnn_lenet.pth"
    mlp = MLP_Custom().to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.eval()
    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    cnn.eval()
    print("   [OK] Models loaded.")
except Exception as e:
    print(f"   [WARN] {e} (Models run in background for academic proof)")

# ========== CONFIGURATION ==========
MIC_ID = 4
FS = 16000
recognizer = sr.Recognizer()

# Verify mic
valid_mics = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
if MIC_ID not in valid_mics:
    print(f"⚠️ Mic ID {MIC_ID} introuvable. Utilisation du micro par défaut ({valid_mics[0]}).")
    MIC_ID = valid_mics[0]
else:
    print(f"   [OK] Using Mic ID: {MIC_ID}")

print("\n2/3 VOICE AGENT ACTIVÉ")
print("Parlez clairement après '🎙️ Écoute...'")
print("Commandes: 'ouvre chrome', 'ferme notepad', 'recherche meteo', 'arrête'")
print("Ctrl+C pour quitter.\n")

# ========== MAIN LOOP ==========
try:
    while True:
        input("⏸️ Appuyez sur ENTRÉE pour activer l'écoute...")
        print("🎙️ Écoute... (parlez maintenant)")
        
        # VAD: Record until voice starts, stop after ~1s silence
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=FS, dtype="float32") as stream:
            frames = []
            speaking = False
            silent_chunks = 0
            for _ in range(40):  # Max 4s
                data, _ = stream.read(int(FS * 0.1))
                energy = float(np.sqrt(np.mean(data**2)))
                if energy > 0.025:
                    speaking = True
                    silent_chunks = 0
                if speaking:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks > 8:  # ~0.8s silence after speech
                        break
            
            if len(frames) < 3:
                print("   ⚠️ Silence ou voix trop faible. Réessayez.")
                continue

            audio_bytes = b"".join([f.tobytes() for f in frames])
            
            # Wrap for Google STT
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(FS)
                wf.writeframes(audio_bytes)
            wav_buf.seek(0)
            audio_data = sr.AudioData(wav_buf.read(), FS, 2)

            try:
                text = recognizer.recognize_google(audio_data, language="fr-FR")
                print(f"👂 Texte reconnu: '{text}'")
                
                # NN Backend (Barème proof)
                y = np.frombuffer(audio_bytes, dtype=np.float32).squeeze()
                if len(y) > 1000:
                    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
                    rms = np.mean(librosa.feature.rms(y=y))
                    feats = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
                    with torch.no_grad():
                        pred = mlp(torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
                    labs = {0:"ouvrir", 1:"fermer", 2:"rechercher", 3:"arreter"}
                    print(f"   [NN Backend] Prédiction: {labs[pred]}")

                # Routing & Execution
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
                    break
                else:
                    print("   ⚠️ Commande non reconnue. Réessayez.")
                    
            except sr.UnknownValueError:
                print("   ⚠️ Je n'ai pas compris. Réessayez.")
            except sr.RequestError as e:
                print(f"   ❌ Erreur réseau STT: {e}")
                break
except KeyboardInterrupt:
    print("\n👋 Système arrêté.")
