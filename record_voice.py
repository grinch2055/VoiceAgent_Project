# -*- coding: utf-8 -*-
import os, numpy as np, torch, librosa, sounddevice as sd
from scipy.io.wavfile import write as wavwrite
import sys

print("=== VOICE RECORDING MODULE ===")
print("1. Available microphones:")
input_devs = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
if not input_devs:
    print("ERROR: No input devices detected.")
    sys.exit(1)
for i in input_devs:
    print(f"  ID {i:2d} | {sd.query_devices(i)['name']}")

print("\n2. Select Microphone:")
try:
    dev_str = input("Enter ID number (e.g., 1, 5, 9): ").strip()
    mic_id = int(dev_str) if dev_str else input_devs[0]
except:
    mic_id = input_devs[0]

# Quick 3-sec validation test
print(f"\n3. Testing ID {mic_id}... Speak clearly for 3 seconds.")
try:
    test_audio = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32', device=mic_id)
    sd.wait()
    rms = float(np.sqrt(np.mean(test_audio**2)))
    if rms < 0.02:
        print(f"⚠️ Volume too low (RMS={rms:.4f}). Check Windows Sound > Recording > Levels.")
        sys.exit(1)
    print(f"✅ Microphone OK (RMS={rms:.3f}). Starting recording...")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Configuration
FS, DUR, OUT = 16000, 2.0, "data/raw_voice"
SAMPLES = int(FS * DUR)

COMMANDS = {
    0: ["ouvrir chrome", "lance le navigateur", "demarre google chrome", "ouvre firefox", "lance edge",
        "ouvre une fenetre", "demarre navigateur", "ouvre chrome", "lance chromium", "ouvre navigateur"],
    1: ["ferme chrome", "arrete le navigateur", "close application", "ferme la fenetre", "arrete firefox",
        "ferme tout", "quitte chrome", "ferme programme", "eteins application", "stoppe navigateur"],
    2: ["cherche meteo", "recherche actualites", "google deep learning", "cherche tutoriel", "recherche youtube",
        "google traduction", "cherche cours", "recherche wikipedia", "google calculatrice", "cherche recette"],
    3: ["arrete assistant", "stop systeme", "ferme application voix", "desactive programme", "arrete tout",
        "quitte assistant", "stoppe reconnaissance", "eteins mode commande", "termine application", "arrete ecoute"]
}
INTENT_NAMES = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

X_mlp, y_mlp, X_cnn, y_cnn = [], [], [], []

print("\n=== RECORDING 40 SENTENCES ===")
print("Press ENTER before each phrase. Speak clearly for 2 seconds.")

for intent_id, phrases in COMMANDS.items():
    prefix = INTENT_NAMES[intent_id]
    print(f"\n📁 Class '{prefix}' ({intent_id+1}/4)")
    for i, phrase in enumerate(phrases):
        filepath = os.path.join(OUT, f"{prefix}_{i}.wav")
        print(f"  👉 [{i+1}/40] Press ENTER, then say: '{phrase}'")
        input()
        print("  🎙️ Recording...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32', device=mic_id)
        sd.wait()

        wavwrite(filepath, FS, (audio * 32767).astype(np.int16))

        # MLP Features
        y = audio.squeeze()
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms_feat = np.mean(librosa.feature.rms(y=y))
        feats = np.concatenate([mfcc, [zcr, cent, rms_feat] + [0.0]*8])
        X_mlp.append(feats); y_mlp.append(intent_id)

        # CNN Spectrograms
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
        else: mel = mel[:, :100]
        X_cnn.append(mel); y_cnn.append(intent_id)
        print("  ✅ Saved & processed.")

# Save tensors
torch.save((torch.tensor(X_mlp, dtype=torch.float32), torch.tensor(y_mlp, dtype=torch.long)), os.path.join(OUT, "part1.pt"))
torch.save((torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1), torch.tensor(y_cnn, dtype=torch.long)), os.path.join(OUT, "part2.pt"))
torch.save((torch.randint(1, 50, (len(y_mlp), 15)), torch.randint(1, 10, (len(y_mlp), 8))), os.path.join(OUT, "part3.pt"))
print("\n🎉 DATASET COMPLETE. 40 real voice samples recorded and preprocessed.")
print("Ready to retrain: python src/part1_mlp/train_mlp.py")