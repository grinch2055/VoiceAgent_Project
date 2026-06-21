# -*- coding: utf-8 -*-
import os, numpy as np, torch, librosa, sounddevice as sd
from scipy.io.wavfile import write as wavwrite

# 1. Check Microphones
print("Available Microphones:")
print(sd.query_devices())
dev = input("\nEnter Microphone ID to use (e.g. 0, 1) or leave empty for default: ")
if dev.strip(): sd.default.device[0] = int(dev)

FS, DUR, OUT = 16000, 2.0, "data/raw_voice"
os.makedirs(OUT, exist_ok=True)
SAMPLES = int(FS * DUR)

# 2. Define Commands (Unaccented to prevent crashes)
COMMANDS = {
    0: ["ouvrir chrome", "lance navigateur", "demarre google chrome", "ouvre firefox", "lance edge", 
        "ouvre fenetre", "demarre navigateur", "ouvre chrome", "lance chromium", "ouvre navigateur"],
    1: ["ferme chrome", "arrete navigateur", "close application", "ferme fenetre", "arrete firefox",
        "ferme tout", "quitte chrome", "ferme programme", "eteins application", "stoppe navigateur"],
    2: ["cherche meteo", "recherche actualites", "google deep learning", "cherche tutoriel", "recherche youtube",
        "google traduction", "cherche cours", "recherche wikipedia", "google calculatrice", "cherche recette"],
    3: ["arrete assistant", "stop systeme", "ferme application voix", "desactive programme", "arrete tout",
        "quitte assistant", "stoppe reconnaissance", "eteins mode commande", "termine application", "arrete ecoute"]
}
INTENT_NAMES = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

def record_and_check():
    audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01:
        print("Warning: Volume too low! Check mic and speak louder.")
    return audio.squeeze(), rms

X_mlp, y_mlp, X_cnn, y_cnn = [], [], [], []

print("\nStarting Recording. Press ENTER then speak clearly.")
for intent_id, phrases in COMMANDS.items():
    prefix = INTENT_NAMES[intent_id]
    print(f"\nClass '{prefix}' ({intent_id+1}/4)")
    for i, phrase in enumerate(phrases):
        filepath = os.path.join(OUT, f"{prefix}_{i}.wav")
        input(f"  -> [{i+1}/{len(phrases)}] Press ENTER, then say: '{phrase}'")
        audio, rms = record_and_check()
        wavwrite(filepath, FS, (audio * 32767).astype(np.int16))
        
        # MLP Feature Extraction
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=FS))
        rms_feat = np.mean(librosa.feature.rms(y=audio))
        feats = np.concatenate([mfcc, [zcr, cent, rms_feat] + [0.0]*8])
        X_mlp.append(feats); y_mlp.append(intent_id)
        
        # CNN Spectrogram Extraction
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
        else: mel = mel[:, :100]
        X_cnn.append(mel); y_cnn.append(intent_id)
        print(f"   [OK] Saved (RMS: {rms:.3f})")

# Save Data
torch.save((torch.tensor(X_mlp, dtype=torch.float32), torch.tensor(y_mlp, dtype=torch.long)), os.path.join(OUT, "part1.pt"))
torch.save((torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1), torch.tensor(y_cnn, dtype=torch.long)), os.path.join(OUT, "part2.pt"))
torch.save((torch.randint(1, 50, (len(y_mlp), 15)), torch.randint(1, 10, (len(y_mlp), 8))), os.path.join(OUT, "part3.pt"))
print("\nDataset successfully generated. Ready for retraining.")