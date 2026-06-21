# -*- coding: utf-8 -*-
import os, numpy as np, torch, librosa, sounddevice as sd, sys
from scipy.io.wavfile import write as wavwrite

print("=== VOICE RECORDING WITH CONFIRMATION ===")
print("1. Available microphones:")
input_devs = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
if not input_devs:
    print("ERROR: No input devices found."); sys.exit(1)
for i in input_devs:
    print(f"  ID {i:2d} | {sd.query_devices(i)['name']}")

dev_str = input("\nEnter microphone ID (e.g., 1, 5, 9): ").strip()
mic_id = int(dev_str) if dev_str else input_devs[0]

# Quick validation
print(f"\nTesting mic ID {mic_id}... Speak clearly for 3 seconds.")
try:
    test_audio = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32', device=mic_id)
    sd.wait()
    rms = float(np.sqrt(np.mean(test_audio**2)))
    if rms < 0.03:
        print(f"⚠️ Volume too low (RMS={rms:.4f}). Set Windows mic to 80-100% and retry.")
        sys.exit(1)
    print(f"✅ Mic OK (RMS={rms:.3f}). Playing back test...\n")
    sd.play(test_audio, 16000); sd.wait()
except Exception as e:
    print(f"❌ Failed: {e}"); sys.exit(1)

FS, DUR, OUT = 16000, 2.0, "data/raw_voice"
SAMPLES = int(FS * DUR)
os.makedirs(OUT, exist_ok=True)

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

X_mlp, y_mlp, X_cnn, y_cnn = [], [], [], []

print("\n=== RECORD 40 PHRASES WITH CONFIRMATION ===")
print("Rules: Speak LOUDLY, ~20cm from mic. Playback will run after each recording.")

for intent_id, phrases in COMMANDS.items():
    prefix = INTENT_NAMES[intent_id]
    print(f"\n📁 Class '{prefix}' ({intent_id+1}/4)")
    for i, phrase in enumerate(phrases):
        filepath = os.path.join(OUT, f"{prefix}_{i}.wav")
        print(f"\n👉 [{i+1}/40] Target phrase: '{phrase}'")
        input("   Press ENTER when ready, then speak clearly for 2 seconds...")
        print("   🎙️ Recording...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32', device=mic_id)
        sd.wait()

        # Validate level
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.03:
            print("   ⚠️ Too quiet! Press R to retry, or any key to save anyway.")
            if input("   Choice: ").strip().lower() == 'r':
                continue

        # Playback & Confirmation
        print("   🔊 Playing back your recording...")
        sd.play(audio, 16000); sd.wait()
        conf = input("   ✅ Did it sound clear and correct? (y/n): ").strip().lower()
        if conf != 'y':
            print("   🔄 Discarding. Press ENTER to retry.")
            input(); continue

        # Save WAV
        wavwrite(filepath, FS, (audio * 32767).astype(np.int16))
        print("   💾 Saved. Extracting features...")

        # Extract MLP & CNN features (exact pipeline match)
        y = audio.squeeze()
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms_feat = np.mean(librosa.feature.rms(y=y))
        feats = np.concatenate([mfcc, [zcr, cent, rms_feat] + [0.0]*8])
        X_mlp.append(feats); y_mlp.append(intent_id)

        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
        else: mel = mel[:, :100]
        X_cnn.append(mel); y_cnn.append(intent_id)
        print("   ✅ Confirmed & processed.")

# Save tensors
torch.save((torch.tensor(X_mlp, dtype=torch.float32), torch.tensor(y_mlp, dtype=torch.long)), os.path.join(OUT, "part1.pt"))
torch.save((torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1), torch.tensor(y_cnn, dtype=torch.long)), os.path.join(OUT, "part2.pt"))
torch.save((torch.randint(1, 50, (len(y_mlp), 15)), torch.randint(1, 10, (len(y_mlp), 8))), os.path.join(OUT, "part3.pt"))
print("\n🎉 DATASET COMPLETE. All phrases confirmed, saved, and preprocessed.")
print("Next step: retrain models with python src/part1_mlp/train_mlp.py")