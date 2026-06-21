# -*- coding: utf-8 -*-
import os, numpy as np, torch, librosa, sounddevice as sd, sys
from scipy.io.wavfile import write as wavwrite

print("🎙️ ENREGISTREMENT VOCALE VALIDÉ (EMSI DLM)")
inputs = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
if not inputs: print("❌ Aucun micro détecté."); sys.exit(1)
for i in inputs: print(f"  ID {i:2d} | {sd.query_devices(i)['name']}")

dev = input("\nID du micro : ").strip()
mic_id = int(dev) if dev else inputs[0]

print(f"\n🧪 Test mic ID {mic_id}... Parle 3s.")
test = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32', device=mic_id)
sd.wait()
rms = float(np.sqrt(np.mean(test**2)))
if rms < 0.03: print("⚠️ Volume trop faible. Vérifie Windows Son > Entrée."); sys.exit(1)
print(f"✅ Mic OK (RMS={rms:.3f}).\n")

FS, DUR, OUT = 16000, 2.0, "data/raw_voice"
SAMPLES = int(FS * DUR)
os.makedirs(OUT, exist_ok=True)

COMMANDS = {
    0: ["ouvrir chrome", "lance navigateur", "demarre google chrome", "ouvre firefox", "lance edge", "ouvre fenetre", "demarre navigateur", "ouvre chrome", "lance chromium", "ouvre navigateur"],
    1: ["ferme chrome", "arrete navigateur", "close application", "ferme fenetre", "arrete firefox", "ferme tout", "quitte chrome", "ferme programme", "eteins application", "stoppe navigateur"],
    2: ["cherche meteo", "recherche actualites", "google deep learning", "cherche tutoriel", "recherche youtube", "google traduction", "cherche cours", "recherche wikipedia", "google calculatrice", "cherche recette"],
    3: ["arrete assistant", "stop systeme", "ferme application voix", "desactive programme", "arrete tout", "quitte assistant", "stoppe reconnaissance", "eteins mode commande", "termine application", "arrete ecoute"]
}
NAMES = {0:"ouvrir", 1:"fermer", 2:"rechercher", 3:"arreter"}

X_mlp, y_mlp, X_cnn, y_cnn = [], [], [], []

print("📁 Recording 40 phrases with playback confirmation...")
for intent_id, phrases in COMMANDS.items():
    print(f"\nClass '{NAMES[intent_id]}' ({intent_id+1}/4)")
    for i, phrase in enumerate(phrases):
        filepath = os.path.join(OUT, f"{NAMES[intent_id]}_{i}.wav")
        print(f"  👉 [{i+1}/40] Target: '{phrase}'")
        input("   Press ENTER, speak clearly for 2s...")
        audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32', device=mic_id)
        sd.wait()
        
        y = audio.squeeze()
        rms_val = float(np.sqrt(np.mean(y**2)))
        if rms_val < 0.02:
            print("   ⚠️ Too quiet. Type 'r' to retry, any key to save.")
            if input("   Choice: ").strip().lower()=='r': continue

        print("   🔊 Playback...")
        sd.play(audio, 16000); sd.wait()
        if input("   ✅ Clear? (y/n): ").strip().lower() != 'y':
            print("   🔄 Discarding."); input("   Press ENTER to retry."); continue

        wavwrite(filepath, FS, (y * 32767).astype(np.int16))
        
        # MLP Features (24D)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
        rms_feat = np.mean(librosa.feature.rms(y=y))
        feats = np.concatenate([mfcc, [zcr, cent, rms_feat] + [0.0]*8])
        X_mlp.append(feats); y_mlp.append(intent_id)

        # CNN Spectrogram (128x100)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100-mel.shape[1])), mode='constant')
        else: mel = mel[:, :100]
        X_cnn.append(mel); y_cnn.append(intent_id)
        print("   ✅ Saved & processed.")

torch.save((torch.tensor(X_mlp, dtype=torch.float32), torch.tensor(y_mlp, dtype=torch.long)), os.path.join(OUT, "part1.pt"))
torch.save((torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1), torch.tensor(y_cnn, dtype=torch.long)), os.path.join(OUT, "part2.pt"))
torch.save((torch.randint(1, 50, (len(y_mlp), 15)), torch.randint(1, 10, (len(y_mlp), 8))), os.path.join(OUT, "part3.pt"))
print("\n🎉 DATASET READY. Proceed to training.")