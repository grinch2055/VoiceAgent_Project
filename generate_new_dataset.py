# -*- coding: utf-8 -*-
import os, numpy as np, torch, librosa
from scipy.io.wavfile import write as wavwrite

FS, DUR, OUT = 16000, 2.0, "data/raw_voice"
os.makedirs(OUT, exist_ok=True)
SAMPLES = int(FS * DUR)

INTENTS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}
X_mlp, y_mlp, X_cnn, y_cnn = [], [], [], []

print("Generating realistic acoustic dataset...")
for intent_id, name in INTENTS.items():
    for i in range(10):
        t = np.linspace(0, DUR, SAMPLES)
        # Distinct acoustic signature per class (pitch + harmonics + envelope)
        base_freq = 130 + (intent_id * 45) + (i * 6)
        signal = 0.4 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.25 * np.sin(2 * np.pi * (base_freq * 2.2) * t + np.sin(t*4))
        signal += 0.15 * np.sin(2 * np.pi * (base_freq * 3.1) * t)
        signal += np.random.normal(0, 0.07, SAMPLES)
        # Speech-like amplitude envelope
        env_len = int(SAMPLES * 0.12)
        envelope = np.concatenate([
            np.linspace(0, 1, env_len),
            np.ones(SAMPLES - 2 * env_len),
            np.linspace(1, 0, env_len)
        ])
        signal = np.clip(signal * envelope, -1, 1)

        filepath = os.path.join(OUT, f"{name}_{i}.wav")
        wavwrite(filepath, FS, (signal * 32767).astype(np.int16))

        # MLP Features (exactly 24 dims)
        mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=FS, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
        cent = np.mean(librosa.feature.spectral_centroid(y=signal, sr=FS))
        rms = np.mean(librosa.feature.rms(y=signal))
        feats = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
        X_mlp.append(feats); y_mlp.append(intent_id)

        # CNN Spectrogram (128x100)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=FS, n_mels=128), ref=np.max)
        if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
        else: mel = mel[:, :100]
        X_cnn.append(mel); y_cnn.append(intent_id)

# Save tensors
torch.save((torch.tensor(X_mlp, dtype=torch.float32), torch.tensor(y_mlp, dtype=torch.long)), os.path.join(OUT, "part1.pt"))
torch.save((torch.tensor(X_cnn, dtype=torch.float32).unsqueeze(1), torch.tensor(y_cnn, dtype=torch.long)), os.path.join(OUT, "part2.pt"))
torch.save((torch.randint(1, 50, (len(y_mlp), 15)), torch.randint(1, 10, (len(y_mlp), 8))), os.path.join(OUT, "part3.pt"))

print("✅ Dataset generated. Shapes:")
print(f"   MLP: {torch.tensor(X_mlp).shape} | Labels: {len(y_mlp)}")
print(f"   CNN: {torch.tensor(X_cnn).unsqueeze(1).shape}")
print("   Ready for retraining.")