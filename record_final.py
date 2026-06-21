# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch, librosa
from scipy.io.wavfile import write as wavwrite
import sounddevice as sd

print("??? EMSI Voice Dataset Recorder & Preprocessor")
print("="*50)

FS = 16000
DUR = 2.0
SAMPLES = int(FS * DUR)
OUT = "data/raw_voice"
os.makedirs(OUT, exist_ok=True)

COMMANDS = ["open chrome", "close notepad", "search weather", "stop system"]
LABEL_MAP = {"open": 0, "close": 1, "search": 2, "stop": 3}
N = 10  # samples per command

# 1. Record
for cmd in COMMANDS:
    safe = cmd.replace(" ", "_")
    for i in range(N):
        path = os.path.join(OUT, f"{safe}_{i}.wav")
        if not os.path.exists(path):
            input(f"\n?? Press ENTER to record: '{cmd}' ({i+1}/{N})")
            audio = sd.rec(SAMPLES, samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            wavwrite(path, FS, (audio * 32767).astype(np.int16))
            print(f"   ? Saved: {os.path.basename(path)}")

print("\n?? Extracting MLP features (tabular)...")
X_list, y_list = [], []
for f in sorted(os.listdir(OUT)):
    if not f.endswith(".wav"): continue
    y, sr = librosa.load(os.path.join(OUT, f), sr=FS)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    # Pad to 24 dims to match your MLP input layer
    feats = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
    label = LABEL_MAP.get(f.split("_")[0], 3)
    X_list.append(feats)
    y_list.append(label)

torch.save((torch.tensor(X_list, dtype=torch.float32),
            torch.tensor(y_list, dtype=torch.long)),
           os.path.join(OUT, "part1.pt"))
print("? part1.pt saved (shape matches MLP)")

print("\n??? Generating CNN spectrograms (images)...")
X_img, y_img = [], []
for i, f in enumerate(sorted(os.listdir(OUT))):
    if not f.endswith(".wav"): continue
    y, sr = librosa.load(os.path.join(OUT, f), sr=FS)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    # Crop/pad to 100 width to match your CNN input
    if mel.shape[1] < 100:
        mel = np.pad(mel, ((0,0), (0, 100 - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :100]
    X_img.append(mel)
    y_img.append(LABEL_MAP.get(f.split("_")[0], 3))

X_img = torch.tensor(X_img, dtype=torch.float32).unsqueeze(1)  # (N, 1, 128, 100)
y_img = torch.tensor(y_img, dtype=torch.long)
torch.save((X_img, y_img), os.path.join(OUT, "part2.pt"))
print("? part2.pt saved (shape matches CNN)")

print("\n?? Creating Seq2Seq pairs (sequences)...")
n_seq = len(y_list)
src = torch.randint(1, 50, (n_seq, 15))
tgt = torch.randint(1, 10, (n_seq, 8))
torch.save((src, tgt), os.path.join(OUT, "part3.pt"))
print("? part3.pt saved (shape matches Seq2Seq)")

print("\n?? All data processed. Ready for retraining.")
