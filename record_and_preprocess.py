# -*- coding: utf-8 -*-
import os, sys, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import librosa
import sounddevice as sd
import pandas as pd

# 1. Record custom voice commands
def record_audio(duration=2, fs=16000):
    print("?? Recording... (2 sec)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()

os.makedirs("data/raw_voice", exist_ok=True)

COMMANDS = ["open chrome", "close notepad", "search weather", "open calculator"]
SAMPLES_PER_CMD = 10

for cmd in COMMANDS:
    for i in range(SAMPLES_PER_CMD):
        safe_name = cmd.replace(" ", "_")
        filepath = f"data/raw_voice/{safe_name}_{i}.wav"
        if not os.path.exists(filepath):
            input(f"Press ENTER to record: '{cmd}'")
            audio = record_audio()
            librosa.output.write_wav(filepath, audio, 16000)

print("? All recordings saved.")

# 2. Extract MLP Tabular Features
records = []
for f in glob.glob("data/raw_voice/*.wav"):
    y, _ = librosa.load(f, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=16000))
    intent = 0 if "open" in f else 1 if "close" in f else 2 if "search" in f else 3
    records.append([*mfcc, zcr, centroid, intent])

df = pd.DataFrame(records, columns=[f"mfcc_{i}" for i in range(13)] + ["zcr", "centroid", "label"])
df.to_csv("data/raw_voice/mlp_features.csv", index=False)
print("? MLP CSV generated.")

# 3. Generate CNN Spectrograms
for i, f in enumerate(glob.glob("data/raw_voice/*.wav")):
    y, _ = librosa.load(f, sr=16000)
    spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128), ref=np.max)
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
    torch.save(spec_tensor, f"data/raw_voice/cnn_{i}.pt")
print("? CNN spectrograms generated.")

# 4. Create Seq2Seq Pairs (Source: phoneme-like int IDs, Target: command IDs)
src_seqs = torch.randint(1, 50, (40, 15))
tgt_seqs = torch.randint(1, 10, (40, 8))
torch.save((src_seqs, tgt_seqs), "data/raw_voice/seq2seq.pt")
print("? Seq2Seq pairs generated.")
