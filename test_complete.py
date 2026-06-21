# -*- coding: utf-8 -*-
import os, glob, torch, numpy as np, librosa
import torch.nn as nn

print("="*50)
print("🔍 EMSI PROJECT TEST PROTOCOL")
print("="*50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {device}")

# 1. Vérification des livrables exigés (§9)
required = [
    "mlp_best.pth", "cnn_lenet.pth", "seq2seq_gru.pth",
    "report/figures/mlp_confusion_matrix.png", "report/figures/cnn_feature_map.png"
]
all_ok = True
for f in required:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"  [{status}] {f}")
    if status == "MISSING": all_ok = False

if not all_ok:
    print("\n⚠️  Certains fichiers manquent. Lance les scripts d'entraînement avant de tester.")
else:
    print("\n✅ Tous les livrables sont présents.")

    # 2. Test MLP (Conforme fiche_synthese_MLP_PyTorch.pdf)
    print("\n🧠 Testing MLP...")
    
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(24, 64)
            self.fc2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 4)
        def forward(self, x):
            return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    mlp = MLP().to(device)
    mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
    mlp.eval()
    with torch.no_grad():
        mlp_out = mlp(torch.randn(1, 24, device=device))
    print(f"   OK | Input: [1,24] -> Output: {mlp_out.shape} | Pred: {mlp_out.argmax().item()}")

    # 3. Test CNN (Conforme fiche_synthese_cnn.pdf)
    print("\n👁️ Testing CNN...")
    
    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1,6,5,padding=2), nn.Sigmoid(), nn.AvgPool2d(2,2),
                nn.Conv2d(6,16,5), nn.Sigmoid(), nn.AvgPool2d(2,2),
                nn.Flatten(), nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(), nn.LazyLinear(4))
        def forward(self, x):
            return self.net(x)

    cnn = LeNet().to(device)
    cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
    cnn.eval()
    with torch.no_grad():
        cnn_out = cnn(torch.randn(1,1,128,100, device=device))
    print(f"   OK | Input: [1,1,128,100] -> Output: {cnn_out.shape} | Pred: {cnn_out.argmax().item()}")

    # 4. Test Seq2Seq (Conforme synthese_rnn_seq2seq.pdf)
    print("\n📝 Testing Seq2Seq...")
    
    class EncoderGRU(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        def forward(self, x):
            _, h = self.gru(self.embedding(x))
            return h

    class DecoderGRU(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_size)
        def forward(self, x, h):
            embedded = self.embedding(x)
            output, h = self.gru(embedded, h)
            return self.fc(output.squeeze(1)), h

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = EncoderGRU(50, 32, 64)
            self.dec = DecoderGRU(10, 32, 64, 10)
        def forward(self, src):
            h = self.enc(src)
            dec_in = torch.ones(src.shape[0], 1, dtype=torch.long, device=src.device)
            for _ in range(10):
                pred, h = self.dec(dec_in, h)
                dec_in = pred.argmax(dim=1, keepdim=True)
            return pred

    s2s = Seq2Seq().to(device)
    s2s.load_state_dict(torch.load("seq2seq_gru.pth", map_location=device, weights_only=True))
    s2s.eval()
    with torch.no_grad():
        s2s_out = s2s(torch.randint(1,50,(1,15), device=device))
    print(f"   OK | Input: [1,15] -> Output: {s2s_out.shape}")

    # 5. Pipeline sur fichiers WAV réels (si existants)
    wavs = sorted(glob.glob("data/raw_voice/*.wav"))
    if wavs:
        print(f"\n🎤 Testing {min(3, len(wavs))} real voice samples...")
        labels = {0:"ouvrir", 1:"fermer", 2:"rechercher", 3:"arreter"}
        for f in wavs[:3]:
            y, sr = librosa.load(f, sr=16000)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            feat = np.concatenate([mfcc, [zcr, cent, rms] + [0.0]*8])
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
            if mel.shape[1] < 100: mel = np.pad(mel, ((0,0), (0, 100-mel.shape[1])), mode='constant')
            else: mel = mel[:, :100]

            with torch.no_grad():
                mlp_p = mlp(torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
                cnn_p = cnn(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)).argmax().item()
            print(f"  {os.path.basename(f)} -> MLP: {labels[mlp_p]} | CNN: {labels[cnn_p]}")

    print("\n🎉 TEST COMPLETE. Project is ready for submission/demo.")