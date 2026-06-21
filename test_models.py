# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"???  Device: {device}")

# ---------------- PART I: MLP ----------------
class MLP_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)
    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

mlp = MLP_Custom().to(device)
mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
mlp.eval()
dummy_mlp = torch.randn(1, 24, device=device)
with torch.no_grad():
    mlp_out = mlp(dummy_mlp)
print(f"? MLP Test OK | Input: {list(dummy_mlp.shape)} ? Output: {list(mlp_out.shape)} | Predicted class: {mlp_out.argmax().item()}")

# ---------------- PART II: CNN ----------------
class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(), nn.LazyLinear(84), nn.Sigmoid(), nn.LazyLinear(num_classes)
        )
    def forward(self, x):
        return self.net(x)

cnn = LeNet().to(device)
cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
cnn.eval()
dummy_cnn = torch.randn(1, 1, 128, 100, device=device)
with torch.no_grad():
    cnn_out = cnn(dummy_cnn)
print(f"? CNN Test OK | Input: {list(dummy_cnn.shape)} ? Output: {list(cnn_out.shape)} | Predicted class: {cnn_out.argmax().item()}")

# ---------------- PART III: Seq2Seq ----------------
class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        embedded = self.embedding(x)
        return self.gru(embedded)

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output.squeeze(1))
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        # EXACT parameters from training script:
        self.enc = EncoderGRU(vocab_size=50, embed_dim=32, hidden_dim=64)
        self.dec = DecoderGRU(vocab_size=10, embed_dim=32, hidden_dim=64, output_size=10)
    
    def greedy_decode(self, src, max_len=10):
        _, enc_hidden = self.enc(src)
        dec_input = torch.ones(src.shape[0], 1, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            pred, enc_hidden = self.dec(dec_input, enc_hidden)
            pred = pred.argmax(dim=1, keepdim=True)
            outputs.append(pred.squeeze(1))
            dec_input = pred
        return torch.stack(outputs, dim=1)

seq2seq = Seq2Seq().to(device)
state_dict = torch.load("seq2seq_gru.pth", map_location=device, weights_only=True)
seq2seq.load_state_dict(state_dict, strict=True)
seq2seq.eval()

dummy_src = torch.randint(1, 50, (1, 15), device=device)
with torch.no_grad():
    decoded = seq2seq.greedy_decode(dummy_src)
print(f"? Seq2Seq Test OK | Input: {list(dummy_src.shape)} ? Decoded: {decoded.squeeze().tolist()}")

print("\n?? All 3 models loaded successfully and produce valid outputs.")
