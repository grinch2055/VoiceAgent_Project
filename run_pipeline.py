# -*- coding: utf-8 -*-
import os, sys

print("?? VoiceAgent Project Pipeline - EMSI Deep Learning")
print("="*50)

models = {"MLP": "mlp_best.pth", "CNN": "cnn_lenet.pth", "Seq2Seq": "seq2seq_gru.pth"}
for name, path in models.items():
    status = "? Found" if os.path.exists(path) else "?? Missing (run training first)"
    print(f"{name:8} | {path:20} | {status}")

print("\n?? Demo command (placeholder mode):")
print("   python src/integration/action_mapper.py open google")
print("   python src/integration/action_mapper.py close notepad")
print("\n?? Next steps:")
print("   1. Record your voice dataset ? data/raw_voice/")
print("   2. Run preprocessing scripts to replace placeholders")
print("   3. Retrain models ? update state_dict paths")
print("   4. Draft report using provided EMSI structure")
