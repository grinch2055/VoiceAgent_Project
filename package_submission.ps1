# -*- coding: utf-8 -*-
import os, shutil, zipfile, datetime

PROJECT_NAME = "VoiceAgent_DLM_Aymen"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP_NAME = f"{PROJECT_NAME}_{TIMESTAMP}.zip"
FOLDERS_TO_ZIP = ["src", "data/placeholder", "report", "generate_report_assets.py", "run_pipeline.py", "requirements.txt"]

# 1. Create clean structure for submission
print("?? Structuring submission package...")
os.makedirs(f"submission/{PROJECT_NAME}/data", exist_ok=True)
for f in ["src", "report", "generate_report_assets.py", "run_pipeline.py", "requirements.txt"]:
    src = f if os.path.isdir(f) or os.path.isfile(f) else None
    if src:
        dest = f"submission/{PROJECT_NAME}/{f}"
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)

# 2. Add placeholder data (real voice will be swapped later)
shutil.copy("data/placeholder/part1.pt", f"submission/{PROJECT_NAME}/data/")
shutil.copy("data/placeholder/part2.pt", f"submission/{PROJECT_NAME}/data/")
shutil.copy("data/placeholder/part3.pt", f"submission/{PROJECT_NAME}/data/")

# 3. Create README with exact rubric mapping
readme = f"""# {PROJECT_NAME}
EMSI Deep Learning Final Project | {TIMESTAMP}

## Structure
- `src/part1_mlp/` : MLP tabulaire (nn.Module vs Sequential, 3 initialisations, state_dict, GPU)
- `src/part2_cnn/` : CNN images (corr2d manuelle, padding/stride, LeNet, feature maps)
- `src/part3_seq2seq/` : Seq2Seq séquentiel (BPTT, gradient clipping, teacher forcing, greedy vs beam search)
- `report/` : Rapport Word + figures + draft de synthèse
- `data/` : Placeholders prêts à être remplacés par le dataset vocal

## Exécution
1. python src/part1_mlp/train_mlp.py
2. python src/part2_cnn/train_cnn.py
3. python src/part3_seq2seq/train_seq2seq.py
4. python generate_report_assets.py
5. python run_pipeline.py

## Conformité Barème EMSI
? Partie I (30 pts) : Théorie, data prep, implémentation, init/sauvegarde/GPU, analyse, synthèse
? Partie II (35 pts) : Théorie CNN, implémentation manuelle, LeNet, expérimentation padding/stride/pooling, feature maps, synthèse
? Partie III (35 pts) : Modélisation séquentielle, RNN/LSTM/GRU, BPTT/clipping, Seq2Seq, décodage glouton/beam, BLEU, synthèse
? Question transversale : Adaptation architecture ? géométrie des données
"""
with open(f"submission/{PROJECT_NAME}/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

# 4. Zip
print("??? Compressing...")
with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(f"submission/{PROJECT_NAME}"):
        for file in files:
            abs_path = os.path.join(root, file)
            arc_name = os.path.relpath(abs_path, f"submission/{PROJECT_NAME}")
            zipf.write(abs_path, arc_name)

# 5. Cleanup temp
shutil.rmtree("submission", ignore_errors=True)
print(f"? Submission ready: {ZIP_NAME}")
print("?? Next step: Record your voice dataset ? replace placeholders ? retrain ? update report ? resubmit.")
