# -*- coding: utf-8 -*-
import os, shutil, zipfile, datetime

PROJECT = "VoiceAgent_DLM_Aymen"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP = f"{PROJECT}_{TIMESTAMP}.zip"

# Ensure all assets exist
for f in ["report/figures/mlp_confusion.png", "report/figures/loss_curves.png", "report/metrics.json"]:
    if not os.path.exists(f):
        print(f"⚠️  Missing: {f}")

# Build submission folder
os.makedirs(f"submission/{PROJECT}/data/placeholder", exist_ok=True)
for src, dst in [("src", "src"), ("report", "report"), ("run_pipeline.py", "."), ("run_full_project.py", ".")]:
    if os.path.isdir(src):
        shutil.copytree(src, f"submission/{PROJECT}/{dst}", dirs_exist_ok=True)
    else:
        shutil.copy2(src, f"submission/{PROJECT}/{dst}")
shutil.copy2("data/placeholder/part1.pt", f"submission/{PROJECT}/data/placeholder/")
shutil.copy2("data/placeholder/part2.pt", f"submission/{PROJECT}/data/placeholder/")
shutil.copy2("data/placeholder/part3.pt", f"submission/{PROJECT}/data/placeholder/")

# Create README
with open(f"submission/{PROJECT}/README.md", "w", encoding="utf-8") as f:
    f.write(f"# {PROJECT}\nEMSI Deep Learning Final Project\nTimestamp: {TIMESTAMP}\n\nRun: python run_full_project.py\nReport: report/main_report.docx")

# Zip
print("📦 Packaging...")
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(f"submission/{PROJECT}"):
        for file in files:
            p = os.path.join(root, file)
            z.write(p, os.path.relpath(p, f"submission/{PROJECT}"))
shutil.rmtree("submission", ignore_errors=True)
print(f"✅ Ready: {ZIP}")
