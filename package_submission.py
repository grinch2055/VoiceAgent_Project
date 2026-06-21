# -*- coding: utf-8 -*-
import os, shutil, zipfile, datetime

PROJECT_NAME = "VoiceAgent_DLM_Aymen"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP_NAME = f"{PROJECT_NAME}_{TIMESTAMP}.zip"

print("Structuring submission package...")
os.makedirs(f"submission/{PROJECT_NAME}/data", exist_ok=True)

items = ["src", "report", "generate_report_assets.py", "run_pipeline.py"]
for item in items:
    if os.path.exists(item):
        dest = f"submission/{PROJECT_NAME}/{item}"
        if os.path.isdir(item):
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

for pt in ["part1.pt", "part2.pt", "part3.pt"]:
    src = f"data/placeholder/{pt}"
    if os.path.exists(src):
        shutil.copy2(src, f"submission/{PROJECT_NAME}/data/{pt}")

readme = f"# {PROJECT_NAME}\nEMSI Deep Learning Final Project\nTimestamp: {TIMESTAMP}\n\nExecute: python run_pipeline.py"
with open(f"submission/{PROJECT_NAME}/README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("Compressing...")
with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(f"submission/{PROJECT_NAME}"):
        for file in files:
            abs_path = os.path.join(root, file)
            arc_name = os.path.relpath(abs_path, f"submission/{PROJECT_NAME}")
            z.write(abs_path, arc_name)

shutil.rmtree("submission", ignore_errors=True)
print(f"Submission ready: {ZIP_NAME}")
