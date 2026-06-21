# -*- coding: utf-8 -*-
import subprocess, webbrowser, sys, os

APP_PATHS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "notepad": "notepad.exe",
    "calc": "calc.exe"
}

def execute_action(intent: str, target: str):
    intent, target = intent.lower().strip(), target.lower().strip()
    try:
        if intent == "open":
            if target in APP_PATHS:
                subprocess.Popen(APP_PATHS[target])
            elif target in ["google", "web", "browser"]:
                webbrowser.open("https://www.google.com")
            else:
                webbrowser.open(f"https://www.google.com/search?q={target}")
            print(f"[OK] Opened/Searching: {target}")
        elif intent == "close":
            # Fallback without psutil for simplicity
            subprocess.run(["taskkill", "/f", "/im", f"{target}.exe"], capture_output=True)
            print(f"[OK] Attempted to close: {target}")
        elif intent == "search":
            webbrowser.open(f"https://www.google.com/search?q={target}")
            print(f"[OK] Searching: {target}")
        else:
            print(f"[WARN] Unknown intent: {intent}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        execute_action(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python action_mapper.py <intent> <target>")
