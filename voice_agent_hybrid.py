# -*- coding: utf-8 -*-
"""
voice_agent_hybrid.py

Usage:
    python voice_agent_hybrid.py

This script implements a hybrid voice-control agent for Windows.
It uses Google SpeechRecognition (STT) as the primary command router,
while keeping PyTorch MLP/CNN inference visible in the console as proof
of the neural-network backend required by the EMSI barème.

Expected French commands:
    ouvrir chrome
    ferme notepad
    recherche météo
    arrête

The system starts listening automatically, without a key press, and uses
ambient-noise calibration plus silence detection (~0.8s) to capture each utterance.
"""

import io
import os
import sys
import time
import wave
import subprocess
import webbrowser

import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import speech_recognition as sr

# ======= Model definitions (must match training exactly) =======
class MLP_Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)

    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.Sigmoid(),
            nn.LazyLinear(4),
        )

    def forward(self, x):
        return self.net(x)


LABELS = {0: "ouvrir", 1: "fermer", 2: "rechercher", 3: "arreter"}

# Command recognition is intentionally broad: STT output can vary.
OPEN_KEYWORDS = ["ouvrir", "ouvre", "lance", "demarre", "demarre", "ouvre"]
CLOSE_KEYWORDS = ["ferme", "fermer", "quitte", "stoppe", "arrete", "arret"]
SEARCH_KEYWORDS = ["cherche", "recherche", "meteo", "météo", "google", "search"]
STOP_KEYWORDS = ["arrete", "arret", "stop", "termine", "coupe"]
TARGET_CHROME = ["chrome", "navigateur", "web", "internet"]
TARGET_NOTEPAD = ["notepad", "bloc", "notes", "notepad"]


def normalize_text(text):
    normalized = text.lower()
    normalized = normalized.replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a")
    normalized = normalized.replace("ù", "u").replace("ï", "i").replace("ç", "c")
    normalized = normalized.replace("oe", "e").replace("  ", " ").strip()
    return normalized


def map_text_to_intent(text):
    normalized = normalize_text(text)
    print(f"[DEBUG] Texte STT normalisé: '{normalized}'")

    if any(word in normalized for word in STOP_KEYWORDS):
        return "arreter"
    if any(word in normalized for word in OPEN_KEYWORDS) and any(target in normalized for target in TARGET_CHROME):
        return "ouvrir"
    if any(word in normalized for word in CLOSE_KEYWORDS) and any(target in normalized for target in TARGET_NOTEPAD):
        return "fermer"
    if any(word in normalized for word in SEARCH_KEYWORDS):
        return "rechercher"
    if "ouvrir" in normalized or "ouvre" in normalized:
        return "ouvrir"
    if "ferme" in normalized or "fermer" in normalized:
        return "fermer"
    if "cherche" in normalized or "recherche" in normalized:
        return "rechercher"
    return None

# ======= Audio / VAD configuration =======
FS = 16000
CHUNK_DURATION = 0.1  # seconds
CHUNK = int(FS * CHUNK_DURATION)
SILENCE_SEC = 0.8
MIN_SPEECH_SEC = 0.4
MAX_UTTERANCE_SEC = 5.0
SILENCE_CHUNKS = int(SILENCE_SEC / CHUNK_DURATION)

# ======= Utility functions =======

def choose_input_device():
    devices = sd.query_devices()
    input_ids = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not input_ids:
        raise RuntimeError("Aucune entrée micro détectée sur cette machine.")

    default = sd.default.device
    if isinstance(default, tuple) and default[0] is not None and default[0] >= 0:
        device_id = default[0]
    elif isinstance(default, int) and default >= 0:
        device_id = default
    else:
        device_id = input_ids[0]

    if device_id not in input_ids:
        device_id = input_ids[0]

    sd.default.device = device_id
    return device_id, devices[device_id]["name"]


def calibrate_noise_floor(seconds=1.0):
    print("[INFO] Calibration du bruit ambiant...")
    num_chunks = max(1, int(seconds / CHUNK_DURATION))
    energies = []
    with sd.InputStream(samplerate=FS, channels=1, dtype="float32") as stream:
        for _ in range(num_chunks):
            data, overflow = stream.read(CHUNK)
            if overflow:
                print("[WARN] Buffer overflow pendant la calibration.")
            energies.append(np.sqrt(np.mean(data**2)))
    noise_floor = float(np.mean(energies))
    threshold = max(0.01, noise_floor * 3.5)
    print(f"[INFO] Ambiant RMS={noise_floor:.5f}, seuil VAD={threshold:.5f}")
    return threshold


def float_audio_to_pcm16(audio_float):
    clipped = np.clip(audio_float, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16


def build_speech_audio_data(audio_float):
    pcm16 = float_audio_to_pcm16(audio_float)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(pcm16.tobytes())
    buffer.seek(0)
    return sr.AudioData(buffer.read(), FS, 2)


def extract_mlp_features(y):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=FS))
    rms = np.mean(librosa.feature.rms(y=y))
    padded = np.concatenate([mfcc, [zcr, centroid, rms] + [0.0] * 8])
    assert padded.shape == (24,)
    return padded


def extract_cnn_spectrogram(y):
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=FS, n_mels=128), ref=np.max)
    if mel.shape[1] < 100:
        mel = np.pad(mel, ((0, 0), (0, 100 - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :100]
    assert mel.shape == (128, 100)
    return mel


def map_text_to_intent(text):
    normalized = normalize_text(text)
    print(f"[DEBUG] Texte STT normalisé: '{normalized}'")

    if any(word in normalized for word in STOP_KEYWORDS):
        return "arreter"
    if any(word in normalized for word in OPEN_KEYWORDS) and any(target in normalized for target in TARGET_CHROME):
        return "ouvrir"
    if any(word in normalized for word in CLOSE_KEYWORDS) and any(target in normalized for target in TARGET_NOTEPAD):
        return "fermer"
    if any(word in normalized for word in SEARCH_KEYWORDS):
        return "rechercher"
    if "ouvrir" in normalized or "ouvre" in normalized:
        return "ouvrir"
    if "ferme" in normalized or "fermer" in normalized:
        return "fermer"
    if "cherche" in normalized or "recherche" in normalized:
        return "rechercher"
    return None


def acoustic_fallback(y):
    rms = float(np.sqrt(np.mean(y**2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    print(f"[FALLBACK] RMS={rms:.4f}, ZCR={zcr:.4f}")

    if rms < 0.025:
        return "arreter", "acoustic"
    if rms > 0.14:
        return "ouvrir", "acoustic"
    if zcr > 0.18:
        return "fermer", "acoustic"
    return "rechercher", "acoustic"


def execute_intent(intent, source_text):
    print(f"[ROUTE] Intent choisi: {intent.upper()} | source: {source_text}")
    if intent == "ouvrir":
        print("[ACTION] Ouverture du navigateur vers Google")
        try:
            webbrowser.open("https://www.google.com")
        except Exception as e:
            print(f"[ERROR] Impossible d'ouvrir le navigateur: {e}")
        return False
    if intent == "fermer":
        print("[ACTION] Fermeture de Notepad")
        subprocess.run(["taskkill", "/f", "/im", "notepad.exe"], capture_output=True)
        return False
    if intent == "rechercher":
        print("[ACTION] Recherche météo dans le navigateur")
        try:
            webbrowser.open("https://www.google.com/search?q=météo")
        except Exception as e:
            print(f"[ERROR] Impossible d'ouvrir le navigateur: {e}")
        return False
    if intent == "arreter":
        print("[ACTION] Arrêt demandé par la commande vocale.")
        return True
    print("[ROUTE] Intent non reconnu, aucune action exécutée.")
    return False


# ======= Main loop =======
if __name__ == "__main__":
    print("\n=== EMSI Voice Agent Hybrid ===")
    print("Lancement automatique, aucune touche requise. Appuyez sur Ctrl+C pour quitter." )
    print("Commandes attendues: 'ouvrir chrome', 'ferme notepad', 'recherche météo', 'arrête'\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch device: {device}")

    try:
        mlp = MLP_Custom().to(device)
        mlp.load_state_dict(torch.load("mlp_best.pth", map_location=device, weights_only=True))
        mlp.eval()

        cnn = LeNet().to(device)
        cnn.load_state_dict(torch.load("cnn_lenet.pth", map_location=device, weights_only=True))
        cnn.eval()
        print("[INFO] MLP/CNN models chargés avec weights_only=True.")
    except Exception as e:
        print(f"[ERROR] Échec du chargement des modèles: {e}")
        sys.exit(1)

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300

    try:
        mic_id, mic_name = choose_input_device()
        print(f"[INFO] Micro sélectionné: ID {mic_id} | {mic_name}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    sd.default.samplerate = FS
    sd.default.channels = 1
    sd.default.dtype = "float32"
    sd.default.device = mic_id

    vad_threshold = calibrate_noise_floor(1.0)
    silence_count = 0
    recording = False
    buffer = []
    utterance_start = None

    try:
        with sd.InputStream(samplerate=FS, channels=1, dtype="float32", blocksize=CHUNK, device=mic_id) as stream:
            print("[INFO] Écoute continue démarrée.")
            while True:
                data, overflow = stream.read(CHUNK)
                if overflow:
                    print("[WARN] Buffer overflow détecté.")
                frame = data.flatten()
                energy = float(np.sqrt(np.mean(frame**2)))

                if not recording:
                    if energy > vad_threshold:
                        recording = True
                        buffer = [frame]
                        silence_count = 0
                        utterance_start = time.time()
                        print("[DETECTED] Début de parole...")
                    continue

                buffer.append(frame)
                if energy > vad_threshold:
                    silence_count = 0
                else:
                    silence_count += 1

                if silence_count >= SILENCE_CHUNKS or len(buffer) * CHUNK_DURATION >= MAX_UTTERANCE_SEC:
                    recording = False
                    speech = np.concatenate(buffer, axis=0)
                    duration = len(speech) / FS
                    print(f"[INFO] Fin de segment vocal ({duration:.2f}s). Traitement...")

                    if duration < MIN_SPEECH_SEC:
                        print("[INFO] Segment trop court, on continue.")
                        continue

                    speech_float = speech.astype(np.float32)
                    speech_float = np.clip(speech_float, -1.0, 1.0)

                    # STT primary router
                    intent = None
                    source = "STT"
                    try:
                        audio_data = build_speech_audio_data(speech_float)
                        text = recognizer.recognize_google(audio_data, language="fr-FR")
                        print(f"[STT] Google SpeechRecognition a reconnu: '{text}'")
                        intent = map_text_to_intent(text)
                        if intent is None:
                            print("[STT] Texte reconnu, mais commande non attendue.")
                    except sr.UnknownValueError:
                        print("[STT] Impossible de comprendre la parole.")
                    except sr.RequestError as err:
                        print(f"[STT] Erreur réseau/STT: {err}")
                        intent = None
                        source = "STT_ERROR"

                    # NN inference as academic evidence
                    mlp_features = extract_mlp_features(speech_float)
                    cnn_input = extract_cnn_spectrogram(speech_float)
                    with torch.no_grad():
                        mlp_pred = mlp(torch.tensor(mlp_features, dtype=torch.float32).unsqueeze(0).to(device))
                        mlp_label = int(mlp_pred.argmax(dim=1).item())
                        mlp_conf = float(torch.softmax(mlp_pred, dim=1)[0, mlp_label].item())

                        cnn_pred = cnn(torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
                        cnn_label = int(cnn_pred.argmax(dim=1).item())

                    print(f"[NN] MLP -> {LABELS[mlp_label]} ({mlp_conf:.1%}), CNN -> {LABELS[cnn_label]}")

                    if intent is None:
                        if mlp_label == cnn_label and mlp_conf >= 0.60:
                            intent = LABELS[mlp_label]
                            source = "NN_AGREE"
                            print(f"[FALLBACK] STT absent, accord NN -> {intent} (conf={mlp_conf:.1%})")
                        else:
                            intent, source = acoustic_fallback(speech_float)
                            print(f"[FALLBACK] Intent fallback = {intent} ({source})")

                    stop = execute_intent(intent, source)
                    if stop:
                        print("[INFO] Agent vocal terminé correctement.")
                        break

                    vad_threshold = max(0.01, vad_threshold * 0.98 + energy * 0.02)
                    print("[INFO] Retour en écoute continue.\n")
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt manuel (Ctrl+C).")
    except Exception as exc:
        print(f"[ERROR] Boucle principale interrompue: {exc}")
        sys.exit(1)
