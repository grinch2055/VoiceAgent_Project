# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import sys

print("=== Microphone Test ===")
print("Available input devices:")
input_devs = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
if not input_devs:
    print("No input devices found.")
    sys.exit(1)

for i in input_devs:
    print(f"  ID {i:2d} | {sd.query_devices(i)['name']}")

print("\nEnter ID to test (or press ENTER for first available):")
try:
    choice = input("> ").strip()
    dev_id = int(choice) if choice else input_devs[0]
except:
    dev_id = input_devs[0]
    print(f"Using default ID: {dev_id}")

print(f"\nTesting device ID {dev_id}...")
print("Speak clearly for 3 seconds now!")

try:
    audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32', device=dev_id)
    sd.wait()
except Exception as e:
    print(f"Recording failed: {e}")
    sys.exit(1)

rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"Volume RMS: {rms:.4f}")

if rms > 0.02:
    print("Playing back...")
    sd.play(audio, 16000)
    sd.wait()
    print("SUCCESS: Microphone works!")
else:
    print("WARNING: Volume too low. Check Windows Sound settings or speak closer.")