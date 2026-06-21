import sounddevice as sd
print("?? Valid Input Devices:")
found = False
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"  ID {i:2d} | {dev['name']} | Channels: {dev['max_input_channels']} | SR: {int(dev['default_samplerate'])}")
        found = True
if not found:
    print("? No input devices detected. Check Windows Sound settings or mic connection.")
else:
    print("\n?? Use the ID above in your recording script.")
