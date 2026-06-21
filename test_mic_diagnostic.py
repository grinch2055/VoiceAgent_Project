# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import sys

print("🎤 Microphone Diagnostic Tool")
print("="*45)

# 1. Lister uniquement les périphériques d'entrée
inputs = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
if not inputs:
    print("❌ Aucun périphérique d'entrée détecté. Vérifiez le branchement ou les paramètres Windows.")
    sys.exit(1)

print("📋 Microphones disponibles:")
for i in inputs:
    d = sd.query_devices(i)
    print(f"  ID {i:2d} | {d['name']} | API: {d['hostapi']}")

# 2. Tester chaque micro automatiquement
print("\n🔍 Test automatique en cours... (Parlez FORT pendant 3s à chaque test)")
working_id = None

for dev_id in inputs:
    print(f"\n📡 Testing ID {dev_id}...")
    try:
        # Enregistrement 3 secondes à 16kHz
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32', device=dev_id)
        sd.wait()

        rms = float(np.sqrt(np.mean(audio**2)))
        print(f"   📊 Volume RMS: {rms:.4f}")

        if rms > 0.02:
            print("   🔊 Rejoue l'enregistrement...")
            sd.play(audio, 16000)
            sd.wait()
            print(f"   ✅ SUCCÈS! Le micro ID {dev_id} FONCTIONNE.")
            working_id = dev_id
            break
        else:
            print(f"   ⚠️ Trop faible (RMS < 0.02). Vérifiez le volume dans Windows.")
    except Exception as e:
        err_msg = str(e)
        if "Invalid device" in err_msg or "-9996" in err_msg:
            print(f"   ❌ Bloqué/Invalide (problème driver ou permission Windows)")
        else:
            print(f"   ❌ Erreur: {err_msg[:60]}")

# 3. Résultat final
if working_id is not None:
    print(f"\n🎯 RÉSULTAT: Utilisez l'ID {working_id} pour votre projet.")
    print("💡 Note: Si c'est ID 1, 5 ou 9, c'est votre micro intégré (Array).")
else:
    print("\n❌ AUCUN MICRO FONCTIONNEL DÉTECTÉ.")
    print("🔧 CORRECTIONS RAPIDES:")
    print("   1. Win+I > Confidentialité > Micro > Autoriser les apps de bureau")
    print("   2. mmsys.cpl > Enregistrement > Propriétés > Niveaux > 80-100%")
    print("   3. mmsys.cpl > Enregistrement > Propriétés > Avancé > DÉCOCHER Mode exclusif")
    print("   4. Fermez Zoom/Teams/Discord/Spotify (ils verrouillent le micro)")

print("\n" + "="*45)