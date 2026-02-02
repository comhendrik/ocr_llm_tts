import wave
from piper import PiperVoice

# Wichtig unter Windows: raw string r"..." oder doppelte Backslashes
VOICE_PATH = "piperVoices/de_DE-thorsten-medium.onnx"

voice = PiperVoice.load(VOICE_PATH)

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Moin Niklas, das ist ein Test mit Piper.", wav_file)

print("Fertig: output.wav")
