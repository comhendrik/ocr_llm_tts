import wave
from piper import PiperVoice

VOICE_PATH = "piperVoices/de_DE-thorsten-medium.onnx"

voice = PiperVoice.load(VOICE_PATH)

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Moin zusammen, das ist der erste Test.", wav_file)

print("Fertig: output.wav")
