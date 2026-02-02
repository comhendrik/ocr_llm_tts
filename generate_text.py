import wave
import numpy as np
import sounddevice as sd
from piper import PiperVoice

def tts_to_wav(voice: PiperVoice, text: str, out_path: str) -> None:
    with wave.open(out_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

def tts_live(voice: PiperVoice, text: str) -> None:
    sr = int(voice.config.sample_rate)
    with sd.OutputStream(samplerate=sr, channels=1, dtype="int16") as stream:
        for chunk in voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).reshape(-1, 1)
            stream.write(audio)

if __name__ == "__main__":
    VOICE_PATH = "piperVoices/de_DE-thorsten-medium.onnx"
    voice = PiperVoice.load(VOICE_PATH)

    tts_live(voice, "Moin zusammen, hier einmal der Livetest")
    tts_to_wav(voice, "Und hier einmal das ganze als Datei", "output.wav")
