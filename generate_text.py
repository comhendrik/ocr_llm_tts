import os
from pathlib import Path
from playsound3 import playsound
from piper.voice import PiperVoice

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "piperVoices"

MODEL_PATH = MODEL_DIR / "de_DE-thorsten-medium.onnx"
CONFIG_PATH = MODEL_DIR / "de_DE-thorsten-medium.onnx.json"
OUTPUT_WAV = BASE_DIR / "output.wav"

def text_to_speech(text: str):
    if not text.strip():
        raise ValueError("Text ist leer")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")

    # 1. Initialize Piper Voice
    voice = PiperVoice.load(model_path=str(MODEL_PATH), config_path=str(CONFIG_PATH))

    # 2. Synthesize to file
    # We use 'wb' to overwrite the file every time
    with open(OUTPUT_WAV, "wb") as wav_file:
        voice.synthesize(text, wav_file)

    # 3. Play the file (Cross-Platform)
    # playsound3 handles the blocking so the script waits for the audio to finish
    try:
        playsound(str(OUTPUT_WAV))
    except Exception as e:
        print(f"Playback error: {e}")

if __name__ == "__main__":
    text_to_speech("Dies ist ein Test mit Play-Sound 3 auf Windows und Linux.")