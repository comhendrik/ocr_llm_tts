import wave
import numpy as np
import sounddevice as sd
from piper import PiperVoice

VOICE_PATH = "piperVoices/de_DE-thorsten-medium.onnx"
voice = PiperVoice.load(VOICE_PATH)

def tts_to_wav(text: str, out_path: str):
    with wave.open(out_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

def tts_live(text: str):
    sr = int(voice.config.sample_rate)
    with sd.OutputStream(samplerate=sr, channels=1, dtype="int16") as stream:
        for chunk in voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).reshape(-1, 1)
            stream.write(audio)

if __name__ == "__main__":
    tts_live("Heute ist ein guter Tag, um einen langen Text zu testen. Ich schreibe hier absichtlich viele Saetze hintereinander, damit du pruefen kannst, wie sich eine kontinuierliche Ausgabe verhaelt, wenn keine Zeilenumbrueche vorkommen und der Input laenger wird. Dabei kann es spannend sein zu beobachten, ob die Stimme konstant bleibt, ob Pausen an den richtigen Stellen entstehen, und ob die Geschwindigkeit ueber die gesamte Dauer stabil ist. Wenn du das Ganze fuer Text to Speech nutzt, achte darauf, wie Punkte, Kommas und Doppelpunkte interpretiert werden: Manche Stimmen machen nach einem Punkt eine deutlichere Pause, nach einem Komma nur eine kurze, und bei einem Doppelpunkt oft eine kleine Betonung. Du kannst auch Zahlen einbauen, zum Beispiel 1234567890, oder Datumsangaben wie 2026-02-02, um zu sehen, wie sie ausgesprochen werden. Ausserdem hilft es, unterschiedliche Satzlaengen zu mischen, damit der Rhythmus nicht zu monoton wird. Ein kurzer Satz. Ein sehr langer Satz, der mehrere Nebensaetze enthaelt, damit du testen kannst, ob die Betonung noch nachvollziehbar bleibt, auch wenn der Satz sich zieht und viele Woerter enthaelt, die aehnlich klingen oder schnell hintereinander kommen. Wenn du parallel noch OCR oder andere Verarbeitung laufen hast, ist das ebenfalls ein guter Stresstest, denn dann merkst du schnell, ob deine Pipeline sauber puffert und ob die Ausgabe ruckelfrei bleibt. Am Ende kannst du kontrollieren, ob der komplette Text gesprochen wurde, ob irgendwo etwas abgeschnitten ist, und ob die Ausgabe genau so endet, wie du es erwartest.")
    tts_to_wav("Heute ist ein guter Tag, um einen langen Text zu testen. Ich schreibe hier absichtlich viele Saetze hintereinander, damit du pruefen kannst, wie sich eine kontinuierliche Ausgabe verhaelt, wenn keine Zeilenumbrueche vorkommen und der Input laenger wird. Dabei kann es spannend sein zu beobachten, ob die Stimme konstant bleibt, ob Pausen an den richtigen Stellen entstehen, und ob die Geschwindigkeit ueber die gesamte Dauer stabil ist. Wenn du das Ganze fuer Text to Speech nutzt, achte darauf, wie Punkte, Kommas und Doppelpunkte interpretiert werden: Manche Stimmen machen nach einem Punkt eine deutlichere Pause, nach einem Komma nur eine kurze, und bei einem Doppelpunkt oft eine kleine Betonung. Du kannst auch Zahlen einbauen, zum Beispiel 1234567890, oder Datumsangaben wie 2026-02-02, um zu sehen, wie sie ausgesprochen werden. Ausserdem hilft es, unterschiedliche Satzlaengen zu mischen, damit der Rhythmus nicht zu monoton wird. Ein kurzer Satz. Ein sehr langer Satz, der mehrere Nebensaetze enthaelt, damit du testen kannst, ob die Betonung noch nachvollziehbar bleibt, auch wenn der Satz sich zieht und viele Woerter enthaelt, die aehnlich klingen oder schnell hintereinander kommen. Wenn du parallel noch OCR oder andere Verarbeitung laufen hast, ist das ebenfalls ein guter Stresstest, denn dann merkst du schnell, ob deine Pipeline sauber puffert und ob die Ausgabe ruckelfrei bleibt. Am Ende kannst du kontrollieren, ob der komplette Text gesprochen wurde, ob irgendwo etwas abgeschnitten ist, und ob die Ausgabe genau so endet, wie du es erwartest.", "output.wav")
