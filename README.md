### Schritt 1: System-Abhängigkeiten installieren

Bevor Python installiert wird, müssen die externen Programme auf dem Betriebssystem vorhanden sein.

#### 🐧 Linux (Ubuntu/Debian)

Öffne das Terminal und installiere Tesseract, die Entwickler-Pakete und die Audio-Bibliothek:

```bash
sudo apt update
# Tesseract OCR und Sprachpaket
sudo apt install tesseract-ocr

# PortAudio (Wichtig: auch das dev-Paket für die Header-Dateien)
sudo apt install libportaudio2 portaudio19-dev

```

#### 🪟 Windows

1. **Tesseract:**
* Lade den Installer herunter (z.B. von [UB Mannheim](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki)).
* Installiere es und merke dir den Pfad.
* **Wichtig:** Füge den Installationspfad (meist `C:\Program Files\Tesseract-OCR`) zu deinen Windows-Umgebungsvariablen (`PATH`) hinzu, damit das Terminal den Befehl `tesseract` kennt.


2. **PortAudio:**
* Unter Windows gibt es keine separate "libportaudio2" Installation wie unter Linux. Die benötigten `.dll` Dateien werden normalerweise direkt mit den Python-Bibliotheken (über `pip`) installiert. Falls es Probleme gibt, siehe Schritt 3.


---

### Schritt 2: Ollama einrichten & Modell laden

Ollama muss als Hintergrunddienst laufen, damit dein Python-Skript (über die Library) darauf zugreifen kann.

1. **Installieren:**
* **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
* **Windows:** Lade die `.exe` von [ollama.com](https://ollama.com) und installiere sie.


2. **Modell herunterladen:**
Führe im Terminal folgenden Befehl aus, um das spezifische Modell lokal zu speichern:
```bash
ollama pull gemma3:3b

```


3. **Starten:**
* **Windows:** Ollama läuft meist automatisch nach der Installation (Symbol in der Taskleiste).
* **Linux:** Falls es nicht automatisch läuft, starte es in einem separaten Terminal-Fenster mit:
```bash
ollama serve

```
*(Lass dieses Fenster offen oder den Dienst im Hintergrund laufen, während du dein Python-Projekt nutzt.)*

---

### Schritt 3: Python Umgebung (.venv) einrichten

Gehe in dein Projektverzeichnis, wo deine `.py` Dateien und die `requirements.txt` liegen.

#### 1. Virtuelle Umgebung erstellen

Dies erstellt einen Ordner namens `.venv` in deinem Projekt.

* **Linux & Windows:**
```bash
python -m venv .venv

```


#### 2. Virtuelle Umgebung aktivieren

Du musst dem Terminal sagen, dass es jetzt das Python aus diesem Ordner nutzen soll.

* **Linux:**
```bash
source .venv/bin/activate

```


*(Dein Prompt sollte jetzt `(.venv)` am Anfang zeigen.)*
* **Windows (PowerShell):**
```powershell
.venv\Scripts\Activate

```


*(Falls Fehler wegen Skriptausführung kommen: `Set-ExecutionPolicy Unrestricted -Scope Process` vorher ausführen).*
* **Windows (CMD):**
```cmd
.venv\Scripts\activate.bat

```



#### 3. Libraries aus requirements.txt installieren

Jetzt werden `ollama`, `pytesseract` und die Audio-Libs in die isolierte Umgebung geladen.

* **Linux & Windows:**
```bash
pip install -r requirements.txt

```

> **Hinweis für Windows & Audio:** Falls die `requirements.txt` eine Bibliothek wie `pyaudio` enthält und die Installation mit einer roten Fehlermeldung abbricht (meist wegen fehlenden C++ Tools), musst du `pipwin` nutzen:
> 1. `pip install pipwin`
> 2. `pipwin install pyaudio`
> 3. Versuche danach nochmal `pip install -r requirements.txt`.
> 
> 

---

### Schritt 4: Das Projekt starten

Sobald alles installiert ist und die Umgebung **aktiviert** ist (das `(.venv)` ist sichtbar), startest du dein Hauptskript einfach so:

In diesem Skript kann dann des genutzte Bild angepasst werden. Lege ein Bild im Ordner images ab und referenziere es unter 

```python
IMAGE_PATH = "images/example.png"
```

```bash
python main.py

```