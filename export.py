import os
import shutil

FONT_CANDIDATES = [
    # --- The Arial "Twins" (Metric Compatible) ---
    "LiberationSans-Regular.ttf", # Standard Linux Arial replacement
    "LiberationSans-Bold.ttf",
    "Arimo-Regular.ttf",          # ChromeOS Arial replacement (Excellent match)
    "Arimo-Bold.ttf",
    "FreeSans.ttf",               # GNU Arial replacement
    "FreeSansBold.ttf",

    # --- Modern Sans-Serif (Google/Android style) ---
    "Roboto-Regular.ttf",         # Standard Android font
    "Roboto-Medium.ttf",
    "NotoSans-Regular.ttf",       # Google's universal font
    "OpenSans-Regular.ttf",       # Very common web font
    "OpenSans-Semibold.ttf",

    # --- Classic Linux Sans-Serif ---
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "Ubuntu-R.ttf",               # Ubuntu OS standard
    "Ubuntu-M.ttf",               # Ubuntu Medium

    # --- Other Clean Sans-Serifs ---
    "Carlito-Regular.ttf",        # Metric compatible with Calibri
    "SourceSansPro-Regular.ttf",  # Adobe's UI font
    "Lato-Regular.ttf",

    # --- Originals (If you copy them from Windows) ---
    # Only uncomment if you actually copied these files to your fonts/ folder!
    # "Arial.ttf",
    # "Arialbd.ttf",              # Arial Bold
    # "Verdana.ttf",
    # "Tahoma.ttf",
]

def copy_linux_fonts(destination_folder):
    """Scan all Linux font directories and copy only FONT_CANDIDATES into destination_folder."""
    
    found_fonts = []
    copied = set()  # Prevent duplicate copies

    # Standard Linux font roots
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]

    os.makedirs(destination_folder, exist_ok=True)

    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue

        for root, _, files in os.walk(base_dir):
            for file in files:
                if file in FONT_CANDIDATES and file not in copied:
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(destination_folder, file)

                    try:
                        shutil.copy2(source_path, dest_path)
                        found_fonts.append(dest_path)
                        copied.add(file)
                    except Exception:
                        pass  # Ignore unreadable or permission-restricted files

    return found_fonts

copy_linux_fonts("fonts")