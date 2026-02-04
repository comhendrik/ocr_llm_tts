import numpy as np
import os
import random
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
PADDING = 10          # MATCHES YOUR OPENCV SCRIPT

# Character set
chars_lower = "abcdefghijklmnopqrstuvwxyz"
chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
numbers = "0123456789"
german_specials = "ß"
punctuation = ".,;:!?()+-"

ALL_CHARS = list(chars_lower + chars_upper + numbers + german_specials + punctuation)

# Fonts
FONT_CANDIDATES = [
    "LiberationSans-Regular.ttf",
    "LiberationSerif-Regular.ttf",
    "LiberationMono-Regular.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "Ubuntu-R.ttf",
    "FreeSansBold.ttf",
    "FreeSerif.ttf",
    "OpenSans-Regular.ttf",
    "FreeSans.ttf",
]

def get_linux_fonts():
    """Scans the Linux environment for the installed fonts."""
    found_fonts = []
    search_dirs = [
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/freefont",
        "/usr/share/fonts/truetype/ubuntu"
    ]

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for file in files:
                if file in FONT_CANDIDATES:
                    found_fonts.append(os.path.join(root, file))
    return found_fonts

def generate_and_save():
    # 1. Locate fonts
    valid_fonts = get_linux_fonts()
    print(f"\nFonts found: {len(valid_fonts)}")

    if not valid_fonts:
        print("CRITICAL ERROR: No fonts found. Please install fonts or check paths.")
        return

    total_images = len(ALL_CHARS) * SAMPLES_PER_CHAR
    print(f"Generating {total_images} images (Target: {OUTPUT_PATH})...")

    # 2. Allocate Memory
    X = np.empty((total_images, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y = np.empty(total_images, dtype=np.int32)

    current_idx = 0

    # Calculate the target "Line Height" to match OpenCV script
    # In OpenCV: size(64) - padding(18)*2 = 28 pixels available for the line.
    target_line_height = IMAGE_SIZE - (PADDING * 2)

    # 3. Generation Loop
    pbar = tqdm(total=total_images, desc="Progress")

    for char_idx, char in enumerate(ALL_CHARS):
        for _ in range(SAMPLES_PER_CHAR):
            font_path = random.choice(valid_fonts)

            # ---------------------------------------------------------
            # KEY CHANGE 1: Font Size matches "Line Height"
            # ---------------------------------------------------------
            # We don't scale randomly. We scale to fill the 'line strip'.
            # A font size of X pixels usually results in a line height slightly larger than X.
            # We use a factor (e.g. 0.9 to 1.0) to simulate slight zoom variations in scanning.
            font_size = int(target_line_height * random.uniform(0.90, 1.05))

            # Small random shift to simulate scanning misalignment
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)

            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                continue

            # Create Canvas (1-bit, White background)
            img = Image.new('1', (IMAGE_SIZE, IMAGE_SIZE), color=1)
            draw = ImageDraw.Draw(img)

            # ---------------------------------------------------------
            # KEY CHANGE 2: Horizontal Center, Vertical Anchor
            # ---------------------------------------------------------

            # A. Horizontal Centering (Standard)
            # We still center the character horizontally (Left-to-Right)
            length = draw.textlength(char, font=font)
            pos_x = (IMAGE_SIZE - length) / 2

            # B. Vertical Alignment (Baseline Preservation)
            # instead of centering the *ink* (bbox), we place the *font line* # at the padding line.
            # The OpenCV script centers the line strip in the 64px canvas.
            # This means the top of the line strip is at Y = PADDING (18).
            pos_y = PADDING

            # Apply offsets
            pos_x += offset_x
            pos_y += offset_y

            # Draw text using anchor="la" (Left Ascender)
            # "la" means (pos_x, pos_y) is the Top-Left of the Ascender line.
            # This forces the text to hang correctly from the PADDING line.
            # Note: If this fails on old Pillow versions, update Pillow.
            draw.text((pos_x, pos_y), char, font=font, fill=0, anchor="la")

            # Convert to numpy array (0=Black, 255=White)
            img_array = np.array(img, dtype=np.uint8) * 255

            X[current_idx] = img_array
            y[current_idx] = char_idx
            current_idx += 1
            pbar.update(1)

    pbar.close()

    # 4. Save to Drive
    print(f"Saving to: {OUTPUT_PATH}...")
    chars = np.array(ALL_CHARS)

    try:
        np.savez_compressed(OUTPUT_PATH, X=X, y=y, chars=chars)
        print(f"SUCCESS! File saved: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving: {e}")

if __name__ == "__main__":
    generate_and_save()