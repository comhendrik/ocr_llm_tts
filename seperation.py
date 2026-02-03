import cv2
import pytesseract
import numpy as np
import os

# --------------------
# Configuration
# --------------------
IMAGE_PATH = "images/IMG_5276.JPEG"
OUTPUT_IMAGE_PATH = "master_ocr_image.jpg"
OUTPUT_DIR = "output_words"

CONF_THRESHOLD = 40
MIN_CHAR_WIDTH = 3
MIN_CHAR_HEIGHT_RATIO = 0.3


# --------------------
# Load image
# --------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Preprocessing
# --------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# --------------------
# OCR
# --------------------
# Output directly as a standard Python dictionary
data = pytesseract.image_to_data(
    thresh,
    config="--oem 3 --psm 6",
    output_type=pytesseract.Output.DICT
)

# --------------------
# Parse & Group Data (No Pandas)
# --------------------
# Structure: {(par_num, line_num): [ {word_data}, {word_data} ] }
lines_map = {}

num_boxes = len(data['text'])

for i in range(num_boxes):
    # Check confidence
    conf_val = data['conf'][i]
    # Tesseract sometimes returns -1 or strings for block headers
    if conf_val == '-1' or int(conf_val) < CONF_THRESHOLD:
        continue
        
    text = data['text'][i].strip()
    if not text:
        continue

    # Extract coordinates
    x = data['left'][i]
    y = data['top'][i]
    w = data['width'][i]
    h = data['height'][i]
    
    par_num = data['par_num'][i]
    line_num = data['line_num'][i]
    
    key = (par_num, line_num)
    
    word_entry = {
        "text": text,
        "left": x,
        "top": y,
        "width": w,
        "height": h,
        "right": x + w,
        "bottom": y + h
    }
    
    if key not in lines_map:
        lines_map[key] = []
    
    lines_map[key].append(word_entry)

# --------------------
# Draw Line Bounding Boxes & Build Records
# --------------------
annotated_image = image.copy()
records = []

# Sort lines by paragraph then line number for processing order
sorted_keys = sorted(lines_map.keys())

for key in sorted_keys:
    par_num, line_num = key
    words = lines_map[key]
    
    # Calculate Line Bounding Box (Union of all words in line)
    line_x1 = min(w["left"] for w in words)
    line_y1 = min(w["top"] for w in words)
    line_x2 = max(w["right"] for w in words)
    line_y2 = max(w["bottom"] for w in words)
    
    full_line_text = " ".join(w["text"] for w in words)
    
    # Store record
    records.append({
        "paragraph_number": par_num,
        "line_number": line_num,
        "entry_text": full_line_text,
        "bounding_box": (line_x1, line_y1, line_x2, line_y2)
    })
    
    # Draw rectangle on master image
    cv2.rectangle(
        annotated_image,
        (line_x1, line_y1),
        (line_x2, line_y2),
        (0, 0, 255),
        2
    )

# Save Master Image
resized = cv2.resize(annotated_image, (0, 0), fx=0.5, fy=0.5)
cv2.imwrite(OUTPUT_IMAGE_PATH, resized)

# ==========================================================
# WORD & CHARACTER EXTRACTION
# ==========================================================

for key in sorted_keys:
    par_num, line_num = key
    words = lines_map[key]

    # Setup directory: output/page_1/paragraph_X/line_Y
    line_dir = os.path.join(
        OUTPUT_DIR,
        f"paragraph_{par_num}",
        f"line_{line_num}"
    )
    os.makedirs(line_dir, exist_ok=True)

    # Sort words left-to-right based on 'left' coordinate
    words.sort(key=lambda x: x["left"])

    for word_idx, word in enumerate(words, start=1):
        
        # --------------------
        # Crop word image
        # --------------------
        x1, y1 = word["left"], word["top"]
        x2, y2 = word["right"], word["bottom"]

        word_img = gray[y1:y2, x1:x2]

        word_dir = os.path.join(line_dir, f"word_{word_idx}")
        char_dir = os.path.join(word_dir, "chars")

        os.makedirs(char_dir, exist_ok=True)
        cv2.imwrite(os.path.join(word_dir, "word.png"), word_img)

        # --------------------
        # Character segmentation (WHITESPACE SPLITTING)
        # --------------------
        # Threshold the specific word image
        _, binary = cv2.threshold(
            word_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Vertical projection (count black pixels per column)
        # binary==0 checks for black pixels (text)
        col_sum = np.sum(binary == 0, axis=0)

        char_ranges = []
        in_char = False
        start_x = 0

        for i, val in enumerate(col_sum):
            if val > 0 and not in_char:
                in_char = True
                start_x = i
            elif val == 0 and in_char:
                end_x = i
                if end_x - start_x >= MIN_CHAR_WIDTH:
                    char_ranges.append((start_x, end_x))
                in_char = False

        # Handle edge case where character ends at the image boundary
        if in_char:
            char_ranges.append((start_x, len(col_sum)))

        # --------------------
        # Save characters (LEFT → RIGHT)
        # --------------------
        for char_idx, (cx1, cx2) in enumerate(char_ranges, start=1):
            
            char_img = binary[:, cx1:cx2]

            # Trim top/bottom whitespace
            row_sum = np.sum(char_img == 0, axis=1)
            non_empty = np.where(row_sum > 0)[0]

            if len(non_empty) == 0:
                continue

            # Crop vertical
            cy1, cy2 = non_empty[0], non_empty[-1] + 1
            char_img = char_img[cy1:cy2, :]

            # Filter noise by height ratio
            if char_img.shape[0] < MIN_CHAR_HEIGHT_RATIO * word_img.shape[0]:
                continue

            cv2.imwrite(
                os.path.join(char_dir, f"char_{char_idx:03d}.png"),
                char_img
            )

# --------------------
# Final Result
# --------------------
print("Processing complete.")
print(f"Processed {len(records)} lines.")
# Optional: print first few records to verify
for r in records[:3]:
    print(r)