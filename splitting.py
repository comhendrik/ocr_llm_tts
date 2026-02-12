import cv2
import pytesseract
import numpy as np
import os
import platform
import shutil
from pytesseract import Output

# --------------------------
# Helper Functions
# --------------------------

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def resize_and_pad(img, size=64, padding=10):
    """
    Centers a binary image on a fixed size square canvas (white background).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0: return None
    
    # 1. Calculate scaling
    max_dim = size - (padding * 2)
    scale = min(max_dim / h, max_dim / w)
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # Resize
    try:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        return None
    
    # 2. White Canvas
    canvas = np.full((size, size), 255, dtype=np.uint8)
    
    # 3. Center
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def get_segments(data, axis):
    """
    Finds start/end coordinates of non-zero pixel clusters.
    axis=0 for Vertical Projection (splitting Chars).
    """
    projection = np.sum(data, axis=axis)
    
    segments = []
    start = -1
    
    for i, val in enumerate(projection):
        if val > 0 and start == -1:
            start = i 
        elif val == 0 and start != -1:
            segments.append((start, i)) 
            start = -1
            
    if start != -1:
        segments.append((start, len(projection)))
        
    return segments


def split_image_into_characters(image_path, save_images=True):
    """
    Returns: A nested list structure: [ Lines [ Words [ Characters (numpy arrays) ] ] ]
    """
    output_dir = "characters"
    
    # Setup Output Directory only if saving
    if save_images:
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    debug_img = img.copy()

    # ---------------------------------------------------------
    # STEP A: Detect Lines using PYTESSERACT
    # ---------------------------------------------------------
    
    d = pytesseract.image_to_data(gray, config='--psm 6', output_type=Output.DICT)
    
    lines_map = {}
    n_boxes = len(d['text'])
    
    for i in range(n_boxes):
        if int(d['conf'][i]) < 0: continue
            
        key = (d['block_num'][i], d['par_num'][i], d['line_num'][i])
        if key not in lines_map: lines_map[key] = []
            
        lines_map[key].append({
            'left': d['left'][i], 'top': d['top'][i],
            'width': d['width'][i], 'height': d['height'][i]
        })

    # ---------------------------------------------------------
    # STEP B: Process Lines -> Words -> Chars
    # ---------------------------------------------------------
    
    # Structure to return: [ [ [char, char], [word2] ], [line2] ]
    all_lines_data = []

    sorted_lines = sorted(lines_map.items(), key=lambda item: item[1][0]['top'])

    for l_idx, (key, words) in enumerate(sorted_lines):
        
        # Determine Line Bounding Box
        min_x = min(w['left'] for w in words)
        min_y = min(w['top'] for w in words)
        max_x = max(w['left'] + w['width'] for w in words)
        max_y = max(w['top'] + w['height'] for w in words)
        
        # Add padding
        pad = 2
        min_x = max(0, min_x - pad)
        min_y = max(0, min_y - pad)
        max_x = min(img.shape[1], max_x + pad)
        max_y = min(img.shape[0], max_y + pad)

        if save_images:
            cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            line_dir = os.path.join(output_dir, f"line_{l_idx}")
            os.makedirs(line_dir, exist_ok=True)

        line_crop_gray = gray[min_y:max_y, min_x:max_x]
        _, line_thresh = cv2.threshold(line_crop_gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        char_blobs = get_segments(line_thresh, axis=0)
        
        if not char_blobs: continue

        # --- Dynamic Word Spacing Logic ---
        if len(char_blobs) > 1:
            gaps = [char_blobs[i+1][0] - char_blobs[i][1] for i in range(len(char_blobs)-1)]
            median_gap = np.median(gaps)
            normal_char_gaps = [g for g in gaps if g < median_gap * 3]
            avg_char_gap = sum(normal_char_gaps) / len(normal_char_gaps) if normal_char_gaps else median_gap
            space_threshold = avg_char_gap * 2
        else:
            space_threshold = 1000 

        # Current storage for the line being processed
        current_line_words = []
        current_word_chars = []
        
        word_idx = 0
        char_idx_in_word = 0
        
        if save_images:
            word_dir = os.path.join(line_dir, f"word_{word_idx}")
            os.makedirs(word_dir, exist_ok=True)

        for i, (c_start, c_end) in enumerate(char_blobs):
            if (c_end - c_start) < 2: continue # Noise filter

            # Process Char
            char_crop = line_thresh[:, c_start:c_end]
            char_inverted = cv2.bitwise_not(char_crop)
            final_char = resize_and_pad(char_inverted, size=64, padding=10)

            if final_char is not None:
                # Add to memory array
                current_word_chars.append(final_char)
                
                # Save to disk
                if save_images:
                    save_path = os.path.join(word_dir, f"char_{char_idx_in_word}.png")
                    cv2.imwrite(save_path, final_char)
            
            # Check for new word
            if i < len(char_blobs) - 1:
                next_start = char_blobs[i+1][0]
                current_gap = next_start - c_end
                
                if current_gap > space_threshold:
                    # WORD COMPLETE: Store it and reset
                    if current_word_chars:
                        current_line_words.append(current_word_chars)
                        current_word_chars = []
                    
                    word_idx += 1
                    char_idx_in_word = 0
                    
                    if save_images:
                        word_dir = os.path.join(line_dir, f"word_{word_idx}")
                        os.makedirs(word_dir, exist_ok=True)
                else:
                    char_idx_in_word += 1

        # Append the last word of the line if it has characters
        if current_word_chars:
            current_line_words.append(current_word_chars)
        
        # Append the full line to the master list
        all_lines_data.append(current_line_words)

    if save_images:
        cv2.imwrite("debug_lines_tesseract.jpg", debug_img)
    
    return all_lines_data