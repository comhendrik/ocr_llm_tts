import cv2
import numpy as np
import os
import shutil

# --------------------------
# Helper Functions
# --------------------------

def resize_and_pad(img, size=64, padding=10):
    """
    Centers a binary image on a fixed size square canvas (white background).
    Prevents crash on tiny inputs by ensuring dimensions are at least 1px.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0: return None
    
    # 1. Calculate scaling to fit within (size - padding)
    max_dim = size - (padding * 2)
    
    scale = min(max_dim / h, max_dim / w)
    
    # FIX: Ensure dimensions are at least 1 to prevent cv2.error
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # Resize preserving aspect ratio
    try:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Skipping resize error: {e}")
        return None
    
    # 2. Create White Canvas
    canvas = np.full((size, size), 255, dtype=np.uint8)
    
    # 3. Center the image
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def get_segments(data, axis):
    """
    Finds start/end coordinates of non-zero pixel clusters.
    axis=1 for Lines (Horizontal), axis=0 for Chars (Vertical).
    """
    # Collapse the image into a single profile array
    projection = np.sum(data, axis=axis)
    
    segments = []
    start = -1
    
    for i, val in enumerate(projection):
        if val > 0 and start == -1:
            start = i # Start of a segment
        elif val == 0 and start != -1:
            segments.append((start, i)) # End of a segment
            start = -1
            
    # Capture the last segment if it touches the edge
    if start != -1:
        segments.append((start, len(projection)))
        
    return segments

# --------------------------
# Main Processing Logic
# --------------------------

def process_image_opencv(image_path):
    output_dir = "processed_structure_cv2"
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # Create a copy for visualization
    debug_img = img.copy()
    h_img, w_img = img.shape[:2]

    # 2. Preprocess: Near White -> White, Near Black -> Black
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold 150: Pixels brighter than 150 become 0 (Black), darker become 255 (White text)
    # We use BINARY_INV because OpenCV projection calculations work best on 'white' pixels against black background
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 3. Detect Lines (Horizontal Projection)
    line_segments = get_segments(thresh, axis=1)

    print(f"Detected {len(line_segments)} lines.")

    for l_idx, (l_start, l_end) in enumerate(line_segments):
        # -- Visualization: Draw Box around Line --
        cv2.rectangle(debug_img, (0, l_start), (w_img, l_end), (0, 255, 0), 2)
        
        # Create Folder for Line
        line_dir = os.path.join(output_dir, f"line_{l_idx}")
        os.makedirs(line_dir, exist_ok=True)

        # Crop the line
        line_img = thresh[l_start:l_end, :]

        # 4. Detect "Blobs" (Potential Characters/Words)
        char_blobs = get_segments(line_img, axis=0)
        
        if not char_blobs: continue

        # -------------------------------------------------------------------------
        # ROBUST DYNAMIC WORD SPACING LOGIC
        # -------------------------------------------------------------------------
        if len(char_blobs) > 1:
            # Calculate all distances between connected components
            gaps = [char_blobs[i+1][0] - char_blobs[i][1] for i in range(len(char_blobs)-1)]
            
            # Step A: Find the Median. 
            median_gap = np.median(gaps)
            
            # Step B: Filter out the "Word Gaps" to find the "True Average Character Gap"
            # We assume anything less than 3x the median is just a gap between letters.
            normal_char_gaps = [g for g in gaps if g < median_gap * 3]
            
            if normal_char_gaps:
                avg_char_gap = sum(normal_char_gaps) / len(normal_char_gaps)
            else:
                avg_char_gap = median_gap # Fallback
            
            # Step C: Set Threshold
            space_threshold = max(avg_char_gap * 2.5, 6)
        else:
            space_threshold = 1000 
        # -------------------------------------------------------------------------

        word_idx = 0
        char_idx_in_word = 0
        
        # Ensure first word directory exists
        word_dir = os.path.join(line_dir, f"word_{word_idx}")
        os.makedirs(word_dir, exist_ok=True)

        for i, (c_start, c_end) in enumerate(char_blobs):
            
            # --- NOISE FILTER ---
            # If the blob is less than 2 pixels wide, it's likely noise. Skip it.
            if (c_end - c_start) < 2:
                continue

            # Crop Character (vertically from the line crop)
            # IMPORTANT: We keep full line height (l_start to l_end) for alignment
            char_crop = line_img[:, c_start:c_end]

            # 5. Format Output (Invert back to Black-on-White & Pad)
            char_inverted = cv2.bitwise_not(char_crop)
            final_char = resize_and_pad(char_inverted, size=64, padding=10)

            if final_char is not None:
                save_path = os.path.join(word_dir, f"char_{char_idx_in_word}.png")
                cv2.imwrite(save_path, final_char)
            
            # Check gap to next blob to decide if we need a new word folder
            if i < len(char_blobs) - 1:
                next_start = char_blobs[i+1][0]
                current_gap = next_start - c_end
                
                if current_gap > space_threshold:
                    word_idx += 1
                    char_idx_in_word = 0
                    word_dir = os.path.join(line_dir, f"word_{word_idx}")
                    os.makedirs(word_dir, exist_ok=True)
                else:
                    char_idx_in_word += 1

    # Show the Line Detection Debug Image
    # (Optional: Comment out cv2.imshow if running on a headless server)
    try:
        cv2.imshow("Detected Lines (OpenCV)", debug_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass # Handle cases where display isn't available
        
    cv2.imwrite("debug_lines_detected.jpg", debug_img) # Save reference
    print(f"Processing Complete. Output saved to: {os.path.abspath(output_dir)}")
