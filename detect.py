import cv2
import numpy as np
import os
import tensorflow as tensorflow
from tensorflow.keras.models import load_model
from character_splitting import process_image_opencv

# --------------------------
# Configuration
# --------------------------
DATA_DIR = "processed_structure_cv2"  # The folder created by the previous script
MODEL_PATH = "ocr_text_model.keras"    # Path to your trained model
IMG_SIZE = 64

# Define your label map (Index -> Character)
# This MUST match exactly how your model was trained.
# Example: If your model outputs 0 for 'A', 1 for 'B', etc.
chars_lower = "abcdefghijklmnopqrstuvwxyz"
chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
numbers = "0123456789"
german_specials = "ß"
punctuation = ".,;:!?()+-"

LABELS = list(chars_lower + chars_upper + numbers + german_specials + punctuation)

# --------------------------
# Helper Functions
# --------------------------

def natural_sort_key(s):
    """
    Sorts strings with numbers naturally (e.g., line_2 comes before line_10).
    Expected format: 'prefix_number' (e.g., 'line_12')
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def predict_character(model, image_path):
    """
    Loads a 64x64 image, preprocesses it, and returns the predicted character.
    """
    # 1. Load as Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return "?"

    # 2. Preprocess to match training (Resize + Normalize)
    # Ensure it's 64x64 (just in case)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values to 0-1 range (Standard for Neural Networks)
    img = img.astype('float32') / 255.0
    
    # 3. Reshape for the model: (Batch_Size, Height, Width, Channels)
    # Becomes (1, 64, 64, 1)
    img = np.expand_dims(img, axis=-1) 
    img = np.expand_dims(img, axis=0)

    # 4. Predict
    prediction = model.predict(img, verbose=0)
    class_idx = np.argmax(prediction)
    
    # Safety check if model predicts outside our label range
    if class_idx < len(LABELS):
        return LABELS[class_idx]
    else:
        return "?"

# --------------------------
# Main Logic
# --------------------------

def reconstruct_text_from_images(data_dir, model_path):
    print("Loading model...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy prediction mode (for testing without a model file).")
        model = None

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    full_text = []

    # 1. Iterate Lines
    lines = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))], key=natural_sort_key)
    
    print("\n--- OCR RECONSTRUCTION ---")
    
    for line_folder in lines:
        line_path = os.path.join(data_dir, line_folder)
        line_text = []

        # 2. Iterate Words inside the Line
        words = sorted([d for d in os.listdir(line_path) if os.path.isdir(os.path.join(line_path, d))], key=natural_sort_key)

        for word_folder in words:
            word_path = os.path.join(line_path, word_folder)
            word_chars = []

            # 3. Iterate Characters inside the Word
            chars = sorted([f for f in os.listdir(word_path) if f.endswith('.png')], key=natural_sort_key)

            for char_file in chars:
                char_path = os.path.join(word_path, char_file)
                
                if model:
                    predicted_char = predict_character(model, char_path)
                else:
                    # Dummy fallback if no model is provided
                    predicted_char = "?" 
                
                word_chars.append(predicted_char)
            
            # Form the word and add to line
            full_word = "".join(word_chars)
            line_text.append(full_word)

        # Form the full line (joined by spaces)
        full_line = " ".join(line_text)
        full_text.append(full_line)
        print(f"{line_folder}: {full_line}")

    # --------------------------
    # Final Output
    # --------------------------
    print("\n--- FINAL TEXT ---")
    final_output = "\n".join(full_text)
    return final_output



def detect_text_in_image(image_path: str) -> str:
    process_image_opencv(image_path)
    return reconstruct_text_from_images(DATA_DIR, MODEL_PATH)