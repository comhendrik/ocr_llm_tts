import cv2
import numpy as np
import os
import tensorflow as tf
from splitting_tesseract import process_image_hybrid

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "ocr_text_model.keras"
IMG_SIZE = 64

# Define your label map (Index -> Character)
chars_lower = "abcdefghijklmnopqrstuvwxyz"
chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
numbers = "0123456789"
german_specials = "ß"
punctuation = ".,;:!?()+-"
LABELS = list(chars_lower + chars_upper + numbers + german_specials + punctuation)

# --------------------------
# Helper Functions
# --------------------------

def predict_from_memory(model, img_array):
    """
    Takes a pre-loaded numpy image array (64x64), preprocesses it, 
    and returns the predicted character.
    """
    # 1. Image is already a numpy array from process_image_hybrid
    # It should be shape (64, 64) or (64, 64, 1)
    
    # Safety: Ensure it is 64x64
    if img_array.shape[0] != IMG_SIZE or img_array.shape[1] != IMG_SIZE:
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    # 2. Normalize pixel values to 0-1 range
    img_array = img_array.astype('float32') / 255.0
    
    # 3. Reshape for the model: (Batch_Size, Height, Width, Channels)
    # Input is usually (64, 64). We need (1, 64, 64, 1)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1) # Becomes (64, 64, 1)
    
    img_input = np.expand_dims(img_array, axis=0)      # Becomes (1, 64, 64, 1)

    # 4. Predict
    prediction = model.predict(img_input, verbose=0)
    class_idx = np.argmax(prediction)
    
    # 5. Map to Character
    if class_idx < len(LABELS):
        return LABELS[class_idx]
    else:
        return "?"

def reconstruct_text_from_memory(images_array, model_path):
    """
    Iterates through the nested list structure provided by process_image_hybrid.
    Structure: [ Lines [ Words [ Characters (Images) ] ] ]
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return ""

    full_text_lines = []

    
    # 1. Iterate Lines (List of Lists)
    for line_idx, line_words in enumerate(images_array):
        line_text_parts = []
        
        # 2. Iterate Words (List of Lists of Images)
        for word_chars in line_words:
            word_string = ""
            
            # 3. Iterate Characters (Numpy Arrays)
            for char_img in word_chars:
                predicted_char = predict_from_memory(model, char_img)
                word_string += predicted_char
            
            line_text_parts.append(word_string)
        
        # Join words with a space to form the line
        full_line_text = " ".join(line_text_parts)
        full_text_lines.append(full_line_text)
        print(f"Line {line_idx}: {full_line_text}")

    # --------------------------
    # Final Output
    # --------------------------
    final_output = "\n".join(full_text_lines)
    return final_output

# --------------------------
# Main Entry Point
# --------------------------

def detect_text_in_image(image_path: str) -> str:
    """
    1. Processes the image to extract character arrays (in memory).
    2. Passes those arrays to the neural network for prediction.
    3. Returns the full reconstructed text.
    """
    # Step 1: Get the data structure (No saving to disk needed for logic)
    # We set save_images=False to speed it up, or True if you still want debug images.
    images_array = process_image_hybrid(image_path, save_images=False)
    
    if not images_array:
        return "Error: No text detected or image load failed."

    # Step 2: Reconstruct text from the array
    return reconstruct_text_from_memory(images_array, MODEL_PATH)