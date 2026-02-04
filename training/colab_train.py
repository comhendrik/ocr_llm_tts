import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================


# --- CONFIG ---
# Path to your .npz file in Google Drive
# If you uploaded it to the root of your Drive, this path is likely correct:
DATA_PATH = 'german_chars_2000_aug.npz'

# Where to save the model checkpoints
CHECKPOINT_DIR = 'ocr_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
TEST_SPLIT = 0.2  # 20% for testing

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print(f"\nLoading data from {DATA_PATH}...")

try:
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    chars = data['chars']
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: File not found at {DATA_PATH}. Please check the path.")
    raise

# Normalize images: Convert 0-255 to 0.0-1.0 (Crucial for Neural Networks)
print("Normalizing images...")
X = X.astype('float32') / 255.0

# Reshape: Add the 'Channel' dimension.
# TensorFlow expects (Batch, Height, Width, Channels) -> (N, 128, 128, 1)
# If your images are grayscale, channels = 1.
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

print(f"Input Shape: {X.shape}")

# One-Hot Encoding for Labels
num_classes = len(chars)
print(f"Number of classes: {num_classes}")
y = keras.utils.to_categorical(y, num_classes)

# Split into Train and Test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
# A solid CNN architecture for 128x128 images
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 4 (Optional for deeper learning)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten & Dense Layers
    layers.Flatten(),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Output layer
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 4. CALLBACKS (SAVING & LOGGING)
# ==========================================

# Checkpoint: Save the model after every epoch to Google Drive
# We save only the 'best' model based on validation loss to save space, 
# OR you can set save_best_only=False to save absolutely everything.
checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.keras")

checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False, # Save the whole model (architecture + weights)
    monitor='val_accuracy',
    mode='max',
    save_best_only=False,    # Change to True if you only want to save if it improves
    verbose=1
)

# Early Stopping: Stop if training doesn't improve for 3 epochs
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==========================================
# 5. TRAINING
# ==========================================
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, early_stopping]
)

# ==========================================
# 6. EVALUATION & SAVING FINAL MODEL
# ==========================================
print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save the final model explicitly
final_save_path = os.path.join(CHECKPOINT_DIR, 'final_german_ocr_model.keras')
model.save(final_save_path)
print(f"Final model saved to: {final_save_path}")

# Plot History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()