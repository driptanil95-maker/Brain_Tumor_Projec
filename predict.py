import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = "brain_tumor_unet.h5"

print("Loading trained model...")
model = load_model(MODEL_PATH)

# 2. Pick a valid image from your folder
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

if not os.path.exists(IMAGE_DIR):
    print(f"Error: Directory '{IMAGE_DIR}' does not exist!")
    exit()

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if not os.path.isdir(os.path.join(IMAGE_DIR, f))])

if len(image_files) == 0:
    print("No image files found in data/images/. Please drop some MRI images there.")
    exit()

sample_name = image_files[0]
img_path = os.path.join(IMAGE_DIR, sample_name)
mask_path = os.path.join(MASK_DIR, sample_name)

print(f"Using sample image: {sample_name}")

# 3. Read and preprocess the image
orig_img = cv2.imread(img_path)
if orig_img is None:
    print(f"Error: Could not read image at {img_path}. Check file format.")
    exit()

orig_img_resized = cv2.resize(orig_img, (IMG_WIDTH, IMG_HEIGHT))
input_img = orig_img_resized / 255.0
input_img = np.expand_dims(input_img, axis=0)

# Load true mask if it exists
true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if true_mask is not None:
    true_mask = cv2.resize(true_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
else:
    true_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))

# 4. Predict using your trained U-Net
print("Running prediction...")
pred_probs = model.predict(input_img)
pred_mask = np.argmax(pred_probs[0], axis=-1)

# 5. Visualize and Save the Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original MRI Scan")
plt.imshow(cv2.cvtColor(orig_img_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(true_mask, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Model Prediction (Tumor/Edema)")
plt.imshow(pred_mask, cmap='jet')
plt.axis('off')

plt.tight_layout()

output_path = "prediction_result.png"
plt.savefig(output_path)
print(f"\nSuccess! Visualization saved as '{output_path}'. Check your VS Code file explorer to view it!")