import numpy as np
import tensorflow as tf
from model import build_unet

# 1. Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 3
NUM_CLASSES = 4  # 0: Background, 1: Tumor Core, 2: Edema, 3: Enhancing Tumor
BATCH_SIZE = 8
EPOCHS = 5

print("Initializing U-Net model...")
model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)

# 2. Compile Model
# Sparse categorical crossentropy is used because masks contain integer labels (0, 1, 2, 3)
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# 3. Generating Mock Dataset (Replace this block later with your actual MRI dataset loader)
# X_train / X_val: Normalized MRI image slices shape -> (num_samples, 128, 128, 3)
# y_train / y_val: Segmentation masks shape -> (num_samples, 128, 128) containing values 0, 1, 2, or 3
print("Generating mock MRI data for testing training pipeline...")
X_train = np.random.rand(64, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS).astype(np.float32)
y_train = np.random.randint(0, NUM_CLASSES, size=(64, IMG_HEIGHT, IMG_WIDTH)).astype(np.int32)

X_val = np.random.rand(16, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS).astype(np.float32)
y_val = np.random.randint(0, NUM_CLASSES, size=(16, IMG_HEIGHT, IMG_WIDTH)).astype(np.int32)

# 4. Train the Model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 5. Evaluate Model Accuracy (Dice Coefficient for Boundaries & Edema)
print("\nEvaluating model performance...")
preds = model.predict(X_val)
pred_labels = np.argmax(preds, axis=-1)

def compute_dice_coefficient(y_true, y_pred, class_id):
    """Calculates Dice Similarity Coefficient for a specific tumor region/class."""
    true_class = (y_true == class_id)
    pred_class = (y_pred == class_id)
    intersection = np.logical_and(true_class, pred_class).sum()
    return (2. * intersection) / (true_class.sum() + pred_class.sum() + 1e-6)

# Calculate accuracy scores per feature region
core_dice = compute_dice_coefficient(y_val, pred_labels, class_id=1)
edema_dice = compute_dice_coefficient(y_val, pred_labels, class_id=2)
enhancing_dice = compute_dice_coefficient(y_val, pred_labels, class_id=3)

print(f"--- Evaluation Results ---")
print(f"Tumor Core Dice Score: {core_dice:.4f}")
print(f"Edema Detection Dice Score: {edema_dice:.4f}")
print(f"Enhancing Tumor Dice Score: {enhancing_dice:.4f}")

# Save the trained model weights
model.save("brain_tumor_unet.h5")
print("\nTraining complete! Model saved as 'brain_tumor_unet.h5'.")