import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(128, 128, 3), num_classes=4):
    inputs = layers.Input(input_shape)

    # --- ENCODER (Contracting Path) ---
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- BOTTLENECK ---
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # --- DECODER (Expanding Path) ---
    u3 = layers.UpSampling2D((2, 2))(c4)
    u3 = layers.concatenate([u3, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u1 = layers.UpSampling2D((2, 2))(c6)
    u1 = layers.concatenate([u1, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    # --- OUTPUT LAYER ---
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="BrainTumor_UNet")
    
    # CRITICAL: This line must be here so python returns the model object!
    return model