import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import os
import cv2

# --- Configuration ---
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
DATASET_PATH = "train_data" # Your new folder
MODEL_PATH = "models/mask_detector.h5"

def prepare_data(dataset_path):
    print("[INFO] Loading images from 'masked' and 'unmasked' folders...")
    data = []
    labels = []
    
    # We iterate only through the two categories we created
    for category in ["masked", "unmasked"]:
        path = os.path.join(dataset_path, category)
        if not os.path.exists(path):
            print(f"[WARNING] Path {path} not found. Skipping.")
            continue
            
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (224, 224))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    data.append(image)
                    labels.append(category)
                except Exception as e:
                    print(f"[ERROR] Could not load {img_name}: {e}")

    # Convert to NumPy and Normalize
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    # One-hot encoding for labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    # If binary, fit_transform returns 1D array; we need 2D for Categorical Crossentropy
    if labels.shape[1] == 1:
        labels = to_categorical(labels)
    else:
        labels = to_categorical(labels) # For safety

    # Split into 80% training, 20% testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    
    return trainX, testX, trainY, testY, lb

# --- Model Building Functions remain the same ---
def build_model(width, height, depth, classes):
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(height, width, depth)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(classes, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False
    return model

if __name__ == "__main__":
    trainX, testX, trainY, testY, lb = prepare_data(DATASET_PATH)
    
    # Augmentation helps the model learn from small datasets
    aug = ImageDataGenerator(
        rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
        fill_mode="nearest")

    model = build_model(224, 224, 3, len(lb.classes_))
    # Using 'Adam' directly (ensure your import matches: from tensorflow.keras.optimizers import Adam)
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Training head...")
    model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    print(f"[INFO] Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH, save_format="h5")
    print("[SUCCESS] Training complete.")