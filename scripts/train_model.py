import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import h5py

# -----------------------------
# 1. DATASET PATH
# -----------------------------
data_dir = r"C:\Eyedetection\medical-eye-disease\dataset"
img_height, img_width = 128, 128

# -----------------------------
# 2. LOAD DATA
# -----------------------------
images = []
labels = []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)

    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    target_size=(img_height, img_width)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

                images.append(img_array)
                labels.append(class_name)

            except Exception as e:
                print(f"Skipping: {img_path} | Error: {e}")

images = np.array(images)
labels = np.array(labels)

# -----------------------------
# 3. ENCODE LABELS
# -----------------------------
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = tf.keras.utils.to_categorical(labels_encoded)

# Save label encoder (IMPORTANT for deployment)
os.makedirs("models", exist_ok=True)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels_categorical,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 5. DATA AUGMENTATION
# -----------------------------
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10
)

val_datagen = ImageDataGenerator()

train_data = train_datagen.flow(X_train, y_train, batch_size=32)
val_data = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)

# -----------------------------
# 6. MODEL
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 7. CHECKPOINT
# -----------------------------
model_checkpoint = ModelCheckpoint(
    "models/best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# -----------------------------
# 8. TRAINING
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[model_checkpoint]
)

# -----------------------------
# 9. EVALUATION
# -----------------------------
val_loss, val_acc = model.evaluate(val_data)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Predictions (IMPORTANT FIX)
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# F1 Score
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
print(f"\nF1 Score: {f1:.4f}")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=label_encoder.classes_
))

# -----------------------------
# 10. SAVE FINAL MODEL
# -----------------------------
tf.keras.models.save_model(model, "models/final_model.h5")

print("\nModel saved successfully!")