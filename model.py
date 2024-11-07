import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os

# Load and preprocess the dataset
def load_dataset(data_path, img_size=(224, 224)):
    images = []
    labels = []
    label_map = {}
    label_counter = 0
    
    for dirpath, dirnames, filenames in os.walk(data_path):
        for dirname in dirnames:
            label_map[dirname] = label_counter
            label_counter += 1
        break
    
    for label in label_map.keys():
        folder_path = os.path.join(data_path, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label_map[label])
    
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels, label_map

# Load data
data_path = 'path/to/IMFDB'
X, y, label_map = load_dataset(data_path)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save as TFLite model
model.save('efficientnetb0_face_recognition.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('efficientnetb0_face_recognition.h5')
tflite_model = converter.convert()
with open('efficientnetb0_face_recognition.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model training and conversion to TFLite complete.")
