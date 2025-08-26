import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os

# Dataset path (adjust for your system)
DATASET_PATH = r'D:\Devanagri_character_classification\Images'  # Update this path to your Devanagari dataset folder

# Number of classes for Devanagari characters (adjust if your dataset has a different number of classes)
NUM_CLASSES = 46

# Function to load images and labels
def load_data():
    images = []
    labels = []
    
    # Loop through folders (0-45 for 46 classes)
    for label in range(NUM_CLASSES):
        folder_path = os.path.join(DATASET_PATH, str(label))  # Folder for each character
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} not found, skipping.")
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Open and process the image
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to match MNIST size
                img = np.array(img) / 255.0  # Normalize the pixel values (0-1)
                images.append(img)  # Append to the images list
                labels.append(label)  # Append the corresponding label (0-45)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load data from the specified path
X, y = load_data()

# Preprocessing
X = np.expand_dims(X, axis=-1)  # Add the channel dimension for grayscale images
y = to_categorical(y, NUM_CLASSES)  # One-hot encode the labels (for 46 Devanagari characters)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save('X_test.npy', X_test)  # Save test images
np.save('y_test.npy', y_test)  # Save test labels


# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # 46 classes for Devanagari characters
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=128)

# Save the model after training

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2%}")

model.save('devanagari_character_cnn.h5')
print("Model training complete and saved as 'devanagari_character_cnn.h5'.")
