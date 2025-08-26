import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
import random

# Load the model
model = load_model('devanagari_character_cnn.h5')

# Load the test dataset
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Mapping of class indices to Devanagari characters
devanagari_characters = [
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',  
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',  
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',  
    'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', '0', '1', '2', '3',  
    '4', '5', '6', '7', '8', '9'  
]

# Predict on a subset of test data
indices = random.sample(range(len(X_test)), 5)  # Randomly select 5 test samples
sample_images = X_test[indices]
sample_labels = y_test[indices]

predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(sample_labels, axis=1)

# Create the plot
fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # 2 rows, 5 columns
axes = axes.flatten()

for i, ax in enumerate(axes[:5]):
    ax.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f"True: {devanagari_characters[true_classes[i]]}\nPred: {devanagari_characters[predicted_classes[i]]}")

# Display predicted characters for additional examples (like the table in your image)
for i, ax in enumerate(axes[5:]):
    ax.text(0.5, 0.5, devanagari_characters[predicted_classes[i]], 
            fontsize=20, ha='center', va='center', color='black')
    ax.axis('off')

# Save and show the figure
plt.tight_layout()
plt.savefig('model_predictions.png')  # Save as an image
plt.show()
