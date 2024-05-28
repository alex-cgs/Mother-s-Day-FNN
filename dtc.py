import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random as rd

def select_random_image(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out non-image files (optional, depending on your use case)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Ensure there are image files in the directory
    if not image_files:
        raise ValueError(f"No image files found in directory {directory}")
    
    # Randomly select an image file
    selected_image = rd.choice(image_files)
    
    # Full path of the selected image
    selected_image_path = os.path.join(directory, selected_image)
    
    return selected_image_path

# Example usage
test_image_path = select_random_image('db/test/')
print(f"Selected image path: {test_image_path}")

# Constants
image_size = (256, 256)
input_dim = 256 * 256 * 3
classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 
           'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize(image_size)  # Ensure consistent size and RGB
        img_array = np.array(img).flatten()  # Flatten the image
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_dataset(directory):
    X, y = [], []
    for class_label in classes:
        class_dir = os.path.join(directory, class_label)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(class_dir, img_file)
                img_array = preprocess_image(img_path)
                if img_array is not None:
                    X.append(img_array)
                    y.append(classes.index(class_label))
    return np.array(X), np.array(y)

# Load dataset
X, y = load_dataset('db/train/')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Function to classify a new image
def classify_image(image_path, classifier):
    img_array = preprocess_image(image_path).reshape(1, -1)
    prediction = classifier.predict(img_array)
    return classes[prediction[0]]

# Test with a new image
test_image_path = select_random_image('db/test/')
predicted_class = classify_image(test_image_path, clf)
print(f"Predicted class: {predicted_class}")
