import numpy as np
import os
import random as rd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the classes
classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

def select_random_image(directory):
    files = os.listdir(directory)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    selected_image = rd.choice(image_files)
    selected_image_path = os.path.join(directory, selected_image)
    return selected_image_path

def preprocess_image(image_path):
    img = Image.open(image_path).resize((256, 256))
    img_array = np.array(img).flatten()
    return img_array

def load_dataset(num_samples=200):
    X = []
    y = []
    
    for _ in range(num_samples):
        label = classes[rd.randint(0, len(classes) - 1)]
        img_path = select_random_image(f"db/train/{label}/")
        img_array = preprocess_image(img_path)
        
        while img_array.shape[0] != 196608:
            label = classes[rd.randint(0, len(classes) - 1)]
            img_path = select_random_image(f"db/train/{label}/")
            img_array = preprocess_image(img_path)
        
        X.append(img_array)
        y.append(label)
    
    return np.array(X), np.array(y)

X, y = load_dataset(num_samples=200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train, y_train)

def evaluate_classifier(clf):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    for true_label, pred_label in zip(y_test, y_pred):
        print(f'Test Image Prediction: {pred_label}, Actual Label: {true_label}')

# Evaluate the classifier
evaluate_classifier(clf)
