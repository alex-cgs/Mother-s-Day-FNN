import numpy as np
import json
import pandas as pd
from PIL import Image
import random as rd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

input_dim = 196608
num_clusters = 14
num_iterations = 500

def select_random_image(directory):
    files = os.listdir(directory)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    selected_image = rd.choice(image_files)
    selected_image_path = os.path.join(directory, selected_image)
    return selected_image_path

def preprocess_image(image_path):
    img = Image.open(image_path).resize((256, 256))  # Ensure consistent size
    img_array = np.array(img).flatten()  # Flatten the image to a 1D array
    if img_array.shape[0] != input_dim:
        raise ValueError(f"Image at {image_path} does not have the expected size of {input_dim}.")
    return img_array

def save_model(model, filepath):
    data = {
        'cluster_centers': model.cluster_centers_.tolist(),
    }
    with open(filepath, 'w') as file:
        json.dump(data, file)

def load_model(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    model = KMeans(n_clusters=num_clusters)
    model.cluster_centers_ = np.array(data['cluster_centers'])
    return model

# Create directory if not exists
os.makedirs('/nn', exist_ok=True)

# Load image templates from train.csv
train_data = pd.read_csv('db/train.csv')

# Prepare the dataset
dataset = []
image_paths = []

for label in classes:
    for _ in range(50):  # Select 50 random images per class for training
        img_path = select_random_image(os.path.join("db/train", label))
        inp = preprocess_image(img_path)
        dataset.append(inp)
        image_paths.append(img_path)

dataset = np.array(dataset)

# Train K-means
kmeans = KMeans(n_clusters=num_clusters, max_iter=num_iterations, random_state=42)
kmeans.fit(dataset)

# Save trained model
save_model(kmeans, '/nn/kmeans_model.json')

# Test loop
for i in range(20):
    label = classes[rd.randint(0, num_clusters - 1)]
    img_path = select_random_image(os.path.join("db/train", label))
    inp = preprocess_image(img_path).reshape(1, -1)
    
    # Predict the cluster
    cluster = kmeans.predict(inp)
    print(f"Image belongs to cluster: {cluster[0]}")

# Function to plot images at specific coordinates
def plot_image_at_coordinates(ax, image_path, coords):
    img = Image.open(image_path).resize((32, 32))  # Resize image for better display on plot
    img = np.array(img)
    imagebox = OffsetImage(img, zoom=1)
    ab = AnnotationBbox(imagebox, coords, frameon=False)
    ax.add_artist(ab)

# Visualize clusters with images
def visualize_clusters(kmeans, dataset, image_paths):
    plt.figure(figsize=(12, 12))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    ax = plt.gca()
    
    for i in range(num_clusters):
        cluster_points = dataset[kmeans.labels_ == i]
        cluster_images = [image_paths[j] for j in range(len(image_paths)) if kmeans.labels_[j] == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')
        
        # Plot images at their respective coordinates
        for point, img_path in zip(cluster_points, cluster_images):
            plot_image_at_coordinates(ax, img_path, point)

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='X')
    plt.title('K-means Clustering with Images')
    plt.legend()
    plt.show()

# Reduce dimensions to 2D for visualization using PCA
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

pca = PCA(n_components=2)
dataset_2d = pca.fit_transform(dataset)
kmeans_2d = KMeans(n_clusters=num_clusters, max_iter=num_iterations, random_state=42)
kmeans_2d.fit(dataset_2d)
visualize_clusters(kmeans_2d, dataset_2d, image_paths)
