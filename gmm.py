import numpy as np
import json
import pandas as pd
from PIL import Image
import random as rd
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

input_dim = 196608
reduced_dim = 50
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
    img_array = np.array(img).flatten()  # 1D array
    if img_array.shape[0] != input_dim:
        raise ValueError(f"Image at {image_path} does not have the expected size of {input_dim}.")
    return img_array

def save_model(pca, gmm, filepath):
    data = {
        'pca_components': pca.components_.tolist(),
        'pca_mean': pca.mean_.tolist(),
        'gmm_means': gmm.means_.tolist(),
        'gmm_covariances': gmm.covariances_.tolist(),
        'gmm_precisions_cholesky': gmm.precisions_cholesky_.tolist(),
        'gmm_weights': gmm.weights_.tolist(),
    }
    with open(filepath, 'w') as file:
        json.dump(data, file)

def load_model(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    pca = PCA(n_components=reduced_dim)
    pca.components_ = np.array(data['pca_components'])
    pca.mean_ = np.array(data['pca_mean'])
    
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.means_ = np.array(data['gmm_means'])
    gmm.covariances_ = np.array(data['gmm_covariances'])
    gmm.precisions_cholesky_ = np.array(data['gmm_precisions_cholesky'])
    gmm.weights_ = np.array(data['gmm_weights'])
    
    return pca, gmm

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

# Apply PCA for 1D
pca = PCA(n_components=reduced_dim)
dataset_reduced = pca.fit_transform(dataset)

# Train Gaussian Mixture Model on the reduced dataset
gmm = GaussianMixture(n_components=num_clusters, max_iter=num_iterations, random_state=42)
gmm.fit(dataset_reduced)

# Save trained model
save_model(pca, gmm, '/nn/u_nn.json')

# Test loop
for i in range(20):
    label = classes[rd.randint(0, num_clusters - 1)]
    img_path = select_random_image(os.path.join("db/train", label))
    inp = preprocess_image(img_path).reshape(1, -1)
    
    inp_reduced = pca.transform(inp)
    
    cluster = gmm.predict(inp_reduced)
    print(f"Image belongs to cluster: {cluster[0]}")

# Function to plot images at specific coordinates
def plot_image_at_coordinates(ax, image_path, coords):
    img = Image.open(image_path).resize((32, 32))  # Resize image for better display on plot
    img = np.array(img)
    imagebox = OffsetImage(img, zoom=1)
    ab = AnnotationBbox(imagebox, coords, frameon=False)
    ax.add_artist(ab)

# Visualize clusters with images
def visualize_clusters(pca, gmm, dataset, image_paths):
    dataset_reduced = pca.transform(dataset)  # Ensure we have 50 features for GMM
    dataset_2d = PCA(n_components=2).fit_transform(dataset_reduced)  # Reduce to 2D for visualization
    
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot the data points
    for i in range(num_clusters):
        cluster_points = dataset_2d[gmm.predict(dataset_reduced) == i]
        cluster_images = [image_paths[j] for j in range(len(image_paths)) if gmm.predict(dataset_reduced)[j] == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}', alpha=0.5)
        
        # Plot images at their respective coordinates
        for point, img_path in zip(cluster_points, cluster_images):
            plot_image_at_coordinates(ax, img_path, point)

    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=300, c='black', marker='X')
    plt.title('GMM Clustering with PCA-reduced data')
    plt.legend()
    plt.show()

# Visualize the clusters
visualize_clusters(pca, gmm, dataset, image_paths)
