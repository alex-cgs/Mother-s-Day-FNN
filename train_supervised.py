import numpy as np
import json
import pandas as pd
from PIL import Image
import random as rd
import os

learning_rate = 0.001  # Initial learning rate
classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

def sigmoid(x):
    exp_neg_x = np.exp(np.clip(-x, a_min=None, a_max=700))  # Clip to prevent overflow
    return 1 / (1 + exp_neg_x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Agent:
    def __init__(self):
        self.w1 = np.random.randn(196608, 256) * np.sqrt(2. / 196608)  # He initialization
        self.w2 = np.random.randn(256, 128) * np.sqrt(2. / 256)  # He initialization
        self.w3 = np.random.randn(128, 14) * np.sqrt(2. / 128)  # He initialization
        self.b1 = np.zeros((1, 256))
        self.b2 = np.zeros((1, 128))
        self.b3 = np.zeros((1, 14))
        
    def propagate(self, inp):
        self.Z1 = inp.dot(self.w1) + self.b1
        self.A1 = relu(self.Z1)
        
        self.Z2 = self.A1.dot(self.w2) + self.b2
        self.A2 = relu(self.Z2)
        
        self.Z3 = self.A2.dot(self.w3) + self.b3
        A3 = softmax(self.Z3)
        return A3
    
    def backpropagate(self, inp, out):
        m = inp.shape[0]
        
        # Forward propagation
        A3 = self.propagate(inp)
        
        # Loss
        loss = -np.sum(out * np.log(A3 + 1e-8)) / m  # Cross-entropy loss
        
        # Backpropagation
        dZ3 = A3 - out
        dW3 = self.A2.T.dot(dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        dA2 = dZ3.dot(self.w3.T)
        dZ2 = dA2 * np.where(self.Z2 > 0, 1, 0)
        
        dW2 = self.A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2.dot(self.w2.T)
        dZ1 = dA1 * np.where(self.Z1 > 0, 1, 0)
        
        dW1 = inp.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Gradient clipping to prevent exploding gradients
        for dparam in [dW3, db3, dW2, db2, dW1, db1]:
            np.clip(dparam, -1, 1, out=dparam)
        
        # Update weights and biases
        self.w3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def save_agent(agent):
    data = {
        'w1': agent.w1.tolist(),
        'w2': agent.w2.tolist(),
        'w3': agent.w3.tolist(),
        'b1': agent.b1.tolist(),
        'b2': agent.b2.tolist(),
        'b3': agent.b3.tolist()
    }
    with open('/nn/s_nn.json', 'w') as file:
        json.dump(data, file)

def load_agent():
    with open('/nn/s_nn.json', 'r') as file:
        data = json.load(file)
    
    agent = Agent()
    agent.w1 = np.array(data['w1'])
    agent.w2 = np.array(data['w2'])
    agent.w3 = np.array(data['w3'])
    agent.b1 = np.array(data['b1'])
    agent.b2 = np.array(data['b2'])
    agent.b3 = np.array(data['b3'])
    
    return agent

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

def one_hot_encode(label, num_classes):
    one_hot = np.zeros((1, num_classes))
    one_hot[0, classes.index(label)] = 1
    return one_hot

# Load image templates from train.csv
train_data = pd.read_csv('db/train.csv')

# Create agent
agent = Agent()

# Train on 200 images with more iterations
for i in range(200):
    label = classes[rd.randint(0, 13)]
    img_path = select_random_image(f"db/train/{label}/")
    inp = preprocess_image(img_path).reshape(1, -1)
    
    while inp.shape[1] != 196608:
        label = classes[rd.randint(0, 13)]
        img_path = select_random_image(f"db/train/{label}/")
        inp = preprocess_image(img_path).reshape(1, -1)
    
    out = one_hot_encode(label, len(classes))
    
    A3 = agent.propagate(inp)
    prediction = np.argmax(A3)
    print(f"Predicted label: {classes[prediction]}, Label: {label}")

    loss = -np.sum(out * np.log(A3 + 1e-8)) / inp.shape[0]
    agent.backpropagate(inp, out)

    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss}")
        save_agent(agent)

# Testing loop
for i in range(20):
    label = classes[rd.randint(0, 13)]
    img_path = select_random_image(f"db/train/{label}/")
    inp = preprocess_image(img_path).reshape(1, -1)
    
    while inp.shape[1] != 196608:
        label = classes[rd.randint(0, 13)]
        img_path = select_random_image(f"db/train/{label}/")
        inp = preprocess_image(img_path).reshape(1, -1)
    
    out = one_hot_encode(label, len(classes))
    A3 = agent.propagate(inp)
    prediction = np.argmax(A3)
    print(f"Test Image Prediction: {classes[prediction]}, Actual Label: {label}")

save_agent(agent)