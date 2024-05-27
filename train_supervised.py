import numpy as np
import json
import pandas as pd
from PIL import Image
import random as rd
import os

learning_rate = 0.01

classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

def sigmoid(x):
    exp_neg_x = np.exp(np.clip(-x, a_min=None, a_max=700))  # Clip to prevent overflow
    return 1 / (1 + exp_neg_x)

def relu(x):
    return np.maximum(0, x)

class Agent:
    def __init__(self):
        self.w1 = np.random.uniform(-1, 1, size=(196608, 1400))
        self.w2 = np.random.uniform(-1, 1, size=(1400, 14))
        self.b1 = np.random.uniform(-1, 1, size=(1, 1400))
        self.b2 = np.random.uniform(-1, 1, size=(1, 14))
        
    def propagate(self, inp):
        A1 = sigmoid(inp.dot(self.w1) + self.b1)
        A2 = relu(A1.dot(self.w2) + self.b2)
        return A2
    
    def backpropagate(self, inp, out):
        m = inp.shape[0]
        
        # Forward propagation
        A1 = sigmoid(inp.dot(self.w1) + self.b1)
        A2 = relu(A1.dot(self.w2) + self.b2)
        
        # Calculate loss
        loss = np.sum((A2 - out) ** 2) / m
        
        # Backpropagation
        dA2 = 2 * (A2 - out) / m
        dZ2 = dA2 * np.where(A2 > 0, 1, 0)
        
        dW2 = A1.T.dot(dZ2)  # Should be (1400, m).dot(m, 14) = (1400, 14)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.w2.T)
        dZ1 = dA1 * sigmoid(A1) * (1 - sigmoid(A1))
        
        dW1 = inp.T.dot(dZ1)  # Should be (196608, m).dot(m, 1400) = (196608, 1400)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.w2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def save_agent(agent):
    data = {
        'w1': agent.w1.tolist(),
        'w2': agent.w2.tolist(),
        'b1': agent.b1.tolist(),
        'b2': agent.b2.tolist()
    }
    with open('/nn/s_nn.json', 'w') as file:
        json.dump(data, file)

def load_agent():
    with open('/nn/s_nn.json', 'r') as file:
        data = json.load(file)
    
    agent = Agent()
    agent.w1 = np.array(data['w1'])
    agent.w2 = np.array(data['w2'])
    agent.b1 = np.array(data['b1'])
    agent.b2 = np.array(data['b2'])
    
    return agent

def select_random_image(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter out non-image files (optional, depending on your use case)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Randomly select an image file
    selected_image = rd.choice(image_files)
    
    # Full path of the selected image
    selected_image_path = os.path.join(directory, selected_image)
    
    return selected_image_path

# Load image templates from train.csv
train_data = pd.read_csv('db/train.csv')

# Create agent
agent = Agent()

# Train on 500 images
for i in range(500):
    label = classes[rd.randint(0, 13)]  # Select a random class label
    # Get input and output for current image
    img_path = select_random_image("db/train/" + label + "/")  # Correct path formatting
    # print(label)
    # print(img_path)
    img = Image.open(img_path)
    inp = np.array(img).reshape(1, -1)
    
    while inp.shape[1] != 196608:
        label = classes[rd.randint(0, 13)]  # Select a random class label
        # Get input and output for current image
        img_path = select_random_image("db/train/" + label + "/")  # Correct path formatting
        # print(label)
        # print(img_path)
        img = Image.open(img_path)
        inp = np.array(img).reshape(1, -1)
    
    # One-hot encode the label
    out = np.zeros((1, 14))
    out[0, classes.index(label)] = 1  # Use index of the class label
    
    # Forward propagation
    A2 = agent.propagate(inp)
    prediction = np.argmax(A2)
    print(f"Predicted label: {classes[prediction]}, Label: {label}")

    # Calculate loss using the predicted label index
    loss = (A2[0, prediction] - 1) ** 2 / inp.shape[0]

    # Backpropagation
    agent.backpropagate(inp, out)

    # Print progress every iteration
    if i % 10 == 0:
        print(f"Iteration: {i}, Loss: {loss}")

        # Save trained agent
        save_agent(agent)
        
for i in range(20):
    label = classes[rd.randint(0, 13)]  # Select a random class label
    # Get input and output for current image
    img_path = select_random_image("db/train/" + label + "/")  # Correct path formatting
    print(label)
    print(img_path)
    img = Image.open(img_path)
    inp = np.array(img).reshape(1, -1)
    
    while inp.shape[1] != 196608:
        label = classes[rd.randint(0, 13)]  # Select a random class label
        # Get input and output for current image
        img_path = select_random_image("db/train/" + label + "/")  # Correct path formatting
        # print(label)
        # print(img_path)
        img = Image.open(img_path)
        inp = np.array(img).reshape(1, -1)
    
    # One-hot encode the label
    out = np.zeros((1, 14))
    out[0, classes.index(label)] = 1  # Use index of the class label
    
    # Forward propagation
    A2 = agent.propagate(inp)
    prediction = np.argmax(A2)