# Neural Network from Scratch for MNIST Digit Recognition

This project implements a feedforward neural network from scratch in Python for recognizing handwritten digits from the MNIST dataset. The implementation uses only NumPy and includes the fundamental components of neural networks: forward propagation, backpropagation, and gradient descent optimization.

## Table of Contents
- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
  - [Network Architecture](#network-architecture)
  - [Forward Propagation](#forward-propagation)
  - [Backpropagation](#backpropagation)
  - [Parameter Updates](#parameter-updates)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Example](#example)
- [Results](#results)

## Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. This project aims to classify these images using a neural network built from scratch, with a focus on understanding the underlying mathematical principles.

## Mathematical Foundation

### Network Architecture

The neural network consists of:
- An input layer with 784 neurons (28×28 pixels)
- One or more hidden layers with configurable sizes
- An output layer with 10 neurons (one for each digit, 0-9)

Each layer uses the sigmoid activation function: 

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Forward Propagation

For each layer $l$ in the network:

1. Calculate the weighted sum:
   $$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$

2. Apply the sigmoid activation function:
   $$A^{[l]} = \sigma(Z^{[l]}) = \frac{1}{1 + e^{-Z^{[l]}}}$$

Where:
- $W^{[l]}$ is the weight matrix for layer $l$
- $A^{[l-1]}$ is the activation from the previous layer
- $b^{[l]}$ is the bias vector for layer $l$
- $A^{[l]}$ is the activation output for layer $l$

### Backpropagation

Backpropagation computes the gradients of the loss function with respect to the weights and biases:

1. For the output layer ($L$):
   $$dZ^{[L]} = A^{[L]} - Y$$

2. For hidden layers:
   $$dZ^{[l]} = (W^{[l+1]})^T \cdot dZ^{[l+1]} \odot A^{[l]} \odot (1 - A^{[l]})$$

3. Compute gradients:
   $$dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot (A^{[l-1]})^T$$
   $$db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}_i$$

Where:
- $Y$ is the true label matrix (one-hot encoded)
- $m$ is the number of training examples
- $\odot$ represents element-wise multiplication

### Parameter Updates

The weights and biases are updated using gradient descent:

$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

Where $\alpha$ is the learning rate.

## Implementation Details

The implementation includes:

- `NeuralNet` class with methods for:
  - Initialization with configurable layer dimensions
  - Forward propagation
  - Backpropagation
  - Parameter updates
  - Training with mini-batch support
  - Prediction and evaluation

Key features:
- Weight initialization using He initialization for better convergence
- Mini-batch gradient descent for efficient training
- Binary cross-entropy loss function
- Sigmoid activation function for all layers

## Usage

```python
from NN import NeuralNet
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32') / 255.0  # Normalize to [0,1]
y = mnist.target.astype('int')

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_one_hot_train, y_one_hot_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Create and train the neural network
model = NeuralNet(dims=[784, 128, 64, 10])
losses = model.train(X_train, y_one_hot_train, nb_iters=100, learning_rate=0.01, batch_size=32, verbose=True)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
```

## Example

```python
# Example of classifying a single image
import matplotlib.pyplot as plt

# Select a test image
image_idx = 42
test_image = X_test[image_idx:image_idx+1]
true_label = y_test[image_idx]

# Make prediction
prediction = model.predict(test_image)
predicted_label = np.argmax(prediction)

# Display image and predictions
plt.figure(figsize=(4, 4))
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()
```

## Results

With proper hyperparameter tuning, this neural network implementation can achieve accuracy above 95% on the MNIST test set, which is impressive for a simple feedforward neural network with sigmoid activation functions.

For better performance, consider:
- Adding more hidden layers
- Increasing the number of neurons per layer
- Using different activation functions like ReLU
- Implementing more advanced optimization techniques
