import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load MNIST dataset from scikit-learn
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(np.int32)

# Reshape and normalize data
X = X.to_numpy().reshape(-1, 28, 28) / 255.0  # Normalize to range [0, 1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Convolutional layer
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

# Max-pooling layer
def max_pooling(feature_map, size=2, stride=2):
    height, width = feature_map.shape
    output_height = (height - size) // stride + 1
    output_width = (width - size) // stride + 1

    output = np.zeros((output_height, output_width))

    for i in range(0, height - size + 1, stride):
        for j in range(0, width - size + 1, stride):
            region = feature_map[i:i + size, j:j + size]
            output[i // stride, j // stride] = np.max(region)

    return output

# Fully connected layer
def fully_connected(flattened_input, weights, bias):
    return np.dot(flattened_input, weights) + bias

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability trick
    return exp_x / exp_x.sum(axis=0)

# Cross-entropy loss
def cross_entropy_loss(predictions, label):
    return -np.log(predictions[label])

# Forward pass
def forward_pass(image, weights, biases, kernel):
    # Convolution
    conv_output = convolve(image, kernel)
    conv_output = relu(conv_output)

    # Max pooling
    pooled_output = max_pooling(conv_output)

    # Flatten
    flattened = pooled_output.flatten()

    # Fully connected layer
    fc_output = fully_connected(flattened, weights, biases)
    predictions = softmax(fc_output)

    return predictions, flattened

# Initialize parameters
kernel = np.random.randn(3, 3) * 0.1
weights = np.random.randn(13 * 13, 10) * 0.1
biases = np.zeros(10)

# Training loop
learning_rate = 0.01
num_epochs = 3

for epoch in range(num_epochs):
    loss_sum = 0
    correct = 0

    for i in range(len(X_train)):
        image = X_train[i]
        label = y_train[i]

        # Forward pass
        predictions, flattened = forward_pass(image, weights, biases, kernel)

        # Loss
        loss = cross_entropy_loss(predictions, label)
        loss_sum += loss

        # Accuracy
        if np.argmax(predictions) == label:
            correct += 1

        # Backpropagation
        grad_output = predictions
        grad_output[label] -= 1

        grad_weights = np.outer(flattened, grad_output)
        grad_biases = grad_output

        weights -= learning_rate * grad_weights
        biases -= learning_rate * grad_biases

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_sum / len(X_train):.4f}, Accuracy: {correct / len(X_train):.4f}")

# Evaluate on test set
correct = 0
for i in range(len(X_test)):
    image = X_test[i]
    label = y_test[i]

    predictions, _ = forward_pass(image, weights, biases, kernel)

    if np.argmax(predictions) == label:
        correct += 1

print(f"Test Accuracy: {correct / len(X_test):.4f}")
