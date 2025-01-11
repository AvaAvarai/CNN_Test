# standard libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# our modules
from adam import AdamOptimizer

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(np.int32)

# Reshape and normalize data
X = X.to_numpy().reshape(-1, 28, 28) / 255.0  # Normalize to range [0, 1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Define CNN functions
def convolve(image, kernel):
    output = np.zeros((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    return output

def max_pooling(feature_map, size=2, stride=2):
    output = np.zeros((feature_map.shape[0] // size, feature_map.shape[1] // size))
    for i in range(0, feature_map.shape[0], stride):
        for j in range(0, feature_map.shape[1], stride):
            output[i // stride, j // stride] = np.max(feature_map[i:i + size, j:j + size])
    return output

def fully_connected(flattened_input, weights, bias):
    return np.dot(flattened_input, weights) + bias

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def cross_entropy_loss(predictions, label):
    return -np.log(predictions[label])

def forward_pass(image, weights, biases, kernel):
    conv_output = leaky_relu(convolve(image, kernel))
    pooled_output = max_pooling(conv_output)
    flattened = pooled_output.flatten()
    fc_output = fully_connected(flattened, weights, biases)
    predictions = softmax(fc_output)
    return predictions, flattened

# Initialize parameters
def initialize_network():
    # He initialization for kernel (input shape is 1 channel)
    kernel = np.random.randn(3, 3) * np.sqrt(2.0 / (28 * 28))
    # He initialization for weights (input shape is 13*13 from pooled feature map)
    weights = np.random.randn(13 * 13, 10) * np.sqrt(2.0 / (13 * 13))
    # biases for the fully connected layer
    biases = np.zeros(10)
    return kernel, weights, biases

# Save visualizations
def save_visualizations(kernel, weights, best_accuracy, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = os.path.join(output_dir, f"visualizations_{timestamp}.png")
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    sns.heatmap(kernel, cmap='coolwarm', center=0)
    plt.title(f'Selected Best Kernel\n(Accuracy: {best_accuracy:.4f})')

    plt.subplot(122)
    sns.heatmap(weights, cmap='coolwarm', center=0)
    plt.title(f'Selected Best Weights\n(Accuracy: {best_accuracy:.4f})')

    plt.tight_layout()
    plt.savefig(combined_filename)
    plt.close()
    print(f"Visualizations saved to {combined_filename}")

# Training loop
batch_size = 64
num_epochs = 10

kernel, weights, biases = initialize_network()

# Calculate initial accuracy on full test set
y_pred_initial = []
for i in range(len(X_test)):
    predictions, _ = forward_pass(X_test[i], weights, biases, kernel)
    y_pred_initial.append(np.argmax(predictions) == y_test[i])
initial_accuracy = sum(y_pred_initial) / len(y_pred_initial)

# Create output directory and save initial visualizations
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)
save_visualizations(kernel, weights, initial_accuracy, output_dir)

adam_optimizer = AdamOptimizer(learning_rate=0.001)

loss_sums = []
accuracies = []

for epoch in range(num_epochs):
    loss_sum = 0
    correct = 0

    for batch_start in range(0, len(X_train), batch_size):
        batch_images = X_train[batch_start:batch_start + batch_size]
        batch_labels = y_train[batch_start:batch_start + batch_size]

        for image, label in zip(batch_images, batch_labels):
            predictions, flattened = forward_pass(image, weights, biases, kernel)
            loss_sum += cross_entropy_loss(predictions, label)
            correct += (np.argmax(predictions) == label)

            # Gradients
            grad_output = predictions
            grad_output[label] -= 1
            grad_weights = np.outer(flattened, grad_output)
            grad_biases = grad_output

            # Update parameters using Adam optimizer
            weights, biases = adam_optimizer.update(weights, biases, grad_weights, grad_biases)

    loss_sums.append(loss_sum / len(X_train))
    accuracies.append(correct / len(X_train))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_sums[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")

# Save final visualizations
save_visualizations(kernel, weights, accuracies[-1], output_dir)

# Evaluate on test data
y_pred = []
for i in range(len(X_test)):
    predictions, _ = forward_pass(X_test[i], weights, biases, kernel)
    y_pred.append(np.argmax(predictions))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot the loss and accuracy saved in the visualizations folder
plt.figure(figsize=(10, 5))
plt.plot(loss_sums, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Loss and Accuracy over Epochs')
plt.legend()
plt.savefig(os.path.join(output_dir, f'loss_accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
plt.close()
