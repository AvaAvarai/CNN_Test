import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt

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

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def cross_entropy_loss(predictions, label):
    return -np.log(predictions[label])

def forward_pass(image, weights, biases, kernel):
    conv_output = relu(convolve(image, kernel))
    pooled_output = max_pooling(conv_output)
    flattened = pooled_output.flatten()
    fc_output = fully_connected(flattened, weights, biases)
    predictions = softmax(fc_output)
    return predictions, flattened

# Initialize parameters
kernel = np.random.randn(3, 3) * 0.1
weights = np.random.randn(13 * 13, 10) * 0.1
biases = np.zeros(10)

# Training loop with adaptive learning rate
initial_learning_rate = 0.01
decay_rate = 0.1

num_epochs = 10
batch_size = 64

loss_sums = []
accuracies = []

for epoch in range(num_epochs):
    loss_sum = 0
    correct = 0
    learning_rate = initial_learning_rate / (1 + decay_rate * epoch)

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
            weights -= learning_rate * np.outer(flattened, grad_output)
            biases -= learning_rate * grad_output

    loss_sums.append(loss_sum / len(X_train))
    accuracies.append(correct / len(X_train))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_sums[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}, Learning Rate: {learning_rate:.6f}")

# Evaluate
y_pred = []
for i in range(len(X_test)):
    predictions, _ = forward_pass(X_test[i], weights, biases, kernel)
    y_pred.append(np.argmax(predictions))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot Loss and Accuracy
plt.plot(range(num_epochs), loss_sums, label='Loss')
plt.plot(range(num_epochs), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Training Progress')
plt.legend()
plt.show()
