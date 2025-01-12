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
from cnn import CNN

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(np.int32)

# Reshape and normalize data
X = X.to_numpy().reshape(-1, 28, 28) / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Initialize network and optimizer
cnn = CNN()
adam_optimizer = AdamOptimizer(learning_rate=0.001)

# Training parameters
batch_size = 64
num_epochs = 10

# Calculate initial accuracy on test set
y_pred_initial = []
for i in range(len(X_test)):
    predictions = cnn.forward(X_test[i])
    y_pred_initial.append(np.argmax(predictions) == y_test[i])
initial_accuracy = sum(y_pred_initial) / len(y_pred_initial)

# Create output directory and save initial visualizations
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

def save_visualizations(cnn, accuracy, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = os.path.join(output_dir, f"visualizations_{timestamp}.png")
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    sns.heatmap(cnn.kernel, cmap='coolwarm', center=0)
    plt.title(f'CNN Kernel\n(Accuracy: {accuracy:.4f})')

    plt.subplot(122)
    sns.heatmap(cnn.weights, cmap='coolwarm', center=0)
    plt.title(f'CNN Weights\n(Accuracy: {accuracy:.4f})')

    plt.tight_layout()
    plt.savefig(combined_filename)
    plt.close()
    print(f"Visualizations saved to {combined_filename}")

save_visualizations(cnn, initial_accuracy, output_dir)

# Training loop
loss_sums = []
accuracies = []

for epoch in range(num_epochs):
    loss_sum = 0
    correct = 0

    for batch_start in range(0, len(X_train), batch_size):
        batch_images = X_train[batch_start:batch_start + batch_size]
        batch_labels = y_train[batch_start:batch_start + batch_size]

        for image, label in zip(batch_images, batch_labels):
            # Forward pass
            predictions = cnn.forward(image)
            loss_sum += -np.log(predictions[label])  # Cross entropy loss
            correct += (np.argmax(predictions) == label)

            # Backward pass
            grad_kernel, grad_weights, grad_biases = cnn.backward(label)

            # Update parameters using Adam optimizer
            cnn.weights, cnn.biases = adam_optimizer.update(
                cnn.weights, cnn.biases, grad_weights, grad_biases)
            
            # Update kernel (you might want to add separate optimizer for kernel)
            cnn.kernel -= 0.001 * grad_kernel  # Simple SGD for kernel

    loss_sums.append(loss_sum / len(X_train))
    accuracies.append(correct / len(X_train))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_sums[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")

# Save final visualizations
save_visualizations(cnn, accuracies[-1], output_dir)

# Evaluate on test data
y_pred = []
for i in range(len(X_test)):
    predictions = cnn.forward(X_test[i])
    y_pred.append(np.argmax(predictions))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the loss and accuracy
plt.figure(figsize=(10, 5))
plt.plot(loss_sums, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Loss and Accuracy over Epochs')
plt.legend()
plt.savefig(os.path.join(output_dir, f'loss_accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
plt.close()
