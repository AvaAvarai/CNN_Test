import numpy as np

class CNN:
    def __init__(self, input_shape=(28, 28), kernel_size=(3, 3), num_classes=10):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        
        # Initialize parameters with He initialization
        self.kernel = np.random.randn(*kernel_size) * np.sqrt(2.0 / (input_shape[0] * input_shape[1]))
        
        # Calculate shape after convolution and pooling
        conv_shape = (input_shape[0] - kernel_size[0] + 1, input_shape[1] - kernel_size[1] + 1)
        pool_shape = (conv_shape[0] // 2, conv_shape[1] // 2)  # Assuming pool size = stride = 2
        self.flattened_size = pool_shape[0] * pool_shape[1]
        
        self.weights = np.random.randn(self.flattened_size, num_classes) * np.sqrt(2.0 / self.flattened_size)
        self.biases = np.zeros(num_classes)

    def convolve(self, image):
        output = np.zeros((image.shape[0] - self.kernel_size[0] + 1, 
                         image.shape[1] - self.kernel_size[1] + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i + self.kernel_size[0], 
                                          j:j + self.kernel_size[1]] * self.kernel)
        return output

    def max_pooling(self, feature_map, size=2, stride=2):
        output = np.zeros((feature_map.shape[0] // size, feature_map.shape[1] // size))
        self.pool_masks = np.zeros_like(feature_map)  # Store for backprop
        
        for i in range(0, feature_map.shape[0], stride):
            for j in range(0, feature_map.shape[1], stride):
                patch = feature_map[i:i + size, j:j + size]
                max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                output[i // stride, j // stride] = patch[max_idx]
                # Save mask of where maximum was found
                self.pool_masks[i + max_idx[0], j + max_idx[1]] = 1
        return output

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, image):
        # Save intermediate values for backprop
        self.image = image
        self.conv_output = self.convolve(image)
        self.relu_output = self.leaky_relu(self.conv_output)
        self.pooled_output = self.max_pooling(self.relu_output)
        self.flattened = self.pooled_output.flatten()
        self.fc_output = np.dot(self.flattened, self.weights) + self.biases
        self.predictions = self.softmax(self.fc_output)
        
        return self.predictions

    def backward(self, label):
        # Initialize gradients
        grad_output = self.predictions.copy()
        grad_output[label] -= 1  # Derivative of cross-entropy loss
        
        # Gradients for fully connected layer
        grad_weights = np.outer(self.flattened, grad_output)
        grad_biases = grad_output
        grad_flattened = np.dot(grad_output, self.weights.T)
        
        # Reshape gradient back to pooled shape
        grad_pooled = grad_flattened.reshape(self.pooled_output.shape)
        
        # Gradient through max pooling
        grad_relu = np.zeros_like(self.relu_output)
        pool_size = 2
        for i in range(0, grad_relu.shape[0], pool_size):
            for j in range(0, grad_relu.shape[1], pool_size):
                grad_relu[i:i + pool_size, j:j + pool_size] = \
                    grad_pooled[i // pool_size, j // pool_size] * \
                    self.pool_masks[i:i + pool_size, j:j + pool_size]
        
        # Gradient through ReLU
        grad_conv = grad_relu * self.leaky_relu_derivative(self.conv_output)
        
        # Gradient for kernel (convolution)
        grad_kernel = np.zeros_like(self.kernel)
        for i in range(grad_conv.shape[0]):
            for j in range(grad_conv.shape[1]):
                grad_kernel += self.image[i:i + self.kernel_size[0], 
                                       j:j + self.kernel_size[1]] * grad_conv[i, j]
        
        return grad_kernel, grad_weights, grad_biases
