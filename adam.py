import numpy as np


# Adam optimizer
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = 0
        self.v_weights = 0
        self.m_biases = 0
        self.v_biases = 0
        self.t = 0

    def update(self, weights, biases, grad_weights, grad_biases):
        self.t += 1
        # Update biased first moments
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * grad_biases

        # Update biased second moments
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights ** 2)
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (grad_biases ** 2)

        # Compute bias-corrected moments
        m_weights_corr = self.m_weights / (1 - self.beta1 ** self.t)
        m_biases_corr = self.m_biases / (1 - self.beta1 ** self.t)
        v_weights_corr = self.v_weights / (1 - self.beta2 ** self.t)
        v_biases_corr = self.v_biases / (1 - self.beta2 ** self.t)

        # Update weights and biases
        weights -= self.lr * m_weights_corr / (np.sqrt(v_weights_corr) + self.epsilon)
        biases -= self.lr * m_biases_corr / (np.sqrt(v_biases_corr) + self.epsilon)

        return weights, biases
