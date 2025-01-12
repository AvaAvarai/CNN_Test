import numpy as np


# Adam optimizer
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # First moment dict
        self.v = {}  # Second moment dict

    def update(self, params, grads):
        """
        params: dictionary of parameters to update
        grads: dictionary of gradients corresponding to parameters
        """
        if not self.m:  # Initialize momentum and velocity dictionaries
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1
        
        for key in params:
            # Update biased first and second moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Compute bias-corrected moments
            m_corr = self.m[key] / (1 - self.beta1 ** self.t)
            v_corr = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[key] -= self.lr * m_corr / (np.sqrt(v_corr) + self.epsilon)

        return params
