import numpy as np

# Parameters
seed = 0
n_samples = 30
noise_std = 1.0
learning_rate = 0.08
epochs = 100
w_init = 0.0
b_init = 0.0

# Synthetic data from y = 2x + 1 + noise
rng = np.random.default_rng(seed)
x = rng.uniform(0, 4, size=n_samples)
y = 2.0 * x + 1.0 + rng.normal(0, noise_std, size=n_samples)  # @step data

# Model parameters: y_hat = w * x + b
w, b = w_init, b_init  # @step init

for epoch in range(epochs):
    y_hat = w * x + b  # @step forward
    residual = y_hat - y  # @step loss
    loss = np.mean(residual ** 2)  # @step loss
    dw = 2 * np.mean(residual * x)  # @step gradient
    db = 2 * np.mean(residual)  # @step gradient
    w = w - learning_rate * dw  # @step update
    b = b - learning_rate * db  # @step update

print(f"after {epochs} epochs: w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}")
