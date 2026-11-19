import numpy as np
import torch
from torch import nn

# Parameters
seed = 0
n_samples = 30
noise_std = 1.0
learning_rate = 0.08
epochs = 100
w_init = 0.0
b_init = 0.0

# Same synthetic data as the NumPy example: y = 2x + 1 + noise
rng = np.random.default_rng(seed)
x_np = rng.uniform(0, 4, size=n_samples)
y_np = 2.0 * x_np + 1.0 + rng.normal(0, noise_std, size=n_samples)
X = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # @step data
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # @step data

torch.manual_seed(seed)
model = nn.Linear(in_features=1, out_features=1)
with torch.no_grad():
    model.weight.fill_(w_init)  # @step init
    model.bias.fill_(b_init)  # @step init
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()  # @step zero_grad
    y_hat = model(X)  # @step forward
    loss = loss_fn(y_hat, y)  # @step loss
    loss.backward()  # @step backward
    optimizer.step()  # @step update

# In the loop, loss is computed before each update; evaluate the final parameters
with torch.no_grad():
    y_hat = model(X)  # @step evaluate
    loss = loss_fn(y_hat, y)  # @step evaluate

w, b = model.weight.item(), model.bias.item()
print(f"after {epochs} epochs: w = {w:.4f}, b = {b:.4f}, loss = {loss.item():.4f}")
