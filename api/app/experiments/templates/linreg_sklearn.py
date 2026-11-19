import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Parameters
seed = 0
n_samples = 30
noise_std = 1.0

# Same synthetic data as the NumPy example: y = 2x + 1 + noise
rng = np.random.default_rng(seed)
x = rng.uniform(0, 4, size=n_samples)
y = 2.0 * x + 1.0 + rng.normal(0, noise_std, size=n_samples)  # @step data
X = x.reshape(-1, 1)  # @step data

# scikit-learn solves least squares directly; .fit() exposes only its results
model = LinearRegression()
model.fit(X, y)  # @step fit
w, b = model.coef_[0], model.intercept_  # @step fit

y_hat = model.predict(X)  # @step predict
mse = mean_squared_error(y, y_hat)  # @step loss

# Independent check with our own NumPy code (not scikit-learn internals):
# solve the normal equations (A^T A) theta = A^T y with A = [x, 1]
A = np.column_stack([x, np.ones_like(x)])  # @step check
theta = np.linalg.solve(A.T @ A, A.T @ y)  # @step check

print(f"scikit-learn: w = {w:.6f}, b = {b:.6f}, MSE = {mse:.6f}")
print(f"normal eqs.:  w = {theta[0]:.6f}, b = {theta[1]:.6f}")
