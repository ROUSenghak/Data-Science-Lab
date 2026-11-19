import numpy as np
from sklearn.cluster import KMeans

# Parameters
seed = 0
n_samples = 60
n_blobs = 3
cluster_std = 1.0
k = 3
init_method = 1  # 0: "random", 1: "k-means++"
n_init = 1
random_state = 0
max_iter = 20
tol = 0.0001

# Same synthetic data as the NumPy example
rng = np.random.default_rng(seed)
blob_centers = rng.uniform(-5, 5, size=(n_blobs, 2))
blob = rng.integers(n_blobs, size=n_samples)
X = blob_centers[blob] + rng.normal(0, cluster_std, size=(n_samples, 2))  # @step data

# The estimator: Lloyd's algorithm, restarted n_init times, best inertia kept
init = ["random", "k-means++"][init_method]
model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state, algorithm="lloyd")  # @step model

# One untraced library call; afterwards only the fitted attributes are visible
model.fit(X)  # @step fit
centroids = model.cluster_centers_  # @step fit
labels = model.labels_  # @step fit
inertia = model.inertia_  # @step fit
n_iter = model.n_iter_  # @step fit

# Public API: Euclidean distance from every point to every cluster centre, shape (n, k)
dist = model.transform(X)  # @step transform

# Independent check with our own NumPy code (not scikit-learn internals)
sq_dist = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # @step check
check_labels = np.argmin(sq_dist, axis=1)  # @step check
check_J = np.sum(sq_dist[np.arange(n_samples), labels])  # @step check

print(f"scikit-learn: n_iter_ = {n_iter}, inertia_ = {inertia:.6f}")
print(f"our check:    J = {check_J:.6f}, labels agree: {np.array_equal(check_labels, labels)}")
