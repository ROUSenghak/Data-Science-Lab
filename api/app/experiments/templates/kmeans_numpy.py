import numpy as np

# Parameters
seed = 0
n_samples = 60
n_blobs = 3
cluster_std = 1.0
k = 3
init_method = 0  # 0: random data points, 1: k-means++, 2: uniform in bounding box
init_seed = 4
max_iter = 20

# Synthetic 2-D data: Gaussian blobs around random centres
rng = np.random.default_rng(seed)
blob_centers = rng.uniform(-5, 5, size=(n_blobs, 2))
blob = rng.integers(n_blobs, size=n_samples)
X = blob_centers[blob] + rng.normal(0, cluster_std, size=(n_samples, 2))  # @step data

# Initial centroids mu_1 ... mu_k
rng_init = np.random.default_rng(init_seed)
if init_method == 0:  # k distinct data points, chosen uniformly at random
    centroids = X[rng_init.choice(n_samples, size=k, replace=False)]
elif init_method == 1:  # k-means++: next centroid drawn with probability ~ D(x)^2
    centroids = X[[rng_init.integers(n_samples)]]
    for _ in range(k - 1):
        D2 = np.min(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
        centroids = np.vstack([centroids, X[rng_init.choice(n_samples, p=D2 / D2.sum())]])
else:  # uniform random positions inside the bounding box of the data
    centroids = rng_init.uniform(X.min(axis=0), X.max(axis=0), size=(k, 2))
centroids_init = centroids.copy()  # @step init
labels = np.full(n_samples, -1)  # no point is assigned yet

for iteration in range(max_iter):
    # Assignment step: squared distance from every point to every centroid, shape (n, k)
    sq_dist = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # @step distances
    prev_labels = labels  # @step assign
    labels = np.argmin(sq_dist, axis=1)  # @step assign
    n_changed = int(np.sum(labels != prev_labels))  # @step assign
    J = np.sum(sq_dist[np.arange(n_samples), labels])  # @step assign

    # Update step: each centroid moves to the mean of the points assigned to it
    R = labels[:, None] == np.arange(k)  # @step update
    counts = R.sum(axis=0)  # @step update
    sums = R.T @ X  # @step update
    new_centroids = centroids.copy()  # empty cluster (count 0): keep its centroid  # @step update
    new_centroids[counts > 0] = sums[counts > 0] / counts[counts > 0, None]  # @step update
    shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1))  # @step update
    centroids = new_centroids  # @step update
    J = np.sum((X - centroids[labels]) ** 2)  # @step objective

    if n_changed == 0:  # no point changed cluster, so the centroids did not move
        break

# Labels and inertia of the returned centroids (differs from the loop only if max_iter was reached)
labels = np.argmin(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)  # @step evaluate
inertia = np.sum((X - centroids[labels]) ** 2)  # @step evaluate

print(f"stopped after {iteration + 1} iterations, inertia J = {inertia:.4f}")
print("cluster sizes:", np.bincount(labels, minlength=k))
