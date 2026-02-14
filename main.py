import numpy as np
import matplotlib.pyplot as plt

# Data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fix: Initialize as (1,1) to match matrix shapes
m = np.random.randn(1, 1)
b = np.random.randn(1, 1)

loss_history = []

# SGD
for _ in range(50):
    idx = np.random.permutation(len(X))
    Xs, ys = X[idx], y[idx]
    for i in range(len(X)):
        xi, yi = Xs[i:i+1], ys[i:i+1]
        pred = m * xi + b
        # Gradients
        dm = 2 * (pred - yi) * xi
        db = 2 * (pred - yi)
        # Update
        m -= 0.01 * dm
        b -= 0.01 * db
    loss_history.append(np.mean((m*X+b - y)**2))

print(f'Final Model: y={m[0,0]:.2f}x+{b[0,0]:.2f}')
plt.plot(loss_history)
plt.title('SGD Training Loss')
plt.show()