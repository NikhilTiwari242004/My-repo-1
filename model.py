import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Example dataset: [area, bedrooms, age] -> price
X = np.array([
    [1000, 2, 5],
    [1500, 3, 10],
    [1800, 3, 15],
    [2400, 4, 20],
    [3000, 5, 25]
], dtype=float)

y = np.array([300000, 400000, 500000, 600000, 700000], dtype=float)

# Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)
epsilon = 1e-8  # Small value to prevent division by zero
X_norm = (X - X_mean) / (X_std + epsilon)

# Add bias term
X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

# Gradient Descent
def gradient_descent(X, y, lr=0.01, epochs=5000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        gradients = (2/m) * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
    return theta

theta = gradient_descent(X_b, y)

# Save model parameters
with open("house_model.pkl", "wb") as f:
    pickle.dump((theta, X_mean, X_std), f)

# Create plots
os.makedirs("static/plots", exist_ok=True)

plt.figure(figsize=(6,4))
plt.scatter(X[:,0], y, color='blue')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("Area vs Price")
plt.savefig("static/plots/area_price.png")
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(X[:,1], y, color='green')
plt.xlabel("Bedrooms")
plt.ylabel("Price ($)")
plt.title("Bedrooms vs Price")
plt.savefig("static/plots/bedrooms_price.png")
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(X[:,2], y, color='red')
plt.xlabel("Age")
plt.ylabel("Price ($)")
plt.title("Age vs Price")
plt.savefig("static/plots/age_price.png")
plt.close()

print("Model trained and plots generated!")
