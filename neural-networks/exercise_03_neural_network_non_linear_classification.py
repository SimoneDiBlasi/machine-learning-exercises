"""
# Exercise 03 - Neural Network: Non-Linear Classification

## Objective

Build a neural network capable of solving a binary classification problem where the two classes cannot be separated effectively using a simple linear decision boundary.

The main goal of this exercise is to understand why neural networks use hidden layers and non-linear activation functions such as ReLU.

## Problem

We want to classify a set of two-dimensional points into two classes:

* Class 0
* Class 1

Each point has two features:

* Feature 1
* Feature 2

Unlike the previous classification exercise, the two classes will have a non-linear distribution.

This means that a simple linear model such as Logistic Regression will not be able to separate the two classes perfectly.

A neural network with hidden layers and non-linear activation functions should be able to learn a non-linear decision boundary.

## Dataset

Use the `make_moons` dataset provided by Scikit-Learn.

Create a dataset with:

* 500 samples
* 2 features
* 2 classes
* some noise

Example:

```python
X, y = make_moons(
    n_samples=500,
    noise=0.20,
    random_state=42
)
```

## Tasks

### 1. Create the dataset

Generate the `make_moons` dataset and inspect:

* `X`
* `y`
* the number of samples
* the number of features

### 2. Visualize the dataset

Create a scatter plot showing the two classes.

The goal is to visually understand the shape of the dataset and why a linear decision boundary would have difficulty separating the classes.

### 3. Split the dataset

Split the dataset into:

* 80% training data
* 20% test data

Use:

```python
train_test_split(...)
```

Use `stratify=y` so that the proportion of the two classes remains similar in the training and test sets.

### 4. Standardize the features

Use `StandardScaler`.

Fit the scaler only on the training data and use it to transform both training and test data.

### 5. Build the neural network

Create the following architecture:

```text
Input (2 features)
        ↓
Dense(16, ReLU)
        ↓
Dense(8, ReLU)
        ↓
Dense(1, Sigmoid)
```

### 6. Compile the model

Use:

* Optimizer: Adam
* Loss: Binary Crossentropy
* Metric: Accuracy

### 7. Train the model

Train the network for approximately 100 epochs.

Use part of the training data as validation data.

Monitor:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

### 8. Evaluate the model

Evaluate the model using the test dataset.

Display:

* Test Loss
* Test Accuracy

### 9. Generate predictions

Use the trained model to calculate the probability that each test sample belongs to class 1.

Convert probabilities into classes using a threshold of `0.5`.

For example:

```text
Probability >= 0.5 → Class 1
Probability < 0.5  → Class 0
```

### 10. Calculate classification metrics

Calculate:

* Accuracy
* Confusion Matrix

Analyze the results.

### 11. Visualize training

Create two plots:

1. Training Accuracy vs Validation Accuracy
2. Training Loss vs Validation Loss

Use these plots to understand how the model learns during training.

### 12. Compare linear and non-linear networks

Create another network without non-linear activation functions in the hidden layers.

Compare:

```text
Network A

Input
  ↓
Dense
  ↓
Dense
  ↓
Output
```

with:

```text
Network B

Input
  ↓
Dense + ReLU
  ↓
Dense + ReLU
  ↓
Output
```

Observe the difference in their ability to classify the dataset.

### 13. Visualize the decision boundary

Create a visualization showing:

* The original points
* Class 0
* Class 1
* The decision boundary learned by the neural network

The objective is to visually see that the neural network can learn a non-linear boundary.

## Learning Goals

After completing this exercise, you should understand:

* What a non-linear classification problem is
* Why Logistic Regression has limitations
* Why neural networks use hidden layers
* Why activation functions such as ReLU are important
* The difference between linear and non-linear transformations
* What a decision boundary represents
* How a neural network learns a non-linear decision boundary
* How to evaluate a binary classification model
* How to interpret training and validation curves
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Create the non-linear classification dataset
X, y = make_moons(
    n_samples=500,
    noise=0.20,
    random_state=42
)


# Visualize the dataset
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Non-Linear Classification Dataset")
plt.legend()
plt.show()


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()

# Train the neural network
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.20,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate prediction probabilities
y_probability = model.predict(X_test)

# Convert probabilities into binary classes
y_pred = (y_probability >= 0.5).astype(int)

print("First 10 prediction probabilities:")
print(y_probability[:10])

print("\nFirst 10 predicted classes:")
print(y_pred[:10])

print("\nFirst 10 real classes:")
print(y_test[:10])

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks([0, 1], ["Class 0", "Class 1"])
plt.yticks([0, 1], ["Class 0", "Class 1"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.show()


# Plot training and validation accuracy
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Create a grid of points
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Convert the grid into a list of points
grid = np.c_[xx.ravel(), yy.ravel()]

# Apply the same scaling used for the training data
grid_scaled = scaler.transform(grid)

# Predict the probability for each point
grid_probability = model.predict(
    grid_scaled,
    verbose=0
)

# Convert probabilities back to the grid shape
grid_probability = grid_probability.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    grid_probability,
    levels=[0, 0.5, 1],
    alpha=0.3
)

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network Decision Boundary")
plt.legend()

plt.show()