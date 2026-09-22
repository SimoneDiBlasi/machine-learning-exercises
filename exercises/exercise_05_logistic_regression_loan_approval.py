"""
Exercise 05 - Logistic Regression: Loan Approval Prediction

Objective:
    Use Logistic Regression to predict whether a loan application
    will be approved based on multiple features.

Features:
    - Age
    - Annual income
    - Credit score

Target:
    - 0 = Loan rejected
    - 1 = Loan approved

Tasks:
    1. Create the dataset
    2. Split the dataset into training and test sets
    3. Create a Logistic Regression model
    4. Train the model
    5. Inspect the model coefficients and intercept
    6. Predict whether a new loan application will be approved
    7. Calculate the approval probabilities
    8. Predict the classes for the test set
    9. Calculate the model accuracy
    10. Compare the real and predicted results

Concepts:
    - Classification
    - Train/Test Split
    - Logistic Regression
    - Model Training
    - Prediction
    - Prediction Probability
    - Accuracy
    - Coefficients
    - Intercept
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Dataset
X = np.array([
    [22, 20000, 550],
    [25, 25000, 580],
    [28, 30000, 600],
    [30, 35000, 620],
    [32, 40000, 650],
    [35, 45000, 680],
    [38, 50000, 700],
    [40, 55000, 720],
    [45, 60000, 740],
    [50, 70000, 760],
    [55, 80000, 780],
    [60, 90000, 800]
])

y = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1
])


# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training set:")
print(X_train)

print("\nTest set:")
print(X_test)


# Model
model = LogisticRegression()


# Training
model.fit(X_train, y_train)


# Model Parameters
coefficients = model.coef_
intercept = model.intercept_

print("\nCoefficients:", coefficients)
print("Intercept:", intercept)


# Prediction for a New Applicant
new_applicant = np.array([[28, 20000, 620]])

loan_prediction = model.predict(new_applicant)
loan_probability = model.predict_proba(new_applicant)

print("\nNew applicant prediction:")
print("Loan approved:", loan_prediction[0])

print("\nLoan probabilities:")
print("Rejected:", loan_probability[0][0])
print("Approved:", loan_probability[0][1])


# Prediction on the Test Set
y_pred = model.predict(X_test)


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# Compare Real and Predicted Results
print("\nReal vs Predicted:")

for real, predicted in zip(y_test, y_pred):
    print(f"Real: {real} | Predicted: {predicted}")


# Visualization
plt.scatter(
    range(len(y_test)),
    y_test,
    label="Real"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Loan (0 = Rejected, 1 = Approved)")
plt.title("Logistic Regression - Real vs Predicted")
plt.legend()
plt.show()