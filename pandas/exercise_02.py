
'''
Exercise 02 - Data Filtering & Sorting

1. Select only the Name, Age, and Sex columns.
2. Retrieve all passengers older than 30.
3. Retrieve all female passengers.
4. Retrieve all passengers who survived.
5. Retrieve all passengers in first class (Pclass == 1).
6. Retrieve all passengers older than 30 who survived.
7. Retrieve all female passengers younger than 18.
8. Retrieve passengers who paid a fare greater than 50.
9. Sort passengers by age in descending order.
10. Display the 10 oldest passengers.
'''

import pandas as pd
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define dataset path
DATASET_PATH = PROJECT_ROOT / "datasets" / "titanic" / "train.csv"

# Load dataset
dataset = pd.read_csv(DATASET_PATH)


# 1. Select only the Name, Age, and Sex columns.
columns = ["Name", "Age", "Sex"]
print(dataset[columns])

# Retrieve all passengers older than 30.
print(dataset[dataset["Age"] > 30])

# Retrieve all female passengers
print(dataset[dataset["Sex"] =='female'])


# 4. Retrieve all passengers who survived.
print(dataset[dataset["Survived"] == 1])

# 5. Retrieve all passengers in first class (Pclass == 1).
print(dataset[dataset["Pclass"] == 1])

# 6. Retrieve all passengers older than 30 who survived.
print(dataset[(dataset["Age"] > 30) & (dataset["Survived"] == 1)])

# 7. Retrieve all female passengers younger than 18.
print(dataset[(dataset["Age"] < 18) & (dataset["Sex"] =='female')])

# 8. Retrieve passengers who paid a fare greater than 50.
print(dataset[dataset["Fare"] > 50])

# 9. Sort passengers by age in descending order.
print(dataset.sort_values(by="Age", ascending=False))

# 10. Display the 10 oldest passengers.
print(dataset.sort_values(by="Age", ascending=True).tail(10))