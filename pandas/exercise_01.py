'''
Load the Titanic CSV dataset using Pandas.
Display the first 10 rows.
Display the last 5 rows.
Print the number of rows and columns.
Display column names.
Display dataset information using info().
Generate descriptive statistics using describe().
Count missing values for each column.
Count the number of passengers by gender.
Calculate the average passenger age.
'''

import pandas as pd 
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define dataset path
DATASET_PATH = PROJECT_ROOT / "datasets" / "titanic" / "train.csv"

# Load dataset
dataset = pd.read_csv(DATASET_PATH)
# Display the first 10 rows.
first_ten_rows = dataset.head(10)
print(first_ten_rows)

# Display the last 5 rows.
last_five_rows = dataset.tail()
print(last_five_rows)

#Print the number of rows and columns 
print(dataset.shape)

#Display column names.
print(dataset.columns.values.tolist())

#Display dataset information using info()
print(dataset.info())

#Generate descriptive statistics using describe()
print(dataset.describe())

#Count missing values for each column
print(dataset.isnull().sum())

#Count the number of passengers by gender.
print(dataset.groupby(by=["Sex"]).count()["PassengerId"])

#Calculate the average passenger age.
print(dataset["Age"].mean())