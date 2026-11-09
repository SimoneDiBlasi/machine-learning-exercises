
"""
EXERCISE 03 - Netflix Movies & TV Shows Analysis

Dataset: netflix_titles.csv

Objective:
Analyze the Netflix catalog using Pandas.
Practice data exploration, filtering, grouping,
aggregation, and data cleaning.

--------------------------------------------------
EXERCISE 3.1 - BASIC DATA ANALYSIS
--------------------------------------------------

1. Load the dataset using Pandas.

2. Display the first 10 rows.

3. Display the dataset dimensions (rows and columns).

4. Display all column names.

5. Display information about the dataset,
   including data types and non-null values.

6. Count the total number of Movies and TV Shows.

7. Count how many unique directors are present.

8. Find the 10 most frequent countries.

9. Count the number of titles released in each year.

10. Find the year with the highest number of releases.


--------------------------------------------------
EXERCISE 3.2 - DATA FILTERING
--------------------------------------------------

11. Retrieve all movies released after 2015.

12. Retrieve all TV Shows with more than 5 seasons.

13. Retrieve all titles belonging to the genre
    "Documentaries".

14. Find all titles added to Netflix in 2020.

15. Retrieve all titles whose descriptions contain
    the word "crime" (case-insensitive).


--------------------------------------------------
EXERCISE 3.3 - GROUPING AND AGGREGATION
--------------------------------------------------

16. Count the number of Movies and TV Shows
    for each country.

17. Find the 10 most popular genres.

18. Calculate the average number of seasons
    for TV Shows.

19. Find the 10 countries with the highest
    number of Movies.

20. Group titles by release year and type.
    Compare Movies and TV Shows for each year.


--------------------------------------------------
EXERCISE 3.4 - DATA CLEANING
--------------------------------------------------

21. Identify all columns containing missing values.

22. Calculate the percentage of missing values
    for each column.

23. Replace missing values in the following columns
    with "Unknown":
    - director
    - cast
    - country

24. Convert the date_added column to datetime.

25. Extract the month and year from date_added
    into two new columns:
    - added_month
    - added_year


--------------------------------------------------
FINAL CHALLENGE - DATA ANALYSIS
--------------------------------------------------

Answer the following questions using Pandas:

1. Does Netflix have more Movies or TV Shows?

2. Which country has the largest number of titles?

3. Which year had the highest number of releases?

4. Which genres are most represented?

5. What is the average number of seasons
   for TV Shows?

6. Which year had the highest number of titles
   added to Netflix?

--------------------------------------------------
IMPORTANT:
- Use Pandas operations whenever possible.
- Avoid manual calculations.
- Do not use loops unless necessary.
- Keep your code clean and readable.
- Try to solve each exercise independently.
"""


import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "netflix_titles.csv"

dataset = pd.read_csv(DATASET_PATH)


# 1. Display the first 10 rows of the dataset.
print("\n========== FIRST 10 ROWS ==========")
print(dataset.head(10).to_string(index=False))


# 2. Display the dataset dimensions (rows and columns).
print("\n========== DATASET DIMENSIONS ==========")
rows, columns = dataset.shape

print(f"Rows:    {rows}")
print(f"Columns: {columns}")


# 3. Display all column names.
print("\n========== DATASET COLUMNS ==========")

for index, column in enumerate(dataset.columns, start=1):
    print(f"{index}. {column}")

# 4. Display information about the dataset, including data types and non-null values.
print("\n========== DATASET INFO ==========")
print(dataset.info())

# 5. Count the total number of Movies and TV Shows.
print("\n========== NUMBER OF MOVIE AND TV SHOW INFO ==========")
movie_count = (dataset["type"] == "Movie").sum()
tv_show_count = (dataset["type"] == "TV Show").sum()

print(f"Movies:   {movie_count:,}")
print(f"TV Shows: {tv_show_count:,}")

# 6. Count how many unique directors are present.
print("\n========== UNIQUE DIRECTORS INFO ==========")
unique_directors = dataset["director"].nunique()
print(f"Unique directors: {unique_directors:,}")


# 7. Find the 10 most frequent countries.
print("\n========== TOP 10 MOST FREQUENT COUNTRIES ==========")

most_frequent_countries = (
    dataset["country"]
    .value_counts()
    .head(10)
)

print(most_frequent_countries)

# 8. Count the number of titles released in each year.
print("\n========== TITLES RELEASED BY YEAR ==========")

number_of_titles_by_year = (
    dataset.groupby("release_year")
    .size()
    .sort_index(ascending=True)
)

print(number_of_titles_by_year)

# 9. Find the year with the highest number of releases.
print( dataset.groupby("release_year")
    .size()
    .sort_values(ascending=False))

# --------------------------------------------------
# EXERCISE 3.2 - DATA FILTERING
# --------------------------------------------------

# 10. Retrieve all movies released after 2015.
movies_after_2015 = dataset[
    (dataset["release_year"] > 2015) &
    (dataset["type"] == "Movie")
]

print(movies_after_2015)

# 11. Retrieve all TV Shows with more than 5 seasons.

tv_shows = dataset[dataset["type"] == "TV Show"]

number_of_seasons = (
    tv_shows["duration"]
    .str.extract(r"(\d+)")
    .astype(int)
    .squeeze()
)

tv_shows_more_than_5_seasons = tv_shows[
    number_of_seasons > 5
]

print(tv_shows_more_than_5_seasons)


# 12. Retrieve all titles belonging to the genre
#     "Documentaries".
documentaries = dataset[
    dataset["listed_in"].str.contains(
        "Documentaries",
        case=False,
        na=False
    )
]
print(documentaries)
# 13. Find all titles added to Netflix in 2020.
dataset["date_added"] = pd.to_datetime(
    dataset["date_added"].str.strip(),
    format="%B %d, %Y",
    errors="coerce"
)

movies_added = dataset[
    dataset["date_added"].dt.year == 2020
]

print(movies_added)

# 14. Retrieve all titles whose descriptions contain
#     the word "crime" (case-insensitive).
print(dataset[
    dataset["description"].str.contains(
        "crime",
        case=False,
        na=False
    )])

# --------------------------------------------------
# EXERCISE 3.3 - GROUPING AND AGGREGATION
# --------------------------------------------------

# 15. Count the number of Movies and TV Shows
#     for each country.
grouped_by_country = dataset.groupby(["country", "type"]).size()

print(grouped_by_country)

# 16. Find the 10 most popular genres.
print(
    dataset["listed_in"]
    .str.split(",")
    .explode()
    .value_counts()
    .head(10)
)

# 17. Calculate the average number of seasons
#     for TV Shows.
tv_shows = dataset[dataset["type"] == "TV Show"]

number_of_seasons = (
    tv_shows["duration"]
    .str.extract(r"(\d+)")
    .astype(int)
    .squeeze()
)

average_number_of_seasons = number_of_seasons.mean()

print(f"Average number of seasons: {average_number_of_seasons:.2f}")

# 18. Find the 10 countries with the highest
#     number of Movies.
print(
    dataset[dataset["type"] == "Movie"]
    .groupby("country")
    .size()
    .sort_values(ascending=False)
    .head(10)
)


# 19. Group titles by release year and type.
#     Compare Movies and TV Shows for each year.
print(dataset.groupby(["release_year", "type"]).size().unstack())

# --------------------------------------------------
# EXERCISE 3.4 - DATA CLEANING
# --------------------------------------------------

# 20. Identify all columns containing missing values.
missing_columns = dataset.isnull().any()
print(missing_columns[missing_columns])

# 21. Calculate the percentage of missing values
#     for each column.
missing_values = dataset.isnull().sum()
len_dataset = len(dataset)
percentage = missing_values /  len_dataset * 100
print(percentage)
 
# 22. Replace missing values in the following columns
#     with "Unknown":
#     - director
#     - cast
#     - country

dataset.fillna({
    "director": "Unknown",
    "cast": "Unknown",
    "country": "Unknown"
}, inplace=True)

# 23. Convert the date_added column to datetime.
dataset["date_added"] = pd.to_datetime(
    dataset["date_added"].str.strip(),
    format="%B %d, %Y",
    errors="coerce"
)

# 24. Extract the month and year from date_added
#     into two new columns:
#     - added_month
#     - added_year
dataset["added_month"] = dataset["date_added"].dt.month
dataset["added_year"] = dataset["date_added"].dt.year

# --------------------------------------------------
# FINAL CHALLENGE - DATA ANALYSIS
# --------------------------------------------------

# 1. Does Netflix have more Movies or TV Shows?

# 2. Which country has the largest number of titles?

# 3. Which year had the highest number of releases?

# 4. Which genres are most represented?

# 5. What is the average number of seasons
#    for TV Shows?

# 6. Which year had the highest number of titles
#    added to Netflix?
