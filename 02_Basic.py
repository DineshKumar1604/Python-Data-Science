# ===================================================================
# 02_numpy_pandas_basics.py
# Description: Demonstrates numerical computing with NumPy and
# data manipulation with Pandas.
# ===================================================================

import numpy as np
import pandas as pd

# ===================================================================
# SECTION 2: NUMPY FOR NUMERICAL COMPUTING
# ===================================================================
print("\n--- NumPy Array Operations ---")

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"1D Array:\n{arr1}")
print(f"2D Array:\n{arr2}")

# Array generation
range_array = np.arange(0, 10, 2) # Start, stop, step
print(f"\nArray with arange: {range_array}")
linspace_array = np.linspace(0, 1, 5) # Start, stop, num of points
print(f"Array with linspace: {linspace_array}")
identity_matrix = np.eye(3)
print(f"3x3 Identity Matrix:\n{identity_matrix}")

# Mathematical operations
add_result = arr1 + arr1
print(f"\nArray addition: {add_result}")
dot_product = np.dot(arr2, arr2.T) # .T is transpose
print(f"Dot product of 2D array with its transpose:\n{dot_product}")
print(f"Square root of arr1: {np.sqrt(arr1)}")
print(f"Sum of arr1: {np.sum(arr1)}")
print(f"Mean of arr1: {np.mean(arr1)}")


# ===================================================================
# SECTION 3: PANDAS FOR DATA MANIPULATION
# ===================================================================
print("\n\n--- Pandas DataFrame and Series ---")

# Creating a Series
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(f"Pandas Series:\n{s}")

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
print(f"\nOriginal DataFrame:\n{df}")

# --- File I/O with Pandas ---
# NOTE: These operations create and read from local files.
# Exporting data
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
df.to_json('output.json', orient='records')

# Importing data
df_from_csv = pd.read_csv('output.csv')
print(f"\nDataFrame from CSV:\n{df_from_csv}")


# --- Handling Missing Data ---
print("\n--- Handling Missing Data ---")
data_missing = {'Name': ['Alice', 'Bob', None, 'David'],
                'Age': [24, None, 35, 40],
                'City': ['New York', 'Los Angeles', 'Chicago', None]}
df_missing = pd.DataFrame(data_missing)
print(f"DataFrame with missing values:\n{df_missing}")

# Drop rows with any missing values
df_dropped = df_missing.dropna()
print(f"\nDataFrame after dropping rows with nulls:\n{df_dropped}")

# Fill missing values
df_filled = df_missing.fillna({'Name': 'Unknown', 'City': 'Unknown'})
df_filled['Age'] = df_filled['Age'].fillna(df_filled['Age'].mean())
print(f"\nDataFrame after filling nulls:\n{df_filled}")


# --- Data Selection and Filtering ---
print("\n--- Data Selection and Filtering ---")
# Boolean indexing
adults = df[df['Age'] >= 30]
print(f"Adults (Age >= 30):\n{adults}")

# Selecting columns
name_and_city = df[['Name', 'City']]
print(f"\nSelected columns (Name, City):\n{name_and_city}")

# Using .iloc for position-based selection (rows 1-2, columns 0 and 2)
selected_iloc = df.iloc[1:3, [0, 2]]
print(f"\nSelection using iloc:\n{selected_iloc}")

# Applying functions
df['Age_Plus_Ten'] = df['Age'].apply(lambda x: x + 10)
print(f"\nDataFrame with 'Age_Plus_Ten' column:\n{df}")

print("\n--- END OF NUMPY/PANDAS SCRIPT ---")
