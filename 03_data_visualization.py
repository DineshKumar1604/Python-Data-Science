# ===================================================================
# 03_data_visualization.py
# Description: Demonstrates how to create plots and charts using
# Matplotlib and Seaborn.
# ===================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("--- Data Visualization ---")
print("NOTE: This script will open plot windows. Close each window to continue.")

# --- Matplotlib for basic plotting ---
print("\nGenerating Matplotlib plots...")

# Matplotlib Line and Bar Plots in subplots
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.figure(figsize=(12, 5)) # Create a figure to hold the plots

# Subplot 1: Line Plot
plt.subplot(1, 2, 1) # (1 row, 2 columns, 1st subplot)
plt.plot(x, y, marker='o', linestyle='--', color='b', label='Prime Numbers')
plt.title('Matplotlib Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()

# Subplot 2: Bar Plot
categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]
plt.subplot(1, 2, 2) # (1 row, 2 columns, 2nd subplot)
plt.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title('Matplotlib Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')

plt.tight_layout() # Adjusts plot to prevent overlap
plt.show()


# --- Seaborn for statistical plotting ---
print("\nGenerating Seaborn plots...")

# Load a sample dataset from Seaborn
tips = sns.load_dataset('tips')
print("\nUsing the 'tips' dataset from Seaborn for the next plots.")

# Seaborn Box Plot and Histogram in subplots
plt.figure(figsize=(14, 6))

# Subplot 1: Box Plot
plt.subplot(1, 2, 1)
sns.boxplot(x='day', y='total_bill', data=tips, palette='pastel')
plt.title('Seaborn Box Plot of Total Bill by Day')

# Subplot 2: Histogram
plt.subplot(1, 2, 2)
sns.histplot(tips['total_bill'], kde=True, bins=20, color='purple')
plt.title('Seaborn Histogram of Total Bill')

plt.tight_layout()
plt.show()

# Seaborn Pair Plot (shows relationships between all variable pairs)
print("\nGenerating Seaborn Pair Plot... this may take a moment.")
sns.pairplot(tips, hue='sex', palette='husl')
plt.suptitle('Seaborn Pair Plot of the Tips Dataset', y=1.02) # Adjust title position
plt.show()

print("\n--- END OF VISUALIZATION SCRIPT ---")
