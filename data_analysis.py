"""
Python Learning Program: Week 7 Assignment
Data Analysis and Visualization

This script loads the Iris dataset, performs exploratory data analysis, 
and creates various visualizations to understand the data patterns.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set the visual style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Task 1: Load and Explore the Dataset
print("Task 1: Load and Explore the Dataset")
print("-" * 50)

# Load the Iris dataset from sklearn
try:
    # Load the iris dataset
    print("Loading the Iris dataset...")
    iris = load_iris()
    
    # Create a pandas DataFrame for easier manipulation
    column_names = iris.feature_names
    df = pd.DataFrame(iris.data, columns=column_names)
    
    # Add the target column (species)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Display the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Dataset structure and information
print("\nDataset structure:")
print(df.info())

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("Cleaning the dataset by filling missing values with column means...")
    # Fill numeric columns with mean
    df = df.fillna(df.mean())
    print("Missing values have been filled.")
else:
    print("No missing values found. The dataset is clean!")

# Task 2: Basic Data Analysis
print("\n\nTask 2: Basic Data Analysis")
print("-" * 50)

# Compute basic statistics
print("\nBasic statistics of the dataset:")
print(df.describe())

# Group by species and compute mean
print("\nMean values grouped by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Interesting findings
print("\nInteresting findings:")
print("1. Iris-setosa has the shortest petal length and width compared to other species.")
print("2. Iris-virginica has the largest measurements in almost all dimensions.")
print("3. Sepal width shows less variation between species compared to other features.")

# Task 3: Data Visualization
print("\n\nTask 3: Data Visualization")
print("-" * 50)

# Create a directory to save plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Line chart: Showing average measurements by species
plt.figure(figsize=(12, 6))
for feature in column_names:
    plt.plot(species_means.index, species_means[feature], marker='o', label=feature)

plt.title('Average Measurements by Species')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/line_chart.png')
plt.show()

# 2. Bar chart: Comparing petal length across species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df, palette='viridis')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('plots/bar_chart.png')
plt.show()

# 3. Histogram: Distribution of sepal length
plt.figure(figsize=(10, 6))
for species in iris.target_names:
    subset = df[df['species'] == species]
    sns.histplot(subset['sepal length (cm)'], kde=True, label=species)

plt.title('Distribution of Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/histogram.png')
plt.show()

# 4. Scatter plot: Relationship between sepal length and petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', 
                hue='species', data=df, palette='viridis', s=80)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plots/scatter_plot.png')
plt.show()

# 5. Additional visualization: Pair plot for all features
print("\nCreating a pair plot to visualize relationships between all features...")
pair_plot = sns.pairplot(df, hue='species', height=2.5, palette='viridis')
pair_plot.fig.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.tight_layout()
plt.savefig('plots/pair_plot.png')
plt.show()

print("\nData analysis and visualization completed successfully!")
print("All plots have been saved in the 'plots' directory.")
