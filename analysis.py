import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# Grouping by species and calculating the mean
print("\nMean values by species:")
print(df.groupby('species').mean())



# Line chart: Sepal Length over index
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sepal_length'], label='Sepal Length', color='green')
plt.title('Sepal Length Over Samples')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal_length')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram of petal width
plt.figure(figsize=(8, 5))
plt.hist(df['petal_width'], bins=20, color='purple', edgecolor='black')
plt.title('Histogram of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()
