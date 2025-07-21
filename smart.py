# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Kaggle Leads Dataset
# Note: Replace 'path_to_file/leads.csv' with the actual path to your downloaded CSV file
df = pd.read_csv('dataset/leads.csv')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset info
print("\nDataset Info:")
print(df.info())


# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())