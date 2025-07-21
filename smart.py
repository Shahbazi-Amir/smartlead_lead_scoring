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



# Replace 'Select' with NaN in columns that contain 'Select' as missing values
columns_with_select = ['Lead Profile', 'City', 'Specialization', 'How did you hear about X Education']
for col in columns_with_select:
    df[col] = df[col].replace('Select', np.nan)

# Display missing values after replacing 'Select'
print("\nMissing Values after replacing 'Select':")
print(df.isnull().sum())

# Handle missing values
# 1. Drop columns with too many missing values (>40%)
columns_to_drop = ['Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 
                   'Asymmetrique Activity Score', 'Asymmetrique Profile Score']
df = df.drop(columns=columns_to_drop)

# 2. Fill missing values for numerical columns with median
df['TotalVisits'] = df['TotalVisits'].fillna(df['TotalVisits'].median())
df['Page Views Per Visit'] = df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median())

# 3. Fill missing values for categorical columns with mode
df['Lead Source'] = df['Lead Source'].fillna(df['Lead Source'].mode()[0])
df['Last Activity'] = df['Last Activity'].fillna(df['Last Activity'].mode()[0])
df['Country'] = df['Country'].fillna(df['Country'].mode()[0])
df['Specialization'] = df['Specialization'].fillna(df['Specialization'].mode()[0])
df['How did you hear about X Education'] = df['How did you hear about X Education'].fillna(df['How did you hear about X Education'].mode()[0])
df['What is your current occupation'] = df['What is your current occupation'].fillna(df['What is your current occupation'].mode()[0])
df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].fillna(df['What matters most to you in choosing a course'].mode()[0])
df['Tags'] = df['Tags'].fillna(df['Tags'].mode()[0])
df['Lead Profile'] = df['Lead Profile'].fillna(df['Lead Profile'].mode()[0])
df['City'] = df['City'].fillna(df['City'].mode()[0])

# Display missing values after handling
print("\nMissing Values after handling:")
print(df.isnull().sum())







print(df['Lead Source'].mode()[0])  # پرتکرارترین مقدار Lead Source
print(df['City'].mode()[0])        # پرتکرارترین مقدار City








# Plot the distribution of the target variable (Converted)
plt.figure(figsize=(6, 4))
sns.countplot(x='Converted', data=df)
plt.title('Distribution of Converted Leads')
plt.xlabel('Converted (0 = Not Converted, 1 = Converted)')
plt.ylabel('Count')
plt.show()








# Plot histograms for numerical features
numerical_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
df[numerical_cols].hist(figsize=(12, 4), bins=20)
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()






plt.figure(figsize=(8, 5))
sns.boxplot(x='Converted', y='Total Time Spent on Website', data=df)
plt.title('Total Time Spent on Website vs Converted')
plt.xlabel('Converted (0 = Not Converted, 1 = Converted)')
plt.ylabel('Total Time Spent on Website (minutes)')
plt.show()






plt.figure(figsize=(10, 6))
sns.countplot(x='Lead Source', hue='Converted', data=df)
plt.title('Lead Source vs Converted')
plt.xlabel('Lead Source')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()






from sklearn.preprocessing import OneHotEncoder

# Load the cleaned dataset (assuming it's already cleaned as per previous steps)
df = pd.read_csv('dataset/leads.csv')

# Select categorical columns to encode
categorical_cols = ['Lead Source', 'Last Activity', 'Country', 'Specialization', 
                   'How did you hear about X Education', 'What is your current occupation', 
                   'What matters most to you in choosing a course', 'Tags', 'Lead Profile', 'City']

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity

# Fit and transform the categorical columns
encoded_cols = encoder.fit_transform(df[categorical_cols])

# Get the feature names from the encoder
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Convert the encoded data to a DataFrame
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_feature_names)

# Drop the original categorical columns from df
df = df.drop(columns=categorical_cols)

# Concatenate the encoded columns with the original DataFrame
df = pd.concat([df, encoded_df], axis=1)

# Display the first few rows to check
print(df.head())








# Import necessary libraries
from sklearn.preprocessing import StandardScaler


# Select numerical columns to scale
numerical_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical columns
scaled_cols = scaler.fit_transform(df[numerical_cols])

# Convert the scaled data to a DataFrame
scaled_df = pd.DataFrame(scaled_cols, columns=numerical_cols)

# Drop the original numerical columns from df
df = df.drop(columns=numerical_cols)

# Concatenate the scaled columns with the original DataFrame
df = pd.concat([df, scaled_df], axis=1)

# Display the first few rows to check
print(df.head())








# Import necessary libraries
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np

# Load the dataset (assuming it's already processed from previous steps)
# If you ran cell 1 and 2, use the updated df instead of reloading
# df = pd.read_csv('dataset/leads.csv')  # Uncomment only if starting fresh

# Define columns to keep (numerical and encoded ones)
columns_to_keep = ['Converted'] + [col for col in df.columns if col.startswith(('Total', 'Lead Source_', 'Last Activity_', 'Country_', 'Specialization_', 
                   'How did you hear about X Education_', 'What is your current occupation_', 
                   'What matters most to you in choosing a course_', 'Tags_', 'Lead Profile_', 'City_'))]

# Filter the DataFrame to keep only selected columns
df = df[columns_to_keep]

# Fill missing values
# For numerical columns (already scaled, but check for NaN)
numerical_cols = [col for col in df.columns if col.startswith('Total')]
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
# For encoded categorical columns
encoded_cols = [col for col in df.columns if col not in ['Converted'] + numerical_cols]
if encoded_cols:
    df[encoded_cols] = df[encoded_cols].fillna(df[encoded_cols].mode().iloc[0])

# Separate features (X) and target (y)
X = df.drop(columns=['Converted'])  # All columns except the target
y = df['Converted']  # Target variable

# Initialize SelectKBest with f_classif (for classification)
selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features

# Fit and transform the data
X_selected = selector.fit_transform(X, y)

# Get the selected feature names
selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask].tolist()

# Update the DataFrame with selected features
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df = pd.concat([df_selected, y.reset_index(drop=True)], axis=1)

# Display the first few rows and selected features
print("Selected Features:", selected_features)
print(df.head())







