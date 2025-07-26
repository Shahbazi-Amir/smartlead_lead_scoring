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



# Import necessary libraries for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb

# Ensure the DataFrame only contains selected features and target
selected_features = [
    'Lead Source_Reference', 
    'Last Activity_Olark Chat Conversation', 
    'Last Activity_SMS Sent', 
    'What is your current occupation_Unemployed', 
    'What is your current occupation_Working Professional', 
    'Tags_Closed by Horizzon', 
    'Tags_Interested in other courses', 
    'Tags_Ringing', 
    'Tags_Will revert after reading the email', 
    'Total Time Spent on Website'
]
X = df[selected_features]  # Features (only selected ones)
y = df['Converted']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Dictionary to store evaluation metrics
results = {'Model': [], 'F1-Score': [], 'AUC': []}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Store results
    results['Model'].append(model_name)
    results['F1-Score'].append(f1)
    results['AUC'].append(auc)
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot comparison of models
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='F1-Score', data=results_df, color='skyblue', label='F1-Score')
sns.barplot(x='Model', y='AUC', data=results_df, color='lightcoral', alpha=0.5, label='AUC')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.legend()
plt.show()

# Display results
print("\nModel Performance Summary:")
print(results_df)




import xgboost as xgb

# Use the trained XGBoost model from the previous step
# Assuming the XGBoost model is stored in 'models' dictionary with key 'XGBoost'
model_xgb = models['XGBoost']

# Get feature importance
feature_importance = model_xgb.feature_importances_
feature_names = [
    'Lead Source_Reference', 
    'Last Activity_Olark Chat Conversation', 
    'Last Activity_SMS Sent', 
    'What is your current occupation_Unemployed', 
    'What is your current occupation_Working Professional', 
    'Tags_Closed by Horizzon', 
    'Tags_Interested in other courses', 
    'Tags_Ringing', 
    'Tags_Will revert after reading the email', 
    'Total Time Spent on Website'
]

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, color='teal')
plt.title('Feature Importance in XGBoost Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Display the importance DataFrame
print("\nFeature Importance Summary:")
print(importance_df)



# Import necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score

# Define the feature set and target
selected_features = [
    'Lead Source_Reference', 
    'Last Activity_Olark Chat Conversation', 
    'Last Activity_SMS Sent', 
    'What is your current occupation_Unemployed', 
    'What is your current occupation_Working Professional', 
    'Tags_Closed by Horizzon', 
    'Tags_Interested in other courses', 
    'Tags_Ringing', 
    'Tags_Will revert after reading the email', 
    'Total Time Spent on Website'
]
X = df[selected_features]
y = df['Converted']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                          cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nPerformance of Best Model on Test Set:")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")


# Assume 'best_model' is the optimized model from the previous step
# Use the first 5 rows of the test set as a sample
sample_data = X_test.iloc[:5].copy()
sample_predictions = best_model.predict(sample_data)
sample_probabilities = best_model.predict_proba(sample_data)[:, 1]

# Create a DataFrame for the dashboard
dashboard_df = pd.DataFrame({
    'Lead Index': range(5),
    'Predicted Conversion': sample_predictions,
    'Conversion Probability': sample_probabilities
})

# Plot the conversion probabilities
plt.figure(figsize=(10, 6))
bars = plt.bar(dashboard_df['Lead Index'], dashboard_df['Conversion Probability'], color='lightgreen')
plt.title('Conversion Probability for Sample Leads')
plt.xlabel('Lead Index')
plt.ylabel('Probability of Conversion')
plt.ylim(0, 1)
for bar, prob in zip(bars, dashboard_df['Conversion Probability']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{prob:.2f}',
             ha='center', va='bottom')
plt.show()

# Display the dashboard table
print("\nDashboard Summary:")
print(dashboard_df)


# Use the entire test set
all_predictions = best_model.predict(X_test)
all_probabilities = best_model.predict_proba(X_test)[:, 1]

# Create a DataFrame for the full dashboard
dashboard_df = pd.DataFrame({
    'Lead Index': range(len(X_test)),
    'Predicted Conversion': all_predictions,
    'Conversion Probability': all_probabilities
})

# Sort by Conversion Probability in descending order
dashboard_df = dashboard_df.sort_values(by='Conversion Probability', ascending=False)

# Plot the top 10 leads
plt.figure(figsize=(12, 6))
bars = plt.bar(dashboard_df['Lead Index'][:10], dashboard_df['Conversion Probability'][:10], color='lightgreen')
plt.title('Top 10 Conversion Probabilities for All Leads')
plt.xlabel('Lead Index')
plt.ylabel('Probability of Conversion')
plt.ylim(0, 1)
for bar, prob in zip(bars, dashboard_df['Conversion Probability'][:10]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{prob:.2f}',
             ha='center', va='bottom')
plt.show()

# Display the top 10 leads
print("\nTop 10 Leads Dashboard Summary:")
print(dashboard_df.head(10))


plt.ylim(0.6, 1)
