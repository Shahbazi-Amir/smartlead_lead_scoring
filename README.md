# SmartLead: AI-Powered Lead Scoring System

## 🧠 Project Objective
The goal of this project is to develop a machine learning classification model that predicts the likelihood of a sales lead converting into a customer. This helps marketing and sales teams prioritize high-potential leads efficiently.

---

## 📊 Dataset
The dataset contains detailed attributes for each lead such as:

- Lead Source
- Last Activity
- Occupation
- Tags
- Time Spent on Website
- City, Country, etc.
- Target: `Converted` (1 = Converted, 0 = Not Converted)

---

## ⚙️ Data Preprocessing

- Replaced 'Select' entries with `NaN`.
- Removed columns with >40% missing data.
- Filled missing values:
  - Numerical: median
  - Categorical: mode
- One-Hot Encoding for categorical features.
- Standardization for numerical features.

---

## 🔍 Feature Selection

Used `SelectKBest` with ANOVA F-test to select top 10 features:
- `Lead Source_Reference`
- `Tags_Closed by Horizzon`
- `Tags_Ringing`
- `What is your current occupation_Working Professional`
- `Tags_Will revert after reading the email`
- `What is your current occupation_Unemployed`
- `Last Activity_SMS Sent`
- `Tags_Interested in other courses`
- `Last Activity_Olark Chat Conversation`
- `Total Time Spent on Website`

---

## 🤖 Models Used & Performance

| Model                | F1-Score | AUC    |
|---------------------|----------|--------|
| Logistic Regression | 0.798    | 0.895  |
| Random Forest       | 0.767    | 0.878  |
| XGBoost             | **0.793**| **0.913** |

- **XGBoost** achieved the best performance.
- Hyperparameter tuning with `GridSearchCV` improved performance further.

---

## 🔬 Feature Importance (XGBoost)

Top features influencing lead conversion:
1. `Lead Source_Reference`
2. `Tags_Closed by Horizzon`
3. `Tags_Ringing`
4. `Occupation`
5. `Last Activity`

---

## 📈 Lead Conversion Dashboard

- Probability scores computed for each lead using the best model.
- Dashboard highlights top 10 leads with >99% conversion probability.
- Marketing recommendation: `Follow-up Call`

---

## ✅ Conclusion

This end-to-end lead scoring system helps:
- Identify leads with high conversion probability
- Reduce time wasted on low-quality leads
- Improve sales targeting using data-driven insights

> The full implementation is available in `smartlead_lead_scoring.ipynb`.
