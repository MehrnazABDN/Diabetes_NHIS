# Exploring Diabetes Risk Factors: A Data Science Analysis Using NHIS 2022 Data

## Table of Contents
- [ Project Overview](#project-overview)
- [Data Source](#data-source)
-  [Tools](#tools)
-  [Data Cleaning and Preparation](#data-cleaning-and-preparation)
-  [Exploratory Data Analysis](#exploratory-data-analysis)
-  [Data Analysis](#data-analysis)
-  [Results](#results)
-  [Key Findings and Recommendations](#key-findings-and-recommendations)

### Project Overview
Diabetes affects 422 million people globally and causes 1.5 million deaths annually. Key factors include obesity, inactivity, high blood pressure, and abnormal cholesterol. Projected to become the seventh leading cause of death by 2030, diabetes cost the USA $412.9 billion in 2022. This study uses the 2022 NHIS data to identify patterns in diabetes risk factors, aiming to inform strategies for reducing new cases and improving public health.

### Data Source
This study uses data from the 2022 National Health Interview Survey (NHIS) Adult dataset, conducted annually by the CDC. The dataset includes 27,651 participants and 637 variables, covering demographics, health status, chronic conditions, disability, health insurance, behaviors, lifestyle factors, healthcare utilization, mental health, and access to healthcare.

### Tools
- Python [Download Here](Link)

### Data Cleaning and Preparation
Key steps included:

1. Renaming columns for clarity.
2. Handling missing values.
3. Removing columns with many null values.
4. Imputing missing values where needed.
5. Encoding categorical variables.
This refined the dataset to 25,621 rows and 22 columns.

### Exploratory Data Analysis
The exploratory data analysis (EDA) phase utilized graphical techniques to gain insights into the NHIS dataset:

- What is the prevalence of diabetes in the dataset?
Visualized the prevalence, showing 21.4% had diabetes, and 78.6% did not.
![PercentageDiabetes](https://github.com/MehrnazABDN/Diabetes_NHIS/assets/132322441/804e20dd-ad1e-4990-bbcb-1b3f04c0a007)

- What is the gender distribution among participants with and without diabetes?
Showed 54.60% female and 45.40% male. Further analysis revealed differences in diabetes prevalence between genders.
  
![diabetes distribution](https://github.com/MehrnazABDN/Diabetes_NHIS/assets/132322441/7bf4cdad-485c-406d-94bd-89473155f80b)

- How does age distribution vary between individuals with and without diabetes?
Indicated a left-skewed distribution for diabetics and kurtosis for non-diabetics, spanning 18 to 99 years. The Mann-Whitney test showed a strong relationship between diabetes and age, focusing on individuals aged 40+, resulting in a filtered dataset of 18,307 rows.

- How can the dataset imbalance between diabetic and non-diabetic individuals be addressed?
Under-sampling yielded a balanced dataset with 3924 rows for each group.

### Data Analysis
- Bar Graphs: Visualized relationships between predictor variables and diabetes occurrence.
``` python
variables = ['Vegetable_Period', 'Soda_Period', 'Coffee_Tea_Sugar_Period',
             'Sleep_Rested', 'Potato_Fried_Period', 'Pizza_Period',
             'Physical_Activity_Time_Period', 'Smoke_Status', 'Depression',
             'Anxiety_Ever', 'Drinking_Status']


for variable in variables:

    contingency_table = pd.crosstab(df_balanced[variable], df_balanced['Diabetes_ever_or_borderline'])


    ax = contingency_table.plot(kind='bar', color=['blue', 'orange'])
    plt.title(f'Association between {variable} and Diabetes')
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.legend(title='Diabetes Status', labels=['No Diabetes', 'Diabetes'])
    plt.show()
```
- Cramer's V: Assessed associations between categorical variables and diabetes status.
  ```python
  ordinal_vars = ['Age', 'General_Health', 'Hypertension_Once', 'Cholesterol', 'Drinking_Status', 'Anxiety_Ever', 'Depression', 'Smoke_Status', 'Physical_Activity_Time_Period', 'Sleep_Rested', 'Soda_Period', 'Coffee_Tea_Sugar_Period', 'Sweetened_Fruit_Drinks_Period', 'Fruit_Period', 'Salad_Period', 'Potato_Fried_Period', 'Beans_Period', 'Vegetable_Period', 'Pizza_Period', 'Balance_Meal_Affordability']


for var in ordinal_vars:
    contingency_table = pd.crosstab(df['Diabetes_ever_or_borderline'], df[var])
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    print(f"Cramer's V for Diabetes_Status and {var}: {v}")
    ```

- Chi-Square and Correlation Analyses (Kendall's Tau, Spearman): Examined associations with ordinal variables and diabetes status.
```python
### Kendall's Tau Rank Correlation
from scipy.stats import kendalltau

variables = ['Gender', 'Age', 'General_Health', 'Hypertension_Once', 'Cholesterol', 'Drinking_Status', 'Anxiety_Ever', 'Depression', 'Smoke_Status', 'Physical_Activity_Time_Period', 'Sleep_Rested', 'Soda_Period', 'Coffee_Tea_Sugar_Period', 'Sweetened_Fruit_Drinks_Period', 'Fruit_Period', 'Salad_Period', 'Potato_Fried_Period', 'Beans_Period', 'Vegetable_Period', 'Pizza_Period', 'Balance_Meal_Affordability']

for variable in variables:
    tau, p_value = kendalltau(df[variable], df['Diabetes_ever_or_borderline'])
    print(f"Kendall's Tau correlation coefficient for {variable}: {tau}, p-value: {p_value}")
categorical_variables = ['Gender', 'General_Health', 'Hypertension_Once', 'Cholesterol', 'Drinking_Status',
                         'Anxiety_Ever', 'Depression', 'Smoke_Status', 'Physical_Activity_Time_Period',
                         'Sleep_Rested', 'Soda_Period', 'Coffee_Tea_Sugar_Period', 'Sweetened_Fruit_Drinks_Period',
                         'Fruit_Period', 'Salad_Period', 'Potato_Fried_Period', 'Beans_Period', 'Vegetable_Period',
                         'Pizza_Period', 'Balance_Meal_Affordability']


results = {}
for variable in categorical_variables:
    contingency_table = pd.crosstab(df[variable], df['Diabetes_ever_or_borderline'])
    chi2_stat, p_value_chi2, _, _ = chi2_contingency(contingency_table)
    results[variable] = {'Chi-square statistic': chi2_stat, 'P-value': p_value_chi2}

# Print the results
for variable, result in results.items():
    print("Chi-square Test for Association between", variable, "and Diabetes Status:")
    print("Chi-square statistic:", result['Chi-square statistic'])
    print("P-value:", result['P-value'])
    print()

### test Spearman correlation matrix

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency

variables_of_interest = ['Gender', 'Age', 'General_Health', 'Hypertension_Once', 'Cholesterol', 'Drinking_Status',
                         'Anxiety_Ever', 'Depression', 'Smoke_Status', 'Physical_Activity_Time_Period',
                         'Sleep_Rested', 'Soda_Period', 'Coffee_Tea_Sugar_Period', 'Sweetened_Fruit_Drinks_Period',
                         'Fruit_Period', 'Salad_Period', 'Potato_Fried_Period', 'Beans_Period', 'Vegetable_Period',
                         'Pizza_Period', 'Balance_Meal_Affordability']

df_subset = df[variables_of_interest]


correlation_matrix_spearman = df_subset.corr(method='spearman')


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Spearman Correlation Heatmap")
plt.show()

### test Spearman correlation matrix
for variable in variables_of_interest:
    cross_tab = pd.crosstab(df['Diabetes_ever_or_borderline'], df[variable])
    chi2, p_value, _, _ = chi2_contingency(cross_tab)
    print(f"For variable {variable}:")
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p_value}\n")
```
![spearman heatmap](https://github.com/MehrnazABDN/Diabetes_NHIS/assets/132322441/8c733862-e3c6-478b-a20e-352e32e5a32e)

- Logistic Regression: Predicted diabetes occurrence using 'Diabetes_ever_or_borderline' based on predictor variables.
```python
# Define the features (X) and the target variable (y)
X = df_balanced.drop(columns=['Diabetes_ever_or_borderline', 'Age_Group'])
y = df_balanced['Diabetes_ever_or_borderline']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, sm.add_constant(X_train_scaled))
result = logit_model.fit()

print(result.summary())
```
```python
X = df[['General_Health', 'Hypertension_Once', 'Cholesterol', 'Physical_Activity_Time_Period', 'Sleep_Rested', 'Soda_Period', 'Sweetened_Fruit_Drinks_Period', 'Pizza_Period']]
y = df['Diabetes_ever_or_borderline']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg_model = LogisticRegression()

log_reg_model.fit(X_train_scaled, y_train)


train_accuracy = log_reg_model.score(X_train_scaled, y_train)
test_accuracy = log_reg_model.score(X_test_scaled, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
```

### Results
- Prevalence of Diabetes: 25.34% of individuals in the NHIS dataset had diabetes.

- Gender Distribution: The dataset consisted of 54.60% females and 45.50% males.

- Cramer's V: Weak to moderate associations were found for "Drinking_Status" and "Smoke_Status", while "Hypertension" and "Cholesterol" showed stronger associations with diabetes.

- Mann-Whitney Test: Significant association between diabetes and age (U-statistic: 61696184.0, p-value: 6.05e-280) was observed among individuals aged 40 and older.

- Chi-Square Tests: Highly significant associations (p < 0.001) were found between diabetes status and "Hypertension_Once", "General_Health", and "Cholesterol".

- Cramer's V Result: "Hypertension_Once", "General_Health", and "Cholesterol" demonstrated stronger associations (0.23 to 0.24), whereas "Smoke_Status" showed weaker association (0.024).

- Kendall’s Tau: Strong correlations were observed for "Hypertension_Once" (τ = 0.239, p < 0.001), "General_Health" (τ = -0.215, p < 0.001), and "Cholesterol" (τ = 0.230, p < 0.001) with diabetes status.

- Logistic Regression: Significant predictors included "Hypertension_Once", "Cholesterol", "Depression", and certain dietary habits (p < 0.05). The model exhibited good performance with training accuracy of 0.781 and testing accuracy of 0.792.


### Key Findings and Recommendations

- Findings: This study identifies hypertension, cholesterol levels, depression, and dietary habits as significant predictors of Type 2 Diabetes (T2D) risk among adults, highlighting the crucial role of maintaining a healthy lifestyle with adequate sleep and physical activity to mitigate these risks.

- Limitations: Limitations include reliance on self-reported data and the NHIS dataset's cross-sectional nature, which restricts longitudinal analysis of diet and diabetes onset trends.

- Recommendations: Future research should prioritize longitudinal studies employing objective measures to validate findings and explore broader socio-economic and environmental factors alongside lifestyle choices that influence T2D risk.

