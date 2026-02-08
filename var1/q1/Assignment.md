# Machine Learning Assignment: Breast Cancer Classification

**Dataset:** `Healthy.csv`

**Target Variable:** `diagnosis` (dependent variable)

**Features:** All other columns (independent variables)

---

## Instructions

Complete the following tasks using the provided dataset. Each question includes the point value indicated in parentheses.

---

### Question 1 (2 points)
**Data Preprocessing**

1. Remove all missing values using the `dropna()` function
2. Convert the categorical variable `diagnosis` to numeric format

---

### Question 2 (3 points)
**Random Forest Classification**

1. Split the data into training and testing sets
2. Train a `RandomForestClassifier` model from the `sklearn.ensemble` library
3. Calculate both the training score and the test score
4. Provide a comment explaining the difference between the training and test scores (e.g., discuss any overfitting or underfitting)

---

### Question 3 (5 points)
**Dimensionality Reduction with PCA**

1. Apply Principal Component Analysis (PCA) to reduce the dimensionality of the data
2. Create 4 new principal components (columns)
3. Train a new model using the transformed data
4. Calculate both the training score and the test score
5. Compare the results with those from Question 2 and discuss the impact of dimensionality reduction on model performance

---

## Notes

- Use appropriate random states for reproducibility
- Document your code with clear comments
- Print all scores and comparisons for evaluation
