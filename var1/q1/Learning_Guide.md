# Learning Guide: Breast Cancer Classification Assignment

This guide breaks down each question into smaller tasks with examples. Use this to learn the concepts and write your own code.

---

## QUESTION 1: Data Preprocessing (2 points)

### Task 1.1: Load the Dataset

**Concept:** Import pandas and read a CSV file into a DataFrame.

**Example pattern:**
```python
import pandas as pd
df = pd.read_csv("filename.csv")
```

**Your task:** Load `Healthy.csv` into a variable called `df`.

---

### Task 1.2: Remove Missing Values

**Concept:** The `dropna()` function removes rows containing any missing (NaN) values.

**Example pattern:**
```python
# Before: df has 150 rows
df = df.dropna()
# After: df might have 145 rows (5 rows with missing values removed)
```

**Your task:** Apply `dropna()` to your dataset.

---

### Task 1.3: Inspect the Data

**Concept:** Always check your data before processing it.

**Example patterns:**
```python
df.head()           # View first 5 rows
df.info()           # See data types and non-null counts
df['diagnosis'].unique()  # See unique values in a column
```

**Your task:** Check what unique values exist in the `diagnosis` column.

---

### Task 1.4: Convert Categorical to Numeric

**Concept:** Machine learning models need numbers, not text. Common methods:
- `map()`: Map specific values to numbers
- `replace()`: Replace values with numbers
- `LabelEncoder()`: Scikit-learn's encoder

**Example using map():**
```python
# Example: converting 'yes'/'no' to 1/0
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
```

**Example using replace():**
```python
# Example: converting 'M'/'B' to 1/0
df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})
```

**Your task:** Convert the `diagnosis` column to numeric values (0 and 1).

---

## QUESTION 2: Random Forest Classification (3 points)

### Task 2.1: Separate Features and Target

**Concept:** Split your data into X (features) and y (target).

**Example pattern:**
```python
# y = target column
# X = all columns except target
y = df['target_column']
X = df.drop('target_column', axis=1)
```

**Your task:** Create `X` (all columns except `diagnosis`) and `y` (the `diagnosis` column).

---

### Task 2.2: Split into Train and Test Sets

**Concept:** Always split data before training to evaluate model performance on unseen data.

**Example pattern:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42     # For reproducible results
)
```

**Common test_size values:** 0.2 (20%), 0.25 (25%), 0.3 (30%)

**Your task:** Split your data with 80% training and 20% testing.

---

### Task 2.3: Create and Train the Random Forest Model

**Concept:** RandomForestClassifier is an ensemble method that combines multiple decision trees.

**Example pattern:**
```python
from sklearn.ensemble import RandomForestClassifier

# Step 1: Create the model
model = RandomForestClassifier(random_state=42)

# Step 2: Train the model
model.fit(X_train, y_train)
```

**Your task:** Create and train a RandomForestClassifier.

---

### Task 2.4: Calculate Scores

**Concept:** The `.score()` method returns the accuracy of the model.

**Example pattern:**
```python
# Score on training data (how well it learned)
train_score = model.score(X_train, y_train)

# Score on test data (how well it generalizes)
test_score = model.score(X_test, y_test)

print(f"Training Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
```

**Your task:** Calculate and print both scores.

---

### Task 2.5: Interpret the Scores

**Concept:** Understanding the gap between training and test scores.

| Scenario | Train Score | Test Score | Interpretation |
|----------|-------------|------------|----------------|
| Good fit | High (e.g., 0.95) | High (e.g., 0.93) | Model performs well |
| Overfitting | Very high (e.g., 1.0) | Lower (e.g., 0.75) | Model memorized training data |
| Underfitting | Low (e.g., 0.70) | Low (e.g., 0.68) | Model too simple |

**Example comment:**
```python
# The training score is 0.98 and the test score is 0.92.
# The small difference (0.06) indicates the model generalizes well
# and is not significantly overfitting.
```

**Your task:** Add a comment explaining the difference between your scores.

---

## QUESTION 3: PCA Dimensionality Reduction (5 points)

### Task 3.1: Understanding PCA

**Concept:** Principal Component Analysis (PCA) reduces the number of features while preserving most of the information (variance).

**Example:**
- Original data: 30 columns (features)
- After PCA with n_components=4: 4 new columns (principal components)

---

### Task 3.2: Scale the Data First

**Concept:** PCA is sensitive to scale, so you must standardize data first.

**Example pattern:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Your task:** Scale your training and test data.

---

### Task 3.3: Apply PCA

**Concept:** Fit PCA on training data, then transform both train and test.

**Example pattern:**
```python
from sklearn.decomposition import PCA

# Create PCA with 4 components
pca = PCA(n_components=4)

# Fit on training data and transform
X_train_pca = pca.fit_transform(X_train_scaled)

# Transform test data (don't fit again!)
X_test_pca = pca.transform(X_test_scaled)

# Check the new shape
print(f"Original shape: {X_train.shape}")
print(f"After PCA shape: {X_train_pca.shape}")
```

**Your task:** Apply PCA to create 4 principal components.

---

### Task 3.4: Train a New Model on PCA Data

**Concept:** Use the same RandomForestClassifier, but with PCA-transformed data.

**Example pattern:**
```python
# Create new model
model_pca = RandomForestClassifier(random_state=42)

# Train on PCA data
model_pca.fit(X_train_pca, y_train)

# Calculate scores
train_score_pca = model_pca.score(X_train_pca, y_train)
test_score_pca = model_pca.score(X_test_pca, y_test)
```

**Your task:** Train a model on the PCA-transformed data and calculate scores.

---

### Task 3.5: Compare Results

**Concept:** Compare performance before and after PCA.

**Example comparison format:**
```python
print("=== Model Comparison ===")
print(f"Without PCA - Train: {train_score:.4f}, Test: {test_score:.4f}")
print(f"With PCA    - Train: {train_score_pca:.4f}, Test: {test_score_pca:.4f}")
print(f"Difference  - Train: {train_score - train_score_pca:.4f}, Test: {test_score - test_score_pca:.4f}")
```

**Example interpretation:**
```python
# PCA reduced the features from 30 to 4 components.
# The test score decreased from 0.92 to 0.88, which is a small
# performance drop for a significant reduction in complexity.
# This trade-off may be acceptable for faster training and reduced storage.
```

**Your task:** Compare the results and discuss the impact of PCA.

---

## HELPFUL TIPS

### Common Import Statements
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### Checking Your Work
```python
print(df.shape)              # Number of rows and columns
print(df.isnull().sum())     # Count of missing values per column
print(X_train.shape)         # Should show (samples, features)
print(X_train_pca.shape)     # Should show (samples, 4)
```

### Good Practices
1. Always use `random_state` for reproducible results
2. Fit scalers and PCA on training data only, then transform test data
3. Print shapes to verify transformations worked correctly
4. Add comments explaining what each step does

---

## CHECKLIST

- [ ] Loaded Healthy.csv
- [ ] Removed missing values with dropna()
- [ ] Converted diagnosis to numeric (0/1)
- [ ] Split into X (features) and y (target)
- [ ] Split into train and test sets
- [ ] Trained RandomForestClassifier
- [ ] Calculated and compared train/test scores
- [ ] Scaled data before PCA
- [ ] Applied PCA with 4 components
- [ ] Trained model on PCA data
- [ ] Compared results with and without PCA
