# Machine Learning Learning Roadmap
## A Comprehensive Practical Guide Based on Course Materials

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Phase 1: Foundation & Data Preparation](#phase-1-foundation--data-preparation)
3. [Phase 2: Classical Machine Learning](#phase-2-classical-machine-learning)
4. [Phase 3: Feature Engineering & Dimensionality Reduction](#phase-3-feature-engineering--dimensionality-reduction)
5. [Phase 4: Ensemble Methods](#phase-4-ensemble-methods)
6. [Phase 5: Clustering Algorithms](#phase-5-clustering-algorithms)
7. [Phase 6: Neural Networks - Tabular Data](#phase-6-neural-networks---tabular-data)
8. [Phase 7: Deep Learning for Computer Vision](#phase-7-deep-learning-for-computer-vision)
9. [Phase 8: Regression Algorithms](#phase-8-regression-algorithms)
10. [Phase 9: Model Optimization & Tuning](#phase-9-model-optimization--tuning)
11. [Phase 10: Capstone Projects](#phase-10-capstone-projects)
12. [Exam Challenges](#exam-challenges)
13. [Recommended Resources](#recommended-resources)

---

## Prerequisites

### Before You Begin
- **Python Programming**: Intermediate level
- **Mathematics**: Basic statistics, linear algebra concepts
- **Libraries to Install**:
  ```bash
  pip install pandas numpy scikit-learn keras tensorflow opencv-python matplotlib seaborn
  ```

---

## Phase 1: Foundation & Data Preparation

### Week 1-2: Data Manipulation & Preprocessing

#### 🎯 Learning Objectives
- Master pandas for data manipulation
- Learn data cleaning techniques
- Understand train/test splitting
- Practice feature scaling methods

#### 📚 Topics Covered

**1. Reading and Understanding Data**
```python
import pandas as pd
import numpy as np

# Load CSV files
data = pd.read_csv('filename.csv')

# Explore dataset
print(data.head())
print(data.info())
print(data.describe())
print(data.shape)
```

**2. Handling Missing Values**
```python
# Method 1: Remove all missing elements
data = data.dropna()

# Method 2: Replace '?' with NaN
data = data.replace('?', np.nan)

# Convert string to numeric
data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'])

# Fill missing values with mean
data['Bare Nuclei'] = round(data['Bare Nuclei'].fillna(
    data['Bare Nuclei'].mean()
), 2)
```

**3. Converting Categorical Variables to Numeric**
```python
from sklearn.preprocessing import LabelEncoder

# Method 1: Label Encoding (for ordinal data)
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Method 2: One-Hot Encoding (for nominal data)
data = pd.get_dummies(data, columns=['category_column'], drop_first=True)

# Method 3: Manual mapping
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Method 4: Factorize (creates numeric codes)
data['class'] = pd.factorize(data['class'])[0]
```

**4. Train/Test Splitting**
```python
from sklearn.model_selection import train_test_split

X = data.drop('target_column', axis=1)
y = data['target_column']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**5. Feature Scaling**

*MinMaxScaler: Scales features to [0, 1] range*
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

*StandardScaler: Scales to mean=0, std=1*
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**When to use which scaler?**
- **MinMaxScaler**: When features have bounded ranges or when using neural networks
- **StandardScaler**: When features have Gaussian distribution or using algorithms like SVM, KNN

#### 📝 Practice Exercises
1. Clean the breast cancer dataset (handle '?' values)
2. Split data multiple ways (70/30, 80/20, 60/40)
3. Compare MinMaxScaler vs StandardScaler effects
4. Practice different categorical encoding methods

#### ✅ Completion Checklist
- [ ] Can load and explore any CSV dataset
- [ ] Understand different missing value strategies
- [ ] Know when to use different scalers
- [ ] Can properly split data avoiding data leakage
- [ ] Master categorical encoding techniques

---

## Phase 2: Classical Machine Learning

### Week 3-4: Individual Classification Algorithms

#### 🎯 Learning Objectives
- Understand core classification algorithms
- Learn how each algorithm makes decisions
- Evaluate model performance
- Compare different algorithms

#### 📚 Topics Covered

**1. Support Vector Machines (SVM)**
```python
from sklearn.svm import SVC

model = SVC(probability=True, kernel='rbf')
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
```

**2. LinearSVC (Linear Support Vector Classification)**
```python
from sklearn.svm import LinearSVC

# Basic LinearSVC
model = LinearSVC(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# With tunable parameters
model = LinearSVC(C=1.0, max_iter=10000, random_state=42)
# C parameter: Regularization parameter (smaller = more regularization)
```

**3. Naive Bayes**
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

**4. Decision Trees**
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

**5. K-Nearest Neighbors (KNN)**
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

**6. Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

#### 📝 Practice Exercises
1. Apply each algorithm to the churn classification dataset
2. Compare accuracy scores across all algorithms
3. Analyze which algorithm performs best for each dataset
4. Tune LinearSVC C parameter and observe differences

#### ✅ Completion Checklist
- [ ] Implemented all 6 base algorithms
- [ ] Understand strengths/weaknesses of each
- [ ] Can explain prediction mechanisms
- [ ] Documented performance comparisons

---

## Phase 3: Feature Engineering & Dimensionality Reduction

### Week 5: Advanced Feature Selection Techniques

#### 🎯 Learning Objectives
- Understand feature importance
- Learn automated feature selection
- Reduce dimensionality with PCA
- Improve model performance

#### 📚 Topics Covered

**1. SelectKBest with ANOVA F-value**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 3 features
selector = SelectKBest(score_func=f_classif, k=3)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]
print("Selected features:", selected_features.tolist())

# Apply SVC on selected features
from sklearn.svm import SVC
model = SVC()
model.fit(X_train_selected, y_train)
train_score = model.score(X_train_selected, y_train)
test_score = model.score(X_test_selected, y_test)
```

**2. Feature Selection Methods**
- **Filter Methods**: SelectKBest, SelectPercentile
- **Wrapper Methods**: RFE (Recursive Feature Elimination)
- **Embedded Methods**: L1 regularization, Tree-based importance

**3. PCA (Principal Component Analysis)**
```python
from sklearn.decomposition import PCA

# Create 4 new columns by reducing dimension
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Check explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

# Apply model on PCA-transformed data
model = SVC()
model.fit(X_train_pca, y_train)
train_score_pca = model.score(X_train_pca, y_train)
test_score_pca = model.score(X_test_pca, y_test)
```

**4. Comparing SelectKBest vs PCA**
```python
# SelectKBest approach
selector = SelectKBest(f_classif, k=3)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

model_kbest = SVC()
model_kbest.fit(X_train_kbest, y_train)
test_score_kbest = model_kbest.score(X_test_kbest, y_test)

# PCA approach
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model_pca = SVC()
model_pca.fit(X_train_pca, y_train)
test_score_pca = model_pca.score(X_test_pca, y_test)

# Comparison
print(f"SelectKBest Test Score: {test_score_kbest:.4f}")
print(f"PCA Test Score: {test_score_pca:.4f}")
print(f"Difference: {test_score_kbest - test_score_pca:.4f}")

if test_score_kbest > test_score_pca:
    print("SelectKBest performs better!")
else:
    print("PCA performs better!")
```

#### 📝 Practice Exercises
1. Apply SelectKBest to medical dataset with k=3
2. Compare PCA vs SelectKBest performance
3. Visualize PCA components
4. Analyze feature importance scores

#### ✅ Completion Checklist
- [ ] Understand different feature selection approaches
- [ ] Can determine optimal number of features/components
- [ ] Improved model performance through selection
- [ ] Documented PCA vs SelectKBest comparison

---

## Phase 4: Ensemble Methods

### Week 6-7: Combining Multiple Models

#### 🎯 Learning Objectives
- Understand ensemble learning principles
- Implement voting and stacking classifiers
- Learn RandomForest algorithm
- Compare ensemble vs individual models

#### 📚 Topics Covered

**1. Voting Classifier**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Hard Voting - Majority vote
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', LinearSVC(max_iter=10000, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ],
    voting='hard'  # 'hard' for majority vote, 'soft' for probabilities
)
voting_clf.fit(X_train, y_train)
test_score = voting_clf.score(X_test, y_test)
print(f"Voting Classifier Test Score: {test_score:.4f}")
```

**2. RandomForest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

# Comment on overfitting
score_diff = train_score - test_score
print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Difference: {score_diff:.4f}")
if score_diff > 0.1:
    print("Comment: Model may be overfitting (large difference)")
else:
    print("Comment: Model generalizes well (small difference)")
```

**3. Stacking Classifier**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier())
    ],
    final_estimator=LogisticRegression(max_iter=1000)
)
stacking_model.fit(X_train, y_train)
```

**4. GridSearchCV for Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

# For KNeighborsClassifier
parameters = {
    'n_neighbors': [7, 13, 17, 19, 31]
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=parameters,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best parameter: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# For LogisticRegression C parameter
param_grid = {
    'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.1]
}

log_reg = LogisticRegression(max_iter=10000)
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best C parameter: {grid_search.best_params_['C']}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

#### 📝 Practice Exercises
1. Build voting classifier with hard vs soft voting
2. Compare RandomForest vs individual Decision Trees
3. Use GridSearchCV to optimize multiple parameters
4. Compare stacking vs voting performance

#### ✅ Completion Checklist
- [ ] Understood difference between hard/soft voting
- [ ] Implemented RandomForest and analyzed feature importance
- [ ] Used GridSearchCV for optimization
- [ ] Documented performance improvements

---

## Phase 5: Clustering Algorithms

### Week 8: Unsupervised Learning

#### 🎯 Learning Objectives
- Understand clustering concepts
- Learn KMeans algorithm
- Determine optimal number of clusters
- Evaluate clustering quality

#### 📚 Topics Covered

**1. KMeans Clustering**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('Clustering_Final_Exam.csv')

# Standardize the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Determine optimal number of clusters using Elbow Method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
```

**2. Fit KMeans and Visualize**
```python
# Fit KMeans with optimal k (example: k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

# Transform centers back to original scale
centers_original = scaler.inverse_transform(centers)

# Scatter plot with clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('KMeans Clustering with Centers')
plt.legend()
plt.colorbar(label='Cluster')
plt.show()

print(f"Cluster Centers:\n{centers_original}")
```

**3. Silhouette Score Analysis**
```python
from sklearn.metrics import silhouette_score

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Interpretation
print("\nInterpretation:")
if silhouette_avg > 0.5:
    print("Excellent clustering - well-separated clusters")
elif silhouette_avg > 0.3:
    print("Good clustering - reasonable structure")
elif silhouette_avg > 0.1:
    print("Weak clustering - overlapping clusters")
else:
    print("Poor clustering - no meaningful structure")

# Find optimal k using silhouette scores
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)
plt.show()
```

**4. Visual Clustering Analysis**
```python
# Load test_cluster.csv for visual inspection
test_data = pd.read_csv('test_cluster.csv')

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(test_data)

# Scatter plot to visually determine clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Visual Inspection of Clusters')
plt.grid(True)
plt.show()

# From visual observation, determine k and fit KMeans
# (example: if we see 3 distinct groups)
k = 3  # Based on visual inspection
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

# Plot with centers
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title(f'KMeans Clustering (k={k})')
plt.legend()
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")
```

#### 📝 Practice Exercises
1. Use Elbow Method on Clustering_Final_Exam.csv
2. Apply KMeans and visualize cluster centers
3. Calculate and interpret silhouette scores
4. Compare clustering with different k values

#### ✅ Completion Checklist
- [ ] Understand KMeans algorithm
- [ ] Can determine optimal k using Elbow Method
- [ ] Can interpret silhouette scores
- [ ] Visualized clusters and centers

---

## Phase 6: Neural Networks - Tabular Data

### Week 9-11: Deep Learning for Structured Data

#### 🎯 Learning Objectives
- Understand neural network architecture
- Build multi-layer perceptrons
- Implement binary classification with NN
- Compare deep learning vs traditional ML

#### 📚 Topics Covered

**1. Simple Neural Network (2 layers, 4 nodes)**
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=4, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=50,
    epochs=30,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
train_score = model.evaluate(X_train, y_train, verbose=0)[1]
test_score = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
```

**2. Advanced Neural Network (7 layers, 40 nodes each)**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

nn_model = Sequential([
    Dense(40, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(40, activation='relu'),
    Dense(40, activation='relu'),
    Dense(40, activation='relu'),
    Dense(40, activation='relu'),
    Dense(40, activation='relu'),
    Dense(40, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = nn_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

**3. Deep Neural Network (11 layers for medical_data.csv)**
```python
# For medical diagnosis - binary classification
# Convert categorical column and scale data first

# Load and preprocess
data = pd.read_csv('medical_data.csv')

# Convert categorical to numeric
data['class'] = pd.factorize(data['class'])[0]

# Encode any remaining categorical columns
data = pd.get_dummies(data, drop_first=True)

# Split data
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build 11-layer neural network (15-25 nodes per layer)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(20, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(18, activation='relu'),
    Dense(22, activation='relu'),
    Dense(15, activation='relu'),
    Dense(25, activation='relu'),
    Dense(17, activation='relu'),
    Dense(20, activation='relu'),
    Dense(16, activation='relu'),
    Dense(24, activation='relu'),
    Dense(19, activation='relu'),
    Dense(21, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# Last 4 epochs output
print("\n=== Last 4 Epochs ===")
for i in range(-4, 0):
    epoch = history.epoch[i] + 1
    loss = history.history['loss'][i]
    acc = history.history['accuracy'][i]
    val_loss = history.history['val_loss'][i]
    val_acc = history.history['val_accuracy'][i]
    print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")

# Final scores
train_score = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
test_score = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
print(f"\nFinal Train Score: {train_score:.4f}")
print(f"Final Test Score: {test_score:.4f}")
```

**4. Key Concepts**
- **Layers**: Input, Hidden, Output
- **Activation Functions**: ReLU, Sigmoid, Softmax
- **Loss Functions**: Binary Crossentropy, Categorical Crossentropy
- **Optimizers**: SGD, Adam, RMSprop
- **Batch Size**: Number of samples per gradient update
- **Epochs**: Number of complete passes through training data

#### 📝 Practice Exercises
1. Build cancer classification with varying layers (2, 4, 6, 8, 11)
2. Experiment with different node counts (4, 10, 20, 40)
3. Compare optimizers (SGD vs Adam)
4. Create confusion matrix and classification report
5. Predict on new unseen data

#### ✅ Completion Checklist
- [ ] Built NN with different architectures
- [ ] Understood activation functions
- [ ] Tuned hyperparameters (layers, nodes, epochs)
- [ ] Compared NN vs traditional ML performance
- [ ] Made predictions on new data

---

## Phase 7: Deep Learning for Computer Vision

### Week 12-14: Convolutional Neural Networks & Transfer Learning

#### 🎯 Learning Objectives
- Work with image data
- Understand CNN architecture
- Build custom CNNs with Conv2D layers
- Apply transfer learning

#### 📚 Topics Covered

**1. Image Loading & Preprocessing**
```python
import cv2
import os
import numpy as np

# Load images from directory structure
X = []
y = []
subdirectories = os.listdir("train")

for label, name in enumerate(subdirectories):
    sub_folder = os.path.join("train", name)
    for file in os.listdir(sub_folder):
        image = cv2.imread(os.path.join(sub_folder, file))
        image = cv2.resize(image, (128, 128))
        X.append(image)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

**2. Custom CNN for Bottle Classification**
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 3 classes: beer, water, wine
num_classes = 3

model = Sequential([
    # First 2 Conv layers: 18-32 filters (choosing 24)
    Conv2D(24, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(24, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Last 5 Conv layers: 10-18 filters (choosing 14)
    Conv2D(14, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(14, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(14, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(12, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(12, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    # First 2 Dense layers: 10-20 nodes (choosing 15)
    Dense(15, activation='relu'),
    Dense(18, activation='relu'),

    # Last 3 Dense layers: 6 nodes each
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),

    # Output layer: 3 classes
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# Last 3 epochs
print("\n=== Last 3 Epochs ===")
for i in range(-3, 0):
    epoch = history.epoch[i] + 1
    loss = history.history['loss'][i]
    acc = history.history['accuracy'][i]
    val_loss = history.history['val_loss'][i]
    val_acc = history.history['val_accuracy'][i]
    print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")
```

**3. Transfer Learning with VGG19**
```python
from keras.applications import VGG19

# Load pre-trained model (excluding top layers)
base = VGG19(
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze the base model
base.trainable = False

# Create custom model
model = Sequential()
model.add(base)
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    batch_size=50,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)
```

**4. Making Predictions on Images**
```python
# Load and preprocess new image
test = cv2.imread("person520_bacteria_2205.jpeg")
test = cv2.resize(test, (128, 128))
test = np.expand_dims(test, axis=0)  # Add batch dimension
test = test.astype('float32') / 255.0

# Predict
prediction = model.predict(test)
if prediction[0][0] > 0.5:
    print("PNEUMONIA detected")
else:
    print("NORMAL")
```

#### 📝 Practice Exercises
1. Build pneumonia detection model
2. Build bottle classifier with custom CNN
3. Experiment with different image sizes (64, 128, 224)
4. Try different filter counts
5. Create data augmentation pipeline

#### ✅ Completion Checklist
- [ ] Loaded and preprocessed image datasets
- [ ] Understood CNN architecture
- [ ] Built custom CNN with Conv2D layers
- [ ] Applied transfer learning successfully
- [ ] Made predictions on new images

---

## Phase 8: Regression Algorithms

### Week 15: Predictive Modeling for Continuous Values

#### 🎯 Learning Objectives
- Understand regression vs classification
- Learn SVR for regression
- Apply RandomForestRegressor
- Handle categorical variables in regression

#### 📚 Topics Covered

**1. Data Preparation for Regression (charges.csv)**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('charges.csv')

# Convert all categorical columns to numeric
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Split data
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**2. Support Vector Regression (SVR)**
```python
from sklearn.svm import SVR

# Basic SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
train_score = svr.score(X_train, y_train)
test_score = svr.score(X_test, y_test)

print(f"SVR Train Score: {train_score:.4f}")
print(f"SVR Test Score: {test_score:.4f}")
print(f"Difference: {train_score - test_score:.4f}")

# SVR with PCA-reduced data and epsilon parameter
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# SVR with epsilon in [0, 1]
svr_pca = SVR(kernel='rbf', epsilon=0.1)
svr_pca.fit(X_train_pca, y_train)
train_score_pca = svr_pca.score(X_train_pca, y_train)
test_score_pca = svr_pca.score(X_test_pca, y_test)

print(f"\nSVR with PCA:")
print(f"Train Score: {train_score_pca:.4f}")
print(f"Test Score: {test_score_pca:.4f}")
```

**3. RandomForest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_reg.fit(X_train, y_train)

train_score_rf = rf_reg.score(X_train, y_train)
test_score_rf = rf_reg.score(X_test, y_test)

print(f"\nRandomForest Regressor:")
print(f"Train Score: {train_score_rf:.4f}")
print(f"Test Score: {test_score_rf:.4f}")

# Comparison with SVR
print(f"\n=== Model Comparison ===")
print(f"SVR Test Score: {test_score:.4f}")
print(f"SVR+PCA Test Score: {test_score_pca:.4f}")
print(f"RandomForest Test Score: {test_score_rf:.4f}")

# Determine best model
best_score = max(test_score, test_score_pca, test_score_rf)
if best_score == test_score_rf:
    print("\nConclusion: RandomForest Regressor performs best!")
elif best_score == test_score_pca:
    print("\nConclusion: SVR with PCA performs best!")
else:
    print("\nConclusion: SVR performs best!")
```

#### 📝 Practice Exercises
1. Apply SVR on charges.csv dataset
2. Compare SVR with and without PCA
3. Tune SVR epsilon parameter
4. Compare SVR vs RandomForestRegressor

#### ✅ Completion Checklist
- [ ] Understand regression evaluation metrics (R² score)
- [ ] Can preprocess data for regression
- [ ] Applied SVR with different parameters
- [ ] Compared regression models

---

## Phase 9: Model Optimization & Tuning

### Week 16: Advanced Techniques

#### 🎯 Learning Objectives
- Hyperparameter optimization strategies
- Cross-validation techniques
- Handling overfitting/underfitting
- Model evaluation metrics

#### 📚 Topics Covered

**1. Hyperparameter Tuning**
- Grid Search
- Random Search
- Bayesian Optimization

**2. Regularization Techniques**
- Dropout layers
- L1/L2 regularization
- Early stopping

**3. Model Evaluation**
```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

#### ✅ Completion Checklist
- [ ] Applied cross-validation
- [ ] Tuned hyperparameters systematically
- [ ] Implemented regularization
- [ ] Created comprehensive evaluation report

---

## Phase 10: Capstone Projects

### Week 17-18: Real-World Applications

#### 🎯 Project Ideas

**Project 1: Medical Diagnosis System**
- Combine all techniques learned
- Build cancer classification pipeline
- Deploy as web application

**Project 2: Customer Churn Prediction**
- Ensemble methods comparison
- Neural network optimization
- Feature importance analysis

**Project 3: Medical Image Analysis**
- Pneumonia detection enhancement
- Multi-class disease classification
- Model interpretability

#### ✅ Completion Checklist
- [ ] Completed at least 2 capstone projects
- [ ] Documented entire process
- [ ] Created model comparison report
- [ ] Prepared presentation of findings

---

## Exam Challenges

### 📚 Comprehensive Practice Problems

#### Challenge 1: Clustering Analysis (6 points)
**Dataset**: `Clustering_Final_Exam.csv`

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and standardize data
data = pd.read_csv('Clustering_Final_Exam.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Elbow Method to determine k
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Fit KMeans with optimal k
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

# Scatter plot with centers
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.title('KMeans Clustering with Centers')
plt.show()

# Silhouette score
score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {score:.4f}')
```

---

#### Challenge 2: VotingClassifier Ensemble (6 points)
**Dataset**: `medical_data_number_1.csv`
**Target**: `fetal_health`

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('medical_data_number_1.csv')
X = data.drop('fetal_health', axis=1)
y = data['fetal_health']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create VotingClassifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', LinearSVC(max_iter=10000, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ],
    voting='hard'
)

voting_clf.fit(X_train, y_train)
test_score = voting_clf.score(X_test, y_test)
print(f'VotingClassifier Test Score: {test_score:.4f}')
```

---

#### Challenge 3: Scaling & LinearSVC (6 points)
**Dataset**: `medical_data_number_1.csv`

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic LinearSVC
svc = LinearSVC(max_iter=10000, random_state=42)
svc.fit(X_train_scaled, y_train)
train_score = svc.score(X_train_scaled, y_train)
test_score = svc.score(X_test_scaled, y_test)

print(f'Train Score: {train_score:.4f}')
print(f'Test Score: {test_score:.4f}')

# Tune C parameter
svc_tuned = LinearSVC(C=0.5, max_iter=10000, random_state=42)
svc_tuned.fit(X_train_scaled, y_train)
train_score_tuned = svc_tuned.score(X_train_scaled, y_train)
test_score_tuned = svc_tuned.score(X_test_scaled, y_test)

print(f'\nTuned (C=0.5):')
print(f'Train Score: {train_score_tuned:.4f}')
print(f'Test Score: {test_score_tuned:.4f}')
print(f'Improvement: {test_score_tuned - test_score:.4f}')
```

---

#### Challenge 4: GridSearchCV with KNN (6 points)
**Dataset**: `medical_data_number_1.csv`

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors': [7, 13, 17, 19, 31]}
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Best parameter: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.4f}')
```

---

#### Challenge 5: Neural Network for Binary Classification (6 points)
**Dataset**: `medical_data.csv`
**Target**: `class`

```python
# Convert categorical and scale
data = pd.read_csv('medical_data.csv')
data['class'] = pd.factorize(data['class'])[0]

X = data.drop('class', axis=1)
y = data['class']

# Encode any remaining categorical columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11-layer NN
model = Sequential([
    Dense(20, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(18, activation='relu'),
    Dense(22, activation='relu'),
    Dense(15, activation='relu'),
    Dense(25, activation='relu'),
    Dense(17, activation='relu'),
    Dense(20, activation='relu'),
    Dense(16, activation='relu'),
    Dense(24, activation='relu'),
    Dense(19, activation='relu'),
    Dense(21, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test_scaled, y_test), verbose=1)

train_score = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
test_score = model.evaluate(X_test_scaled, y_test, verbose=0)[1]
print(f'Train Score: {train_score:.4f}')
print(f'Test Score: {test_score:.4f}')
```

---

#### Challenge 6: CNN for Bottle Classification (10 points)
**Dataset**: `Bottle/` folder (beer, water, wine)

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess images (see Phase 7 for details)
# X_train, X_test, y_train, y_test = load_bottle_images()

model = Sequential([
    Conv2D(25, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(28, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(12, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(15, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(10, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(18, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(14, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(15, activation='relu'),
    Dense(18, activation='relu'),
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)
```

---

#### Challenge 7: RandomForest with PCA (10 points)
**Dataset**: `Healthy.csv`
**Target**: `diagnosis`

```python
# Remove missing values and convert categorical
data = pd.read_csv('Healthy.csv')
data = data.dropna()
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest baseline
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)

print(f'Baseline RandomForest:')
print(f'Train: {train_score:.4f}, Test: {test_score:.4f}')
print(f'Difference: {train_score - test_score:.4f}')

# PCA with 4 components
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_pca, y_train)
train_pca = rf_pca.score(X_train_pca, y_train)
test_pca = rf_pca.score(X_test_pca, y_test)

print(f'\nWith PCA (4 components):')
print(f'Train: {train_pca:.4f}, Test: {test_pca:.4f}')
```

---

#### Challenge 8: SelectKBest vs PCA (6 points)
**Dataset**: `Healthy.csv`

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

# SelectKBest with 3 features
selector = SelectKBest(f_classif, k=3)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

svc_kbest = SVC()
svc_kbest.fit(X_train_kbest, y_train)
train_kbest = svc_kbest.score(X_train_kbest, y_train)
test_kbest = svc_kbest.score(X_test_kbest, y_test)

print(f'SelectKBest (3 features):')
print(f'Train: {train_kbest:.4f}, Test: {test_kbest:.4f}')

# Compare with PCA SVC
svc_pca = SVC()
svc_pca.fit(X_train_pca, y_train)
train_svc_pca = svc_pca.score(X_train_pca, y_train)
test_svc_pca = svc_pca.score(X_test_pca, y_test)

print(f'\nPCA (4 components) with SVC:')
print(f'Train: {train_svc_pca:.4f}, Test: {test_svc_pca:.4f}')

print(f'\nSelectKBest performs better by: {test_kbest - test_svc_pca:.4f}')
```

---

#### Challenge 9: GridSearchCV for LogisticRegression (6 points)
**Dataset**: `Healthy.csv`

```python
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.1]}
log_reg = LogisticRegression(max_iter=10000)

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Best C parameter: {grid_search.best_params_["C"]}')
print(f'Best score: {grid_search.best_score_:.4f}')
```

---

#### Challenge 10: SVR Regression (6 points)
**Dataset**: `charges.csv`
**Target**: `charges`

```python
# Convert categorical and split
data = pd.read_csv('charges.csv')
data = pd.get_dummies(data, drop_first=True)

X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)

print(f'SVR Train: {svr.score(X_train, y_train):.4f}')
print(f'SVR Test: {svr.score(X_test, y_test):.4f}')
```

---

#### Challenge 11: SVR with PCA (4 points)
**Dataset**: `charges.csv`

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svr_pca = SVR(epsilon=0.5)  # epsilon in [0, 1]
svr_pca.fit(X_train_pca, y_train)

print(f'SVR+PCA Train: {svr_pca.score(X_train_pca, y_train):.4f}')
print(f'SVR+PCA Test: {svr_pca.score(X_test_pca, y_test):.4f}')
```

---

#### Challenge 12: RandomForestRegressor (6 points)
**Dataset**: `charges.csv`

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

print(f'RandomForestRegressor Train: {rf_reg.score(X_train, y_train):.4f}')
print(f'RandomForestRegressor Test: {rf_reg.score(X_test, y_test):.4f}')

# Compare with SVR
print(f'\nComparison:')
print(f'SVR Test Score: {svr.score(X_test, y_test):.4f}')
print(f'RandomForest Test Score: {rf_reg.score(X_test, y_test):.4f}')
```

---

#### Challenge 13: Visual Clustering (6 points)
**Dataset**: `test_cluster.csv`

```python
# Visual inspection
test_data = pd.read_csv('test_cluster.csv')
plt.scatter(test_data.iloc[:, 0], test_data.iloc[:, 1])
plt.title('Visual Cluster Inspection')
plt.show()

# From visual observation, determine k
k = 3  # Example based on visual

scaler = StandardScaler()
X_scaled = scaler.fit_transform(test_data)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.title(f'KMeans Clustering (k={k})')
plt.show()

score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {score:.4f}')
```

---

## Recommended Resources

### 📚 Books
1. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron
2. *Deep Learning* by Ian Goodfellow, Yoshua Bengio, Aaron Courville
3. *Python Machine Learning* by Sebastian Raschka

### 🎓 Online Courses
1. Andrew Ng's Machine Learning Specialization (Coursera)
2. Deep Learning Specialization (Coursera)
3. Fast.ai Practical Deep Learning for Coders

### 🛠️ Datasets for Practice
1. [Kaggle Datasets](https://www.kaggle.com/datasets)
2. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
3. [Google Dataset Search](https://datasetsearch.research.google.com/)

### 📖 Documentation
1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
3. [OpenCV Documentation](https://docs.opencv.org/)

---

## 📊 Progress Tracking

### Milestones
- [ ] Week 1-2: Data Preprocessing Complete
- [ ] Week 3-4: Classical ML Algorithms Complete
- [ ] Week 5: Feature Engineering & PCA Complete
- [ ] Week 6-7: Ensemble Methods Complete
- [ ] Week 8: Clustering Algorithms Complete
- [ ] Week 9-11: Neural Networks Complete
- [ ] Week 12-14: Computer Vision Complete
- [ ] Week 15: Regression Algorithms Complete
- [ ] Week 16: Model Optimization Complete
- [ ] Week 17-18: Capstone Projects Complete
- [ ] All Exam Challenges Completed

### Skill Matrix
| Skill | Beginner | Intermediate | Advanced | Expert |
|-------|----------|--------------|----------|--------|
| Data Preprocessing | ☐ | ☐ | ☐ | ☐ |
| Classical ML | ☐ | ☐ | ☐ | ☐ |
| Feature Engineering | ☐ | ☐ | ☐ | ☐ |
| PCA & Dimensionality Reduction | ☐ | ☐ | ☐ | ☐ |
| Ensemble Methods | ☐ | ☐ | ☐ | ☐ |
| Clustering | ☐ | ☐ | ☐ | ☐ |
| Neural Networks | ☐ | ☐ | ☐ | ☐ |
| Computer Vision | ☐ | ☐ | ☐ | ☐ |
| Regression | ☐ | ☐ | ☐ | ☐ |
| Model Tuning | ☐ | ☐ | ☐ | ☐ |

---

## 💡 Tips for Success

1. **Practice Daily**: Even 30 minutes of coding helps
2. **Document Everything**: Keep notes on what works/doesn't
3. **Join Communities**: Stack Overflow, Reddit r/MachineLearning
4. **Build Portfolio**: Create GitHub repository with projects
5. **Stay Updated**: Follow ML researchers and practitioners
6. **Teach Others**: Explaining concepts reinforces learning
7. **Experiment Freely**: Try different parameters and architectures

---

## 🎯 Next Steps After Completion

1. **Specialization**: Choose NLP, Computer Vision, or Reinforcement Learning
2. **Kaggle Competitions**: Test skills against others
3. **Research Papers**: Read latest ML research
4. **Production Deployment**: Learn MLOps and model deployment
5. **Advanced Topics**: Attention mechanisms, Transformers, GANs

---

## 📅 Study Schedule Template

### Daily Routine (2-3 hours)
- 30 min: Review previous day's concepts
- 60-90 min: New learning material
- 30-60 min: Hands-on coding practice

### Weekly Routine
- Monday-Wednesday: New concepts
- Thursday: Review and practice
- Friday: Mini-project or challenge
- Weekend: Revision and exploration

---

**Created**: 2025-01-27
**Last Updated**: 2026-02-09
**Version**: 2.0

*This comprehensive roadmap covers all exam topics including clustering, ensemble methods, neural networks, CNN, regression, PCA, and model optimization.*
