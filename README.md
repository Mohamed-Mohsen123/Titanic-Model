# 🚢 Titanic Survival Prediction — Decision Tree Classifier

A beginner-friendly machine learning project that predicts whether a Titanic passenger survived or not, using a **Decision Tree** model trained on real passenger data.

---

## 🎯 What This Project Does

Given information about a passenger (age, gender, ticket class, etc.), the model predicts:
- ✅ **Survived (1)**
- ❌ **Did Not Survive (0)**

---

## 📦 What's Covered

| Step | Topic |
|---|---|
| 1 | Imports & Setup |
| 2 | Loading the Titanic Dataset |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Data Preprocessing |
| 5 | Training the Decision Tree |
| 6 | Overfitting vs. Underfitting |
| 7 | Gini vs. Entropy Comparison |
| 8 | Hyperparameter Tuning (GridSearchCV) |
| 9 | Visualizing the Tree & Feature Importance |
| 10 | Making a Real Prediction |

---

## 🗂️ The Dataset

The **Titanic dataset** contains records of 891 passengers. Each row represents one person.

| Feature | Description |
|---|---|
| `pclass` | Ticket class — 1st, 2nd, or 3rd |
| `sex` | Gender |
| `age` | Passenger's age |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Price paid for the ticket |
| `embarked` | Port where they boarded (C, Q, or S) |
| `survived` | **Target** — 1 = survived, 0 = didn't |

---

## 🧠 Key Concepts (Before We Start)

### What is a Decision Tree?
A Decision Tree is a model that makes predictions by asking a series of yes/no questions — like a flowchart. At each step, it picks the question that best separates survivors from non-survivors.

```
Is the passenger male?
├── Yes → Did they pay less than £10?
│         ├── Yes → ❌ Likely didn't survive
│         └── No  → ✅ Likely survived
└── No  → ✅ Likely survived
```

### Gini vs. Entropy

Both measure how "mixed" a group is at each split. The tree always picks the split that reduces the mix the most.

| Criterion | Idea | Speed |
|---|---|---|
| **Gini** | Probability of misclassifying a random item | ⚡ Faster |
| **Entropy** | Measures disorder using Information Gain | 🐢 Slightly slower |

---

## 📋 Step-by-Step Code Walkthrough

---

### Step 1 — Imports & Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
%matplotlib inline
```

**What each import does:**

| Library | Purpose |
|---|---|
| `pandas` | Load and manipulate tabular data (like Excel but in Python) |
| `numpy` | Math operations on arrays and numbers |
| `matplotlib` | Draw charts and plots |
| `seaborn` | Higher-level charts built on top of matplotlib — prettier defaults |
| `sklearn.model_selection` | Split data, tune models, run cross-validation |
| `sklearn.tree` | The Decision Tree model itself + a function to draw it |
| `sklearn.metrics` | Tools to measure how good our model is |
| `warnings.filterwarnings('ignore')` | Suppresses minor warnings so output stays clean |
| `sns.set(style='whitegrid')` | Sets a clean white background for all seaborn plots |
| `%matplotlib inline` | Makes plots appear directly inside the notebook |

---

### Step 2 — Loading the Dataset

```python
titanic = sns.load_dataset('titanic')
titanic.head()
```

- `sns.load_dataset('titanic')` downloads the Titanic dataset built into seaborn — no file needed.
- `.head()` shows the first 5 rows so you can see what the data looks like.

```python
print(titanic.info())
print("\nMissing values:\n", titanic.isnull().sum())
```

- `.info()` prints column names, data types, and how many non-null values exist per column — useful for spotting missing data at a glance.
- `.isnull().sum()` counts how many missing values (NaN) exist in each column. For example, `age` is missing for ~177 passengers and `cabin` is missing for most.

---

### Step 3 — Exploratory Data Analysis (EDA)

EDA means looking at the data visually before building anything, to understand patterns and relationships.

#### 3a. Survival, Gender, and Class Charts

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.countplot(x='survived', data=titanic, ax=axes[0])
axes[0].set_title('Survival Count')
axes[0].set_xticklabels(['Did Not Survive', 'Survived'])

sns.countplot(x='sex', hue='survived', data=titanic, ax=axes[1])
axes[1].set_title('Survival by Gender')
axes[1].legend(['Did Not Survive', 'Survived'])

sns.countplot(x='pclass', hue='survived', data=titanic, ax=axes[2])
axes[2].set_title('Survival by Passenger Class')
axes[2].legend(['Did Not Survive', 'Survived'])

plt.tight_layout()
plt.show()
```

- `plt.subplots(1, 3, figsize=(15, 4))` creates a single row of **3 side-by-side plots**, each 15×4 inches total.
- `sns.countplot(x='survived', ...)` draws a bar chart counting how many passengers survived vs didn't.
- `hue='survived'` colors the bars by survival status — so you can see, for example, that far more women survived than men.
- `ax=axes[0]` tells seaborn which of the 3 subplots to draw in.
- `plt.tight_layout()` automatically adjusts spacing so titles and labels don't overlap.

**What you learn from these charts:**
- Most passengers did NOT survive.
- Women had a much higher survival rate than men.
- 1st class passengers survived more than 3rd class passengers.

#### 3b. Age and Fare Distributions

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(titanic['age'].dropna(), kde=True, bins=30, ax=axes[0])
axes[0].set_title('Age Distribution')

sns.histplot(titanic['fare'], kde=True, bins=30, ax=axes[1])
axes[1].set_title('Fare Distribution')

plt.tight_layout()
plt.show()
```

- `sns.histplot(...)` draws a histogram — bars showing how many passengers fall into each age/fare range.
- `kde=True` adds a smooth curve (Kernel Density Estimate) on top of the histogram to show the overall shape of the distribution.
- `bins=30` divides the range into 30 equal-width buckets.
- `.dropna()` on age is necessary because some age values are missing — histplot can't handle NaN.

**What you learn:**
- Most passengers were between 20–40 years old.
- Most fares were low (3rd class), with a long tail of very high fares (1st class).

---

### Step 4 — Data Preprocessing

Raw data can't go directly into a model. This step cleans and transforms it.

#### 4a. Select Relevant Columns

```python
columns_to_use = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
data = titanic[columns_to_use].copy()
```

- The original dataset has many columns we don't need (like `name`, `ticket`, `cabin`).
- We select only the 8 most useful features.
- `.copy()` creates an independent copy so any changes we make don't affect the original `titanic` dataframe.

#### 4b. Fill Missing Values

```python
data['age'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
```

- Machine learning models can't handle missing values (NaN) — they need a number in every cell.
- `data['age'].median()` calculates the middle value of all known ages — we use median instead of mean because it's less affected by extreme outliers (like a very old or very young passenger).
- `data['embarked'].mode()[0]` finds the most common port — the `.mode()` returns a Series so `[0]` picks the first (and most frequent) value.
- `inplace=True` means "apply this change directly to the dataframe, don't return a new one".

#### 4c. One-Hot Encoding (Converting Text to Numbers)

```python
data_encoded = pd.get_dummies(data, columns=['sex', 'embarked'], drop_first=True)
data_encoded.head()
```

Decision Trees (and most ML models) require **numbers**, not text. `sex` has values like "male"/"female" and `embarked` has "C"/"Q"/"S". We convert these using **one-hot encoding**:

- `pd.get_dummies(...)` creates new binary (True/False) columns for each category.
- `drop_first=True` drops one column per feature to avoid redundancy:
  - `sex` → becomes `sex_male` (True = male, False = female — we don't need both)
  - `embarked` → becomes `embarked_Q` and `embarked_S` (if both are False, the passenger boarded at C)

**Before encoding:**
```
sex       embarked
male      S
female    C
male      Q
```

**After encoding:**
```
sex_male  embarked_Q  embarked_S
True      False       True
False     False       False
True      True        False
```

#### 4d. Split into Features and Target

```python
X = data_encoded.drop('survived', axis=1)
y = data_encoded['survived']
```

- `X` = the **features** — everything the model is allowed to look at to make a prediction (age, sex, class, etc.)
- `y` = the **target** — the correct answer we're trying to predict (survived or not)
- `axis=1` means "drop a column" (axis=0 would drop a row)

#### 4e. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Training set size:', X_train.shape)
print('Testing set  size:', X_test.shape)
```

- We split the data into two sets: one to **train** the model, one to **test** it on data it has never seen before.
- `test_size=0.3` → 30% goes to testing (~267 passengers), 70% to training (~624 passengers).
- `random_state=42` → fixes the random shuffle so you get the same split every time you run the code (reproducibility).
- `.shape` prints `(rows, columns)` — useful to confirm the split worked correctly.

> 💡 Testing on data the model hasn't seen is the only honest way to know if it actually learned, not just memorized.

---

### Step 5 — Train the Model

```python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))
```

- `DecisionTreeClassifier(random_state=42)` creates a Decision Tree with default settings (no depth limit — it will grow until pure).
- `.fit(X_train, y_train)` trains the model — it learns which questions (splits) best separate survivors from non-survivors.
- `.predict(X_test)` runs the trained model on the test passengers and returns a prediction (0 or 1) for each.
- `accuracy_score(y_test, y_pred)` compares predictions to real answers and returns the percentage correct.

#### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

A confusion matrix shows **what type of mistakes** the model makes:

```
                  Predicted
                  Not Survived   Survived
Actual  Not Surv  [True Neg]     [False Pos]
        Survived  [False Neg]    [True Pos]
```

- **True Negative** — predicted didn't survive, actually didn't ✅
- **True Positive** — predicted survived, actually survived ✅
- **False Positive** — predicted survived, actually didn't ❌ (model was too optimistic)
- **False Negative** — predicted didn't survive, actually survived ❌ (model missed a survivor)

- `annot=True` prints the counts inside each cell.
- `fmt='d'` formats them as integers (not scientific notation).
- `cmap='Blues'` colors the heatmap in shades of blue — darker = higher count.

#### Classification Report

The report prints three metrics per class:

| Metric | Meaning |
|---|---|
| **Precision** | Of all passengers predicted as "survived", what % actually survived? |
| **Recall** | Of all passengers who actually survived, what % did the model catch? |
| **F1-Score** | Harmonic mean of precision and recall — a single balanced score |

---

### Step 6 — Overfitting vs. Underfitting

```python
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, model.predict(X_train)))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))
```

- This loop trains 20 different trees, each with a different `max_depth` (from 1 to 20).
- For each depth, it records both **train accuracy** and **test accuracy**.
- `max_depth` is the maximum number of questions the tree is allowed to ask. A depth of 1 = one question. A depth of 20 = very complex.

```python
plt.plot(depths, train_scores, marker='o', label='Train Accuracy', color='steelblue')
plt.plot(depths, test_scores, marker='o', label='Test Accuracy', color='coral')
plt.axvline(x=test_scores.index(max(test_scores)) + 1, color='green', linestyle='--', label='Best Test Depth')
```

- `plt.axvline(...)` draws a vertical dashed line at the depth where test accuracy peaks — the ideal `max_depth`.
- `test_scores.index(max(test_scores)) + 1` finds the position of the highest test score and adds 1 (because depth starts at 1, not 0).

**What you see in the plot:**

| Depth | Train Accuracy | Test Accuracy | Meaning |
|---|---|---|---|
| Very low (1–2) | Low | Low | Underfitting — tree is too simple |
| Medium (3–6) | Medium | High | ✅ Just right |
| Very high (15+) | ~100% | Drops | Overfitting — memorizing training data |

---

### Step 7 — Gini vs. Entropy Comparison

```python
for depth in depths:
    gini_model = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
    gini_model.fit(X_train, y_train)
    gini_scores.append(accuracy_score(y_test, gini_model.predict(X_test)))

    entropy_model = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
    entropy_model.fit(X_train, y_train)
    entropy_scores.append(accuracy_score(y_test, entropy_model.predict(X_test)))
```

- This loop trains two trees per depth — one using `gini`, one using `entropy` — and records the test accuracy of each.
- The plot lets you compare which criterion performs better at each depth.
- In most real-world datasets (including Titanic), they perform almost identically, but it's good practice to compare.

---

### Step 8 — Hyperparameter Tuning (GridSearchCV)

Instead of manually guessing good settings, we automate the search.

```python
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
```

- `max_depth` — how deep the tree can grow (`None` = unlimited)
- `min_samples_split` — minimum number of passengers in a node before the tree is allowed to split it further (higher = simpler tree)
- `criterion` — which impurity measure to use

This grid has 5 × 3 × 2 = **30 combinations**.

```python
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

- `cv=5` means **5-fold cross-validation**: the training set is split into 5 parts. The model trains on 4 parts and validates on the 5th — repeated 5 times, rotating which part is the validation set. The final score is the average.
- This means GridSearchCV actually trains 30 × 5 = **150 models** total to find the best combination.
- `scoring='accuracy'` — the metric used to compare combinations.

```python
print('Best Parameters:              ', grid_search.best_params_)
print('Best Cross-Validation Accuracy:', round(grid_search.best_score_, 4))
```

- `grid_search.best_params_` shows which combination of settings won.
- `grid_search.best_score_` is the average accuracy across the 5 folds for the best model.

```python
best_dt = grid_search.best_estimator_
y_pred_best = best_dt.predict(X_test)
```

- `grid_search.best_estimator_` retrieves the already-trained best model — ready to use directly, no need to re-train.

---

### Step 9 — Visualize the Tree & Feature Importance

#### Decision Tree Plot

```python
plt.figure(figsize=(22, 10))
plot_tree(best_dt,
          feature_names=X.columns,
          class_names=['Not Survived', 'Survived'],
          filled=True, rounded=True, fontsize=10)
plt.title('Optimized Decision Tree')
plt.show()
```

- `plot_tree(...)` draws the entire decision tree visually — every node, split, and leaf.
- `feature_names=X.columns` labels each split with the actual column name (e.g. "sex_male ≤ 0.5") instead of just an index number.
- `class_names=['Not Survived', 'Survived']` labels the leaf nodes with readable class names.
- `filled=True` colors each node — blue shades = leaning toward "Not Survived", orange shades = leaning toward "Survived". Darker = more confident.
- `rounded=True` makes the node boxes have rounded corners.

Each node in the diagram shows:
1. The question being asked (e.g. `sex_male <= 0.5`)
2. The impurity (gini or entropy value)
3. The number of samples reaching that node
4. The class distribution at that node

#### Feature Importances

```python
importances = best_dt.feature_importances_
indices = np.argsort(importances)[::-1]

sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance Score')
```

- `best_dt.feature_importances_` returns an array of scores — one per feature — showing how much each feature contributed to the tree's decisions.
- `np.argsort(importances)[::-1]` sorts the indices from highest to lowest importance (`::-1` reverses the order).
- The bar chart shows which features matter most. On Titanic, `sex_male` and `fare` are usually the top two.

> 💡 Feature importance is based on how much each feature reduces impurity across all splits in the tree, weighted by the number of samples.

---

### Step 10 — Prediction Example

```python
new_passenger = pd.DataFrame([{
    'pclass': 3,
    'age': 23.0,
    'sibsp': 1,
    'parch': 0,
    'fare': 6.25,
    'sex_male': True,
    'embarked_Q': True,
    'embarked_S': False
}])
```

- We manually build a passenger's data as a DataFrame — it must have the **exact same columns** in the exact same order as the training data, otherwise the model will error.
- `sex_male: True` → male passenger
- `embarked_Q: True, embarked_S: False` → boarded at Queenstown

```python
prediction = best_dt.predict(new_passenger)[0]
probability = best_dt.predict_proba(new_passenger)[0]
```

- `.predict(new_passenger)` returns `[0]` or `[1]` — the model's best guess.
- `[0]` at the end extracts the single value from the array.
- `.predict_proba(new_passenger)` returns the **probability** for each class: `[prob_not_survived, prob_survived]`. This is more informative than just a 0/1 answer.

```python
print(f'Survival Prediction : {"Survived" if prediction == 1 else "Did Not Survive"}')
print(f'Probability Not Survived : {probability[0]:.2%}')
print(f'Probability Survived     : {probability[1]:.2%}')
```

- `:.2%` formats a decimal like `0.847` as `84.70%` — clean and readable.

**Result for this passenger:**
```
Survival Prediction  : Did Not Survive
Probability Not Survived : ~85%
Probability Survived     : ~15%
```

A 3rd class male with a low fare had very poor survival odds — consistent with the historical data.

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run

```bash
jupyter notebook titanic_dt.ipynb
```

Run all cells from top to bottom. Each step builds on the previous one.

---

## 📁 File Structure

```
├── titanic_dt.ipynb   # Main notebook — all code and explanations
└── README.md          # This file
```
