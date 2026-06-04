# ============================================================
# Task 7: Decision Tree - Titanic Dataset
# Edutech Solution - Data Science Internship
# ============================================================

# ── STEP 1: Import Libraries ──────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   Task 7: Decision Tree on Titanic Dataset")
print("=" * 55)

# ── STEP 2: Load Titanic Dataset ──────────────────────────
# Titanic is built into seaborn — no download needed!
df = sns.load_dataset('titanic')
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ── STEP 3: Explore the Data ──────────────────────────────
print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n📊 Survival Count:")
print(df['survived'].value_counts())

# ── STEP 4: Preprocess the Data ───────────────────────────
# Select useful features
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target   = 'survived'

df_model = df[features + [target]].copy()

# Fill missing values
df_model['age'].fillna(df_model['age'].median(), inplace=True)
df_model['embarked'].fillna(df_model['embarked'].mode()[0], inplace=True)
df_model['fare'].fillna(df_model['fare'].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
df_model['sex']      = le.fit_transform(df_model['sex'])       # male=1, female=0
df_model['embarked'] = le.fit_transform(df_model['embarked'])  # C=0, Q=1, S=2

print("\n✅ Preprocessing done. Sample:")
print(df_model.head())

# ── STEP 5: Split into Train / Test ───────────────────────
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n✅ Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── STEP 6: Train Decision Tree ───────────────────────────
# max_depth=4 prevents overfitting (pruning via depth limit)
dt_model = DecisionTreeClassifier(
    criterion='gini',    # Gini Impurity as split metric
    max_depth=4,         # Pre-pruning: limits tree depth
    min_samples_split=10,
    random_state=42
)
dt_model.fit(X_train, y_train)
print("\n✅ Decision Tree trained successfully!")

# ── STEP 7: Evaluate the Model ────────────────────────────
y_pred = dt_model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\n" + "=" * 40)
print("📈 MODEL EVALUATION")
print("=" * 40)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("=" * 40)

print("\n📋 Full Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Survived', 'Survived']))

# ── STEP 8: Visualizations ────────────────────────────────

# --- 8a. Tree Plot (main deliverable) ---
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    dt_model,
    feature_names=features,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title("Decision Tree – Titanic Survival Prediction", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/tree_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Tree plot saved → tree_plot.png")

# --- 8b. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/confusion_matrix.png', dpi=150)
plt.close()
print("✅ Confusion matrix saved → confusion_matrix.png")

# --- 8c. Feature Importance ---
importances = pd.Series(dt_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', ax=ax)
ax.set_title('Feature Importances – Decision Tree')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=150)
plt.close()
print("✅ Feature importance saved → feature_importance.png")

# --- 8d. Text representation of tree ---
tree_rules = export_text(dt_model, feature_names=features)
print("\n🌳 Text Tree Structure (first 30 lines):")
print('\n'.join(tree_rules.split('\n')[:30]))

# ── STEP 9: Interview Q&A in Code Comments ────────────────
print("\n" + "=" * 55)
print("📚 INTERVIEW ANSWERS")
print("=" * 55)
print("""
Q1: What is Gini Impurity?
    Gini Impurity measures how often a randomly chosen element
    from a set would be incorrectly labelled.
    Formula: Gini = 1 - Σ(pᵢ²)
    • Gini = 0   → perfectly pure node (one class only)
    • Gini = 0.5 → maximally impure (50/50 split)
    The tree picks the split that MINIMISES Gini impurity.

Q2: What is Pruning in Decision Trees?
    Pruning reduces overfitting by simplifying the tree.
    Two types:
    • Pre-Pruning (Early Stopping): Stop growing the tree early
      using max_depth, min_samples_split, min_samples_leaf.
      → Used here: max_depth=4, min_samples_split=10
    • Post-Pruning: Grow the full tree, then remove weak branches.
      Scikit-learn: ccp_alpha (cost-complexity pruning).
""")

print("🎉 Task 7 Complete! All outputs saved.")
