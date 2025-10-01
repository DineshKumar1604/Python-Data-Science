# ===================================================================
# 04_machine_learning_models.py
# Description: Implements and evaluates several supervised machine
# learning models using the Scikit-Learn library.
# ===================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# --- 1. Simple Linear Regression ---
print("\n--- Machine Learning: Simple Linear Regression ---")
# Sample data: house prices based on square footage
square_footage = np.array([1500, 1600, 1700, 1800, 1900, 2000]).reshape(-1, 1)
prices = np.array([300000, 320000, 340000, 360000, 380000, 400000])

lr_model = LinearRegression()
lr_model.fit(square_footage, prices)
predicted_prices = lr_model.predict(square_footage)

print(f"Intercept: {lr_model.intercept_:.2f}")
print(f"Slope (Coefficient): {lr_model.coef_[0]:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(square_footage, prices, color='blue', label='Actual Prices')
plt.plot(square_footage, predicted_prices, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: House Prices')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# --- 2. Logistic Regression for Classification ---
print("\n--- Machine Learning: Logistic Regression ---")
# Sample data: exam pass/fail based on hours studied and previous score
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'previous_score': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df_exam = pd.DataFrame(data)
X = df_exam[['hours_studied', 'previous_score']]
y = df_exam['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_pred_prob = log_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()


# --- 3. Decision Tree and Support Vector Machine (SVM) on Iris Dataset ---
print("\n--- Machine Learning: Decision Tree & SVM on Iris Dataset ---")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Iris Classification')
plt.show()

# Support Vector Classifier (SVC)
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print(f"\nSVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))


print("\n--- END OF MACHINE LEARNING SCRIPT ---")
