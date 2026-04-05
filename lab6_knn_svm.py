# Generated from: lab6_knn_svm.ipynb
# Converted at: 2026-03-08T20:33:00.177Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Lab 6 – K-Nearest Neighbor and SVM Classification Models
# Build K-Nearest Neighbor and SVM classification models and evaluate their performance using appropriate metrics.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

ckd = fetch_ucirepo(id=336)

X = ckd.data.features
y = ckd.data.targets

df = pd.concat([X, y], axis=1)
df.head()

# Replace '?' with NaN and handle missing values
df.replace('?', np.nan, inplace=True)

num_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
cat_cols = [col for col in df.columns if col not in num_cols + ['class']]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df['class'] = df['class'].str.strip().map({'ckd': 0, 'notckd': 1})

print("Shape:", df.shape)
print("Class distribution:\n", df['class'].value_counts())
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_feat = df.drop('class', axis=1)
y_target = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_target, test_size=0.2, random_state=42, stratify=y_target
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Training set:", X_train_sc.shape)
print("Test set:    ", X_test_sc.shape)

# ---
# ## Part A – K-Nearest Neighbor (KNN)
# Classify CKD patients using KNN and find the optimal value of K.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Find best K
k_values = range(1, 21)
train_acc = []
test_acc  = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, y_train)
    train_acc.append(accuracy_score(y_train, knn.predict(X_train_sc)))
    test_acc.append(accuracy_score(y_test, knn.predict(X_test_sc)))

best_k = k_values[np.argmax(test_acc)]
print(f"Best K = {best_k}  |  Test Accuracy = {max(test_acc):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_values, train_acc, marker='o', label='Train Accuracy')
plt.plot(k_values, test_acc,  marker='s', label='Test Accuracy')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs K')
plt.legend()
plt.tight_layout()
plt.show()

# Train final KNN model with best K
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_sc, y_train)
y_knn_pred = knn_model.predict(X_test_sc)

# ### KNN – Evaluation Metrics


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

print("=== KNN Evaluation Metrics ===")
print(f"Accuracy  : {accuracy_score(y_test, y_knn_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_knn_pred, average='weighted'):.4f}")
print(f"Recall    : {recall_score(y_test, y_knn_pred, average='weighted'):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_knn_pred, average='weighted'):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_knn_pred, target_names=['CKD', 'Not CKD']))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_knn_pred)
ConfusionMatrixDisplay(cm_knn, display_labels=['CKD', 'Not CKD']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title('KNN: Confusion Matrix')

# ROC Curve
y_knn_prob = knn_model.predict_proba(X_test_sc)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_knn_prob)
roc_auc_knn = auc(fpr_knn, tpr_knn)
axes[1].plot(fpr_knn, tpr_knn, color='steelblue', lw=2, label=f'AUC = {roc_auc_knn:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('KNN: ROC Curve')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()

# ---
# ## Part B – Support Vector Machine (SVM)
# Classify CKD patients using SVM with different kernels.


from sklearn.svm import SVC

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
svm_results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)
    y_pred = svm.predict(X_test_sc)
    svm_results[kernel] = {
        'model': svm,
        'pred': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    print(f"Kernel: {kernel:8s} | Accuracy: {svm_results[kernel]['accuracy']:.4f} | F1: {svm_results[kernel]['f1']:.4f}")

# Best SVM kernel
best_kernel = max(svm_results, key=lambda k: svm_results[k]['accuracy'])
print(f"Best kernel: {best_kernel}")

best_svm = svm_results[best_kernel]['model']
y_svm_pred = svm_results[best_kernel]['pred']

print("\n=== SVM Evaluation Metrics ===")
print(f"Accuracy  : {accuracy_score(y_test, y_svm_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_svm_pred, average='weighted'):.4f}")
print(f"Recall    : {recall_score(y_test, y_svm_pred, average='weighted'):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_svm_pred, average='weighted'):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_svm_pred, target_names=['CKD', 'Not CKD']))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_svm_pred)
ConfusionMatrixDisplay(cm_svm, display_labels=['CKD', 'Not CKD']).plot(ax=axes[0], cmap='Oranges')
axes[0].set_title(f'SVM ({best_kernel} kernel): Confusion Matrix')

# ROC Curve
y_svm_prob = best_svm.predict_proba(X_test_sc)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_svm_prob)
roc_auc_svm = auc(fpr_svm, tpr_svm)
axes[1].plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'AUC = {roc_auc_svm:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title(f'SVM ({best_kernel} kernel): ROC Curve')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()

# ---
# ## Model Comparison: KNN vs SVM


comparison = pd.DataFrame({
    'Model': ['KNN', f'SVM ({best_kernel})'],
    'Accuracy':  [accuracy_score(y_test, y_knn_pred), accuracy_score(y_test, y_svm_pred)],
    'Precision': [precision_score(y_test, y_knn_pred, average='weighted'),
                  precision_score(y_test, y_svm_pred, average='weighted')],
    'Recall':    [recall_score(y_test, y_knn_pred, average='weighted'),
                  recall_score(y_test, y_svm_pred, average='weighted')],
    'F1 Score':  [f1_score(y_test, y_knn_pred, average='weighted'),
                  f1_score(y_test, y_svm_pred, average='weighted')],
    'ROC AUC':   [roc_auc_knn, roc_auc_svm]
})
comparison.set_index('Model', inplace=True)
print(comparison.to_string())

comparison.plot(kind='bar', figsize=(10, 5), ylim=(0.8, 1.02), rot=0)
plt.title('KNN vs SVM – Performance Comparison')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()