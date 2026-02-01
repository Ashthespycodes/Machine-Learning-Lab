# Machine-Learning-Lab
Implementing all the machine learning techniques on a predefined dataset.

## Dataset
**Chronic Kidney Disease (CKD)** dataset from the UCI Machine Learning Repository (400 samples, 25 features).

## Labs

### Lab 1 – Data Preprocessing and Visualization
- Fetched CKD dataset using `ucimlrepo`
- Explored data shape, types, and missing values
- Handled missing values (median for numeric, mode for categorical)
- Label encoded categorical features
- Visualized class distribution (CKD vs Not CKD)

### Lab 2 – CKD Prediction using Logistic Regression
- Split data into training (80%) and testing (20%) sets
- Applied `StandardScaler` for feature scaling
- Built a Logistic Regression classifier
- Evaluated with accuracy, confusion matrix, and classification report

### Lab 3 – Descriptive Statistical Analysis
- Computed mean, median, and mode for numeric features
- Computed min, max, and sum
- Computed standard deviation and variance
- Calculated quartiles (Q1, Q2, Q3) and percentiles (10th, 25th, 50th, 75th, 90th)
- Generated correlation matrix with heatmap
- Generated covariance matrix with heatmap
- Produced complete descriptive summary using `df.describe()`

### Lab 4 – Linear and Logistic Regression Models
- **Linear Regression**: Predicted hemoglobin levels from numeric features; evaluated with MAE, MSE, RMSE, and R²; plotted actual vs predicted and residual plots
- **Logistic Regression**: Predicted CKD class from all features; evaluated with accuracy, precision, recall, F1 score, confusion matrix, and ROC curve with AUC
