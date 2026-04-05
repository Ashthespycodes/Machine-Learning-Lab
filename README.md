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

### Lab 6 – K-Nearest Neighbor and SVM Classification Models
- **KNN**: Found optimal K (1–20) using accuracy vs K plot; evaluated with accuracy, precision, recall, F1 score, confusion matrix, and ROC curve (AUC)
- **SVM**: Compared linear, RBF, poly, and sigmoid kernels; best kernel (RBF) achieved 100% accuracy; evaluated with full classification report and ROC curve
- **Comparison**: Side-by-side performance bar chart for KNN vs SVM across all metrics

### Lab 7 – Bagging, Boosting and Stacking Ensemble Methods
- **Bagging**: Trained BaggingClassifier (Decision Tree base) and Random Forest; plotted feature importances
- **Boosting**: Trained AdaBoost and Gradient Boosting classifiers with ROC curve comparison
- **Stacking**: Combined DT, KNN, and SVM base learners with Logistic Regression meta-learner (100% accuracy)
- **Comparison**: Overall performance bar chart across all five ensemble models

### Lab 8 – K-Means and Hierarchical Clustering
- **K-Means**: Used Elbow method and Silhouette scores to find optimal K; visualized clusters in 2D PCA space with centroids; evaluated against true labels using ARI and NMI
- **Hierarchical**: Plotted dendrogram (Ward linkage, sample of 80); compared Ward, complete, average, and single linkage methods; visualized best clustering in PCA space
- **Comparison**: Side-by-side PCA scatter plots (true labels vs K-Means vs Hierarchical) and quality metrics bar chart
