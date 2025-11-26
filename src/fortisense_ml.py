import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ============================================================
# FortiSense – Part 2: Classical Machine Learning Models
#
# This module:
#   - Loads the KDD training and testing datasets.
#   - Splits the data into features and labels.
#   - Trains two ML models:
#       1) Random Forest (tree-based ensemble)
#       2) Linear SVM (margin-based classifier, with feature scaling)
#   - Evaluates both models using accuracy, precision, recall and F1 score.
#   - Prints a concise comparison table for reporting.
# ============================================================

# Determines the project root so that dataset paths work reliably.
project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

training_dataset_path = os.path.join(dataset_directory, "KDDTrain.csv")
testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

# ------------------------------------------------------------
# 1. Load training and testing datasets
# ------------------------------------------------------------

# Loads the KDD training and testing datasets into memory.
training_dataframe = pd.read_csv(training_dataset_path)
testing_dataframe = pd.read_csv(testing_dataset_path)

print("=== FortiSense – ML Models: Data Loaded ===")
print("Training dataset shape:", training_dataframe.shape)
print("Testing dataset shape: ", testing_dataframe.shape)
print()

# ------------------------------------------------------------
# 2. Separate features and labels
# ------------------------------------------------------------

# Identifies all feature columns by excluding the label and attack_type columns.
feature_column_names = [
    column_name
    for column_name in training_dataframe.columns
    if column_name not in ("label", "attack_type")
]

# Extracts the feature matrices for training and testing.
training_feature_matrix = training_dataframe[feature_column_names].values
testing_feature_matrix = testing_dataframe[feature_column_names].values

# Extracts the binary labels: 0 = normal, 1 = attack.
training_labels = training_dataframe["label"].values
testing_labels = testing_dataframe["label"].values

print("Number of features used:", len(feature_column_names))
print("Example feature columns:", feature_column_names[:5])
print()

# ------------------------------------------------------------
# Helper function for metric computation
# ------------------------------------------------------------

def evaluate_classification_model(true_labels, predicted_labels, model_name):
    """
    Computes and prints standard classification metrics for a given model.

    The metrics include:
      - Accuracy
      - Precision
      - Recall
      - F1 score

    Returns a dictionary with the metric values for later comparison.
    """
    accuracy_value = accuracy_score(true_labels, predicted_labels)
    precision_value = precision_score(true_labels, predicted_labels, zero_division=0)
    recall_value = recall_score(true_labels, predicted_labels, zero_division=0)
    f1_value = f1_score(true_labels, predicted_labels, zero_division=0)

    print(f"=== Evaluation Results for {model_name} ===")
    print(f"Accuracy : {accuracy_value:.4f}")
    print(f"Precision: {precision_value:.4f}")
    print(f"Recall   : {recall_value:.4f}")
    print(f"F1 score : {f1_value:.4f}")
    print()

    return {
        "model": model_name,
        "accuracy": accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "f1_score": f1_value,
    }

# ------------------------------------------------------------
# 3. Model 1 – Random Forest Classifier
# ------------------------------------------------------------

# The Random Forest model is a tree-based ensemble that generally performs well on tabular data
# and does not require feature scaling.
random_forest_classifier = RandomForestClassifier(
    n_estimators=100,   # number of trees in the forest
    random_state=42,    # ensures reproducible results
    n_jobs=-1,          # uses all available CPU cores
)

# Trains the Random Forest classifier on the full KDD training feature matrix.
random_forest_classifier.fit(training_feature_matrix, training_labels)

# Generates predictions for the testing feature matrix.
random_forest_predictions = random_forest_classifier.predict(testing_feature_matrix)

# Evaluates the Random Forest model and stores its metrics.
random_forest_metrics = evaluate_classification_model(
    true_labels=testing_labels,
    predicted_labels=random_forest_predictions,
    model_name="Random Forest",
)

# ------------------------------------------------------------
# 4. Model 2 – Linear SVM (Support Vector Machine)
# ------------------------------------------------------------

# SVM models are sensitive to the scale of each feature. StandardScaler normalises the features
# so that each has zero mean and unit variance.
feature_scaler = StandardScaler()

# Fits the scaler on the training features and transforms both training and testing features.
scaled_training_feature_matrix = feature_scaler.fit_transform(training_feature_matrix)
scaled_testing_feature_matrix = feature_scaler.transform(testing_feature_matrix)

# LinearSVC implements a linear Support Vector Machine suitable for large feature spaces.
linear_svm_classifier = LinearSVC(
    random_state=42,
    max_iter=10000,  # increases iteration limit to reduce convergence warnings
)

# Trains the linear SVM classifier on the scaled training feature matrix.
linear_svm_classifier.fit(scaled_training_feature_matrix, training_labels)

# Generates predictions for the scaled testing feature matrix.
linear_svm_predictions = linear_svm_classifier.predict(scaled_testing_feature_matrix)

# Evaluates the linear SVM model and stores its metrics.
linear_svm_metrics = evaluate_classification_model(
    true_labels=testing_labels,
    predicted_labels=linear_svm_predictions,
    model_name="Linear SVM",
)

# ------------------------------------------------------------
# 5. Summary comparison table
# ------------------------------------------------------------

# Combines the evaluation metrics from both models into a single comparison dataframe.
model_comparison_dataframe = pd.DataFrame(
    [random_forest_metrics, linear_svm_metrics]
)

print("=== Model Comparison Summary (Test Set) ===")
print(
    model_comparison_dataframe[
        ["model", "accuracy", "precision", "recall", "f1_score"]
    ]
)
print()

print("=== FortiSense – ML model training and evaluation completed ===")
