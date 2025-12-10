import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ============================================================
# FortiSense - Part II: Classical Machine Learning Models
#
# This script trains and evaluates two baseline ML classifiers
# on the KDD-style dataset:
#
#   1. Random Forest
#   2. Linear Support Vector Machine (LinearSVC)
#
# High level workflow:
#   1) Load training and testing datasets from disk
#   2) Split into feature matrices and label vectors
#   3) Train Random Forest on raw features
#   4) Standardise features and train Linear SVM
#   5) Evaluate both on the test set using:
#        - accuracy
#        - precision
#        - recall
#        - F1 score
#   6) Persist models, scaler and feature list for later use in:
#        - Part III (neural network comparison)
#        - Part IV (offline evaluation and reporting)
#
# The printed metrics and saved artefacts are referenced in the
# written report for quantitative comparison between models.
# ============================================================

# ------------------------------------------------------------
# Resolve project paths and ensure models directory exists
# ------------------------------------------------------------
project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")
model_directory = os.path.join(project_root_directory, "models")

# Create the models directory if it is not already present
os.makedirs(model_directory, exist_ok=True)

# Full paths to the training and testing CSV files
training_dataset_path = os.path.join(dataset_directory, "KDDTrain.csv")
testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

print("[*] FortiSense ML - Loading datasets...")

# Load the KDD-style training and testing splits
training_dataframe = pd.read_csv(training_dataset_path)
testing_dataframe = pd.read_csv(testing_dataset_path)

print(f"[+] Training dataset loaded: {training_dataframe.shape}")
print(f"[+] Testing dataset loaded : {testing_dataframe.shape}")
print()

# ------------------------------------------------------------
# 1. Separate features and labels
# ------------------------------------------------------------

print("[*] Preparing feature matrices and label vectors...")

# Explicitly exclude label and attack_type from the feature set.
#   label        -> binary target used for supervised learning
#   attack_type  -> human readable attack family, not used here
feature_column_names = [
    column_name
    for column_name in training_dataframe.columns
    if column_name not in ("label", "attack_type")
]

# .values converts DataFrame slices into NumPy arrays that
# scikit-learn estimators expect as input
training_feature_matrix = training_dataframe[feature_column_names].values
testing_feature_matrix = testing_dataframe[feature_column_names].values

# Binary label vectors (0 = normal, 1 = attack)
training_labels = training_dataframe["label"].values
testing_labels = testing_dataframe["label"].values

print(f"[+] Number of features used: {len(feature_column_names)}")
print(f"[+] Example feature columns: {feature_column_names[:5]}")
print()

# ------------------------------------------------------------
# Helper: metric computation
# ------------------------------------------------------------

def evaluate_classification_model(true_labels, predicted_labels, model_name):
    """
    Compute standard binary classification metrics for a model.

    Parameters
    ----------
    true_labels : array-like
        Ground truth labels from the test set.
    predicted_labels : array-like
        Labels predicted by the trained model.
    model_name : str
        Name or identifier of the model, used for printing and logging.

    Returns
    -------
    dict
        Dictionary with model name and computed metrics. This is useful
        for building comparison tables between multiple classifiers.
    """
    accuracy_value = accuracy_score(true_labels, predicted_labels)
    precision_value = precision_score(true_labels, predicted_labels, zero_division=0)
    recall_value = recall_score(true_labels, predicted_labels, zero_division=0)
    f1_value = f1_score(true_labels, predicted_labels, zero_division=0)

    print(f"=== Evaluation Results for {model_name} ===")
    print(f"Accuracy : {accuracy_value:.4f}")
    print(f"Precision: {precision_value:.4f}")
    print(f"Recall   : {recall_value:.4f}")
    print(f"F1-score : {f1_value:.4f}")
    print()

    return {
        "model": model_name,
        "accuracy": accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "f1_score": f1_value,
    }

# ------------------------------------------------------------
# 2. Model 1 - Random Forest
# ------------------------------------------------------------

print("[*] Training Random Forest classifier...")

# RandomForestClassifier is a strong baseline for tabular data.
#   n_estimators = 100    -> number of trees in the ensemble
#   random_state = 42     -> reproducible results
#   n_jobs = -1           -> use all available CPU cores
random_forest_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

# Train the Random Forest on the raw (unscaled) feature matrix
random_forest_classifier.fit(training_feature_matrix, training_labels)

# Run inference on the held-out test set
random_forest_predictions = random_forest_classifier.predict(testing_feature_matrix)

# Compute performance metrics on the test set
random_forest_metrics = evaluate_classification_model(
    true_labels=testing_labels,
    predicted_labels=random_forest_predictions,
    model_name="Random Forest",
)

# ------------------------------------------------------------
# 3. Model 2 - Linear SVM
# ------------------------------------------------------------

print("[*] Scaling features and training Linear SVM classifier...")

# Many linear models, including linear SVM, are sensitive to feature scale.
# StandardScaler transforms each feature to zero mean and unit variance:
#   z = (x - mean) / std
feature_scaler = StandardScaler()
scaled_training_feature_matrix = feature_scaler.fit_transform(training_feature_matrix)
scaled_testing_feature_matrix = feature_scaler.transform(testing_feature_matrix)

# LinearSVC implements a linear Support Vector Machine using a
# hinge loss and is suitable for high dimensional feature spaces.
linear_svm_classifier = LinearSVC(
    random_state=42,
    max_iter=10000,  # increased iteration limit to help convergence
)

# Train the linear SVM on the scaled training features
linear_svm_classifier.fit(scaled_training_feature_matrix, training_labels)

# Test set predictions using the same scaler
linear_svm_predictions = linear_svm_classifier.predict(scaled_testing_feature_matrix)

# Compute performance metrics for the linear SVM
linear_svm_metrics = evaluate_classification_model(
    true_labels=testing_labels,
    predicted_labels=linear_svm_predictions,
    model_name="Linear SVM",
)

# ------------------------------------------------------------
# 4. Comparison summary
# ------------------------------------------------------------

print("[*] Building comparison table for ML models...")

# Collect metrics for both models into a DataFrame that can be
# printed here and reused later in the report or other scripts
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

# ------------------------------------------------------------
# 5. Save models, scaler and feature list
# ------------------------------------------------------------

print("[*] Saving trained models, scaler and feature list to disk...")

# Paths for all persisted artefacts that will be reused by
# offline evaluation and the online IDS server component
random_forest_model_path = os.path.join(model_directory, "fortisense_random_forest.pkl")
linear_svm_model_path = os.path.join(model_directory, "fortisense_linear_svm.pkl")
feature_scaler_path = os.path.join(model_directory, "fortisense_feature_scaler.pkl")
feature_columns_path = os.path.join(model_directory, "fortisense_feature_columns.pkl")

# joblib is the recommended way to persist scikit-learn models
joblib.dump(random_forest_classifier, random_forest_model_path)
joblib.dump(linear_svm_classifier, linear_svm_model_path)
joblib.dump(feature_scaler, feature_scaler_path)
joblib.dump(feature_column_names, feature_columns_path)

print(f"[+] Random Forest model saved to : {random_forest_model_path}")
print(f"[+] Linear SVM model saved to   : {linear_svm_model_path}")
print(f"[+] Feature scaler saved to     : {feature_scaler_path}")
print(f"[+] Feature column list saved to: {feature_columns_path}")
print()

# Expose metric dictionaries for Part IV so that other modules
# can import fortisense_ml.py and directly access rf_metrics and svm_metrics
rf_metrics = random_forest_metrics
svm_metrics = linear_svm_metrics

print("[✓] FortiSense ML - Model training, evaluation and saving completed")
