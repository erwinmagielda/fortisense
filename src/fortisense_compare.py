"""
FortiSense – Part 4: Model Comparison Across ML and Neural Network Models
"""

import pandas as pd

# Import saved metrics from ML and NN parts
from fortisense_ml import rf_metrics, svm_metrics
from fortisense_nn import nn_metrics  # you'll expose nn_metrics in your nn module

print("\n=== FortiSense – Model Comparison ===")

# Create comparison dataframe
comparison_df = pd.DataFrame([
    {
        "model": "Random Forest",
        "accuracy": rf_metrics["accuracy"],
        "precision": rf_metrics["precision"],
        "recall": rf_metrics["recall"],
        "f1": rf_metrics["f1"],
    },
    {
        "model": "Linear SVM",
        "accuracy": svm_metrics["accuracy"],
        "precision": svm_metrics["precision"],
        "recall": svm_metrics["recall"],
        "f1": svm_metrics["f1"],
    },
    {
        "model": "Neural Network",
        "accuracy": nn_metrics["accuracy"],
        "precision": nn_metrics["precision"],
        "recall": nn_metrics["recall"],
        "f1": nn_metrics["f1"],
    }
])

print("\n=== Full Model Comparison Table ===\n")
print(comparison_df)

print("\n=== Best Model (by F1-score) ===")
best = comparison_df.sort_values("f1", ascending=False).iloc[0]
print(best)
