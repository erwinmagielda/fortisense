import pandas as pd

from fortisense_ml import rf_metrics, svm_metrics
from fortisense_nn import nn_metrics

# ============================================================
# FortiSense - Part IV: Model Comparison
#
# This module aggregates the evaluation metrics produced by:
#   - fortisense_ml.py    (Random Forest + Linear SVM)
#   - fortisense_nn.py    (Neural Network)
#
# Both of those scripts must have been executed at least once,
# because they expose metric dictionaries (rf_metrics,
# svm_metrics, nn_metrics) at module level.
#
# Purpose:
#   - Build a unified comparison table for the written report
#   - Identify the best-performing model using F1-score
#   - Provide a simple interface for external analysis tools
#
# Output:
#   - Printed DataFrame of accuracy, precision, recall, F1
#   - Clear highlight of the top performer
# ============================================================

def main():
    print("[*] FortiSense - Collecting model metrics for comparison...")

    # Build a DataFrame from metric dictionaries exposed by the
    # training modules. Each dictionary contains:
    #   model, accuracy, precision, recall, f1_score
    comparison_dataframe = pd.DataFrame(
        [
            rf_metrics,
            svm_metrics,
            nn_metrics,
        ]
    )

    print("\n=== FortiSense - Model Comparison (Test Set) ===\n")
    print(
        comparison_dataframe[
            ["model", "accuracy", "precision", "recall", "f1_score"]
        ]
    )

    # F1-score is used as the tie-breaker and selection criterion
    # because it balances precision and recall in imbalanced datasets.
    best_model_row = comparison_dataframe.sort_values(
        "f1_score", ascending=False
    ).iloc[0]

    print("\n=== Best Performing Model (by F1-score) ===")
    print(best_model_row)
    print()

    print("[✓] FortiSense - Model comparison completed")


if __name__ == "__main__":
    main()
