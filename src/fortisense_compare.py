import pandas as pd

from fortisense_ml import rf_metrics, svm_metrics
from fortisense_nn import nn_metrics

# ============================================================
# FortiSense - Part IV: Model Comparison and Analysis
#
# This script assumes:
#   ☠ fortisense_ml.py has been run at least once
#   ☠ fortisense_nn.py has been run at least once
#
# It imports the metric dictionaries exposed by those modules
# and prints a single comparison table for the report.
# ============================================================

def main():
    print("[*] FortiSense - Collecting model metrics for comparison...")

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

    best_model_row = comparison_dataframe.sort_values(
        "f1_score", ascending=False
    ).iloc[0]

    print("\n=== Best Performing Model (by F1 score) ===")
    print(best_model_row)
    print()

    print("[✓] FortiSense - Model comparison completed")


if __name__ == "__main__":
    main()
