"""
FortiSense Compare

Aggregates evaluation metrics across models and identifies the best-performing classifier by F1-score.
"""


import pandas as pd

from fortisense_ml import rf_metrics, svm_metrics
from fortisense_nn import nn_metrics


COLUMNS = ["model", "accuracy", "precision", "recall", "f1_score"]


def build_table() -> pd.DataFrame:
    rows = [rf_metrics, svm_metrics, nn_metrics]
    df = pd.DataFrame(rows)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing metric fields: {missing}")

    return df[COLUMNS].copy()


def pick_best(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("f1_score", ascending=False).iloc[0]


def main() -> int:
    print("[*] FortiSense: comparing model metrics")

    try:
        df = build_table()
    except Exception as exc:
        print(f"[!] Compare error: {exc}")
        return 1

    print()
    print("=== FortiSense Model Comparison (Test Set) ===")
    print(df)

    best = pick_best(df)
    print()
    print("=== Best Model (by F1-score) ===")
    print(best)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
