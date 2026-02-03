"""
FortiSense EDA

Performs exploratory analysis on KDD-style datasets to validate feature distributions, label balance, and correlations.
"""


import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def resolve_paths() -> tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(root_dir, "data")
    train_path = os.path.join(data_dir, "KDDTrain.csv")
    test_path = os.path.join(data_dir, "KDDTest.csv")
    return root_dir, train_path, test_path


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def main() -> int:
    sns.set_theme(style="whitegrid")

    _, train_path, test_path = resolve_paths()

    print("[*] FortiSense EDA: loading datasets")
    try:
        train_df = load_csv(train_path)
        test_df = load_csv(test_path)
    except Exception as exc:
        print(f"[!] EDA error: {exc}")
        return 1

    print(f"[+] Training: {train_df.shape}")
    print(f"[+] Testing:  {test_df.shape}")
    print()

    numeric_train = train_df.select_dtypes(include=["int64", "float64"])
    print("[*] Summary statistics (training numeric features)")
    print(numeric_train.describe())
    print()

    if "label" not in train_df.columns:
        print("[!] Missing column: label")
        return 1

    label_counts = train_df["label"].value_counts().sort_index()
    label_pct = (label_counts / len(train_df)) * 100
    label_pct.name = "percentage"

    print("[*] Label distribution (training)")
    print(label_pct)
    print()

    test_counts = test_df["label"].value_counts().sort_index() if "label" in test_df.columns else pd.Series()

    bar_df = pd.DataFrame(
        {
            "Dataset": ["Train", "Train", "Test", "Test"],
            "Label": ["Normal", "Attack", "Normal", "Attack"],
            "Count": [
                int(label_counts.get(0, 0)),
                int(label_counts.get(1, 0)),
                int(test_counts.get(0, 0)),
                int(test_counts.get(1, 0)),
            ],
        }
    )

    plt.figure()
    sns.barplot(data=bar_df, x="Dataset", y="Count", hue="Label")
    plt.title("Normal vs Attack Distribution (Train vs Test)")
    plt.tight_layout()
    plt.show()

    if not numeric_train.empty:
        corr = numeric_train.corr()
        plt.figure()
        sns.heatmap(corr, cmap="coolwarm", linewidths=0.3)
        plt.title("Correlation Heatmap (Training Numeric Features)")
        plt.tight_layout()
        plt.show()

    if "attack_type" in train_df.columns:
        counts = train_df["attack_type"].value_counts()
        pct = (counts / len(train_df)) * 100

        print("=== Attack Type Distribution (Counts) ===")
        print(counts)
        print()
        print("=== Attack Type Distribution (Percentages) ===")
        print(pct)
        print()

        plot_df = counts.reset_index()
        plot_df.columns = ["attack_type", "count"]

        plt.figure(figsize=(12, 6))
        order = plot_df.sort_values("count", ascending=False)["attack_type"]
        sns.barplot(data=plot_df, x="attack_type", y="count", order=order)
        plt.xticks(rotation=90)
        plt.title("Attack Type Distribution (Training Set)")
        plt.tight_layout()
        plt.show()
    else:
        print("[!] Missing column: attack_type (skipping attack distribution plots)")
        print()

    print("[+] FortiSense EDA complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
