import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# FortiSense - Part I: Exploratory Data Analysis (EDA)
#
# This script performs baseline exploratory analysis on the
# KDD-style training and testing datasets to understand:
#
#   1. Dataset shapes (train vs test)
#   2. Summary statistics for numeric features
#   3. Binary label distribution (normal vs attack)
#   4. Normal vs attack bar chart (train vs test)
#   5. Correlation heatmap for numeric features
#   6. Attack type distribution (multi class)
#
# The output is used to:
#   - Validate dataset integrity
#   - Identify class imbalance
#   - Spot correlated or redundant features
#   - Provide visual evidence for the written report
# ============================================================

# Use a consistent Seaborn theme for all plots
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# Resolve dataset paths relative to the project root
# ------------------------------------------------------------
# __file__ points to this script.
# dirname(dirname(__file__)) goes one level up into the project root.
project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

# Full paths to the training and testing CSV files
training_dataset_path = os.path.join(dataset_directory, "KDDTrain.csv")
testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

print("[*] FortiSense EDA - Loading datasets...")

# Load both datasets into memory
training_dataframe = pd.read_csv(training_dataset_path)
testing_dataframe = pd.read_csv(testing_dataset_path)

# Shape is (rows, columns) and is useful to verify expected sizes
print(f"[+] Training dataset loaded: {training_dataframe.shape}")
print(f"[+] Testing dataset loaded : {testing_dataframe.shape}")
print()

# ------------------------------------------------------------
# 1. Summary statistics for numeric features
# ------------------------------------------------------------

# Select only numeric columns (int64 and float64) in the training set.
# This avoids including label and other non numeric attributes in the summary.
numeric_training_columns = training_dataframe.select_dtypes(include=["int64", "float64"])

print("[*] Computing summary statistics for numeric training features...")
print("=== Summary Statistics (Training Set) ===")
# .describe() gives count, mean, std, min, quartiles, and max per feature
print(numeric_training_columns.describe())
print()

# ------------------------------------------------------------
# 2. Percentage distribution of normal vs attack (binary label)
# ------------------------------------------------------------

print("[*] Computing label distribution for normal vs attack...")

# Value counts for the binary label:
#   0 → normal
#   1 → attack
label_counts = training_dataframe["label"].value_counts().sort_index()

# Convert absolute counts to percentages for easier interpretation
label_percentages = (label_counts / len(training_dataframe)) * 100
label_percentages.name = "percentage"

print("=== Percentage Distribution of Normal vs Attack (Training Set) ===")
print(label_percentages)
print()

# ------------------------------------------------------------
# 3. Normal vs attack bar chart for train and test
# ------------------------------------------------------------

print("[*] Generating bar chart for normal vs attack distribution (train vs test)...")

# Extract binary label counts for the training set
train_normal = label_counts.get(0, 0)
train_attack = label_counts.get(1, 0)

# Repeat the same computation for the testing set
test_counts = testing_dataframe["label"].value_counts().sort_index()
test_normal = test_counts.get(0, 0)
test_attack = test_counts.get(1, 0)

# Build a small DataFrame that encodes dataset type, label, and count.
# This makes it easy to pass into Seaborn for a grouped bar plot.
bar_chart_dataframe = pd.DataFrame({
    "Dataset": ["Train", "Train", "Test", "Test"],
    "Label": ["Normal", "Attack", "Normal", "Attack"],
    "Count": [train_normal, train_attack, test_normal, test_attack]
})

# Plot side by side bars for train and test label distributions
plt.figure()
sns.barplot(
    data=bar_chart_dataframe,
    x="Dataset",
    y="Count",
    hue="Label"
)
plt.title("Normal vs Attack Distribution - Training and Testing Sets")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. Correlation heatmap for numeric features
# ------------------------------------------------------------

print("[*] Computing and plotting correlation heatmap for numeric features...")

# Pearson correlation matrix between all numeric features.
# This helps to identify highly correlated features which may be redundant.
correlation_matrix = numeric_training_columns.corr()

plt.figure()
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    linewidths=0.3
)
plt.title("Correlation Heatmap - Numeric Features (Training Set)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Attack type distribution (multi class)
# ------------------------------------------------------------

print("[*] Computing attack-type distribution for the training set...")
print()

# attack_type is a categorical field that maps label 1 to concrete attack families
attack_type_counts = training_dataframe["attack_type"].value_counts()
attack_type_percentages = (attack_type_counts / len(training_dataframe)) * 100

print("=== Attack-Type Distribution (Counts) ===")
print(attack_type_counts)
print()

print("=== Attack-Type Distribution (Percentages) ===")
print(attack_type_percentages)
print()

# Prepare a DataFrame for plotting attack type counts
attack_type_dataframe = attack_type_counts.reset_index()
attack_type_dataframe.columns = ["attack_type", "count"]

# Bar plot of attack types sorted by frequency
plt.figure(figsize=(12, 6))
sns.barplot(
    data=attack_type_dataframe,
    x="attack_type",
    y="count",
    order=attack_type_dataframe.sort_values("count", ascending=False)["attack_type"]
)
plt.xticks(rotation=90)
plt.title("Attack-Type Distribution - Training Dataset")
plt.tight_layout()
plt.show()

print("[✓] FortiSense EDA - Completed")
