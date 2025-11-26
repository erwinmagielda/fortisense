import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# FortiSense – Part 1: Exploratory Data Analysis (EDA)
#
# Performs the exact EDA steps required by the coursework:
#   - Dataset shapes
#   - Summary statistics
#   - Label distribution (normal vs attack)
#   - Bar chart: normal vs attack (train + test)
#   - Correlation heatmap (numeric features)
#   - Attack-type distribution
#
# Structure and commenting style is consistent with Part 2.
# ============================================================

sns.set_theme(style="whitegrid")

project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

training_dataset_path = os.path.join(dataset_directory, "KDDTrain.csv")
testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

training_dataframe = pd.read_csv(training_dataset_path)
testing_dataframe = pd.read_csv(testing_dataset_path)

print("=== Dataset Shapes ===")
print("Training dataset shape:", training_dataframe.shape)
print("Testing dataset shape: ", testing_dataframe.shape)
print()

# ------------------------------------------------------------
# 2. Summary statistics (numeric features)
# ------------------------------------------------------------

numeric_training_columns = training_dataframe.select_dtypes(include=["int64", "float64"])

print("=== Summary Statistics (Training Set) ===")
print(numeric_training_columns.describe())
print()

# ------------------------------------------------------------
# 3. Percentage distribution of normal vs attack
# ------------------------------------------------------------

label_counts = training_dataframe["label"].value_counts().sort_index()
label_percentages = (label_counts / len(training_dataframe)) * 100
label_percentages.name = "percentage"

print("=== Percentage Distribution of Normal vs Attack (Training Set) ===")
print(label_percentages)
print()

# ------------------------------------------------------------
# 4. Normal vs Attack bar chart for train + test
# ------------------------------------------------------------

train_normal = label_counts.get(0, 0)
train_attack = label_counts.get(1, 0)

test_counts = testing_dataframe["label"].value_counts().sort_index()
test_normal = test_counts.get(0, 0)
test_attack = test_counts.get(1, 0)

bar_chart_dataframe = pd.DataFrame({
    "Dataset": ["Train", "Train", "Test", "Test"],
    "Label": ["Normal", "Attack", "Normal", "Attack"],
    "Count": [train_normal, train_attack, test_normal, test_attack]
})

plt.figure()
sns.barplot(
    data=bar_chart_dataframe,
    x="Dataset",
    y="Count",
    hue="Label"
)
plt.title("Normal vs Attack Distribution – Training and Testing Sets")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Correlation heatmap (numeric features)
# ------------------------------------------------------------

correlation_matrix = numeric_training_columns.corr()

plt.figure()
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    linewidths=0.3
)
plt.title("Correlation Heatmap – Numeric Features (Training Set)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Attack-type distribution (all rows)
# ------------------------------------------------------------

attack_type_counts = training_dataframe["attack_type"].value_counts()
attack_type_percentages = (attack_type_counts / len(training_dataframe)) * 100

print("=== Attack-Type Distribution (Counts) ===")
print(attack_type_counts)
print()

print("=== Attack-Type Distribution (Percentages) ===")
print(attack_type_percentages)
print()

attack_type_dataframe = attack_type_counts.reset_index()
attack_type_dataframe.columns = ["attack_type", "count"]

plt.figure(figsize=(12, 6))
sns.barplot(
    data=attack_type_dataframe,
    x="attack_type",
    y="count",
    order=attack_type_dataframe.sort_values("count", ascending=False)["attack_type"]
)
plt.xticks(rotation=90)
plt.title("Attack-Type Distribution – Training Dataset")
plt.tight_layout()
plt.show()

print("=== FortiSense – EDA completed ===")
