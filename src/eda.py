import pandas as pd

train_path = "../data/KDDTrain.csv"
test_path = "../data/KDDTest.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("\nTraining columns:", df_train.columns.tolist())
print("\nHead of train:")
print(df_train.head())
