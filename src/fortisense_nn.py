import os

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# FortiSense - Part 3: Neural Network Model (PyTorch)
#
# Steps:
#   - Load KDD datasets
#   - Scale features
#   - Define an MLP architecture
#   - Train on CPU or GPU
#   - Evaluate accuracy, precision, recall, F1
#   - Save trained model for later inference
# ============================================================

computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] FortiSense NN - Initialising on device: {computation_device}")

project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")
model_directory = os.path.join(project_root_directory, "models")

os.makedirs(model_directory, exist_ok=True)

training_dataset_path = os.path.join(dataset_directory, "KDDTrain.csv")
testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")
model_output_path = os.path.join(model_directory, "fortisense_nn.pth")

# ------------------------------------------------------------
# 1. Load and preprocess datasets
# ------------------------------------------------------------

print("[*] Loading training and testing datasets...")

training_dataframe = pd.read_csv(training_dataset_path)
testing_dataframe = pd.read_csv(testing_dataset_path)

print(f"[+] Training dataset loaded: {training_dataframe.shape}")
print(f"[+] Testing dataset loaded : {testing_dataframe.shape}")
print()

feature_column_names = [
    col for col in training_dataframe.columns
    if col not in ("label", "attack_type")
]

training_features = training_dataframe[feature_column_names].values
testing_features = testing_dataframe[feature_column_names].values

training_labels = training_dataframe["label"].values
testing_labels = testing_dataframe["label"].values

print(f"[+] Total features:      {len(feature_column_names)}")
print(f"[+] Training samples:    {training_features.shape[0]}")
print(f"[+] Testing samples:     {testing_features.shape[0]}")
print()

print("[*] Applying feature scaling for neural network input...")

feature_scaler = StandardScaler()
scaled_training_features = feature_scaler.fit_transform(training_features)
scaled_testing_features = feature_scaler.transform(testing_features)

training_feature_tensor = torch.tensor(scaled_training_features, dtype=torch.float32).to(computation_device)
testing_feature_tensor = torch.tensor(scaled_testing_features, dtype=torch.float32).to(computation_device)

training_label_tensor = torch.tensor(training_labels, dtype=torch.long).to(computation_device)
testing_label_tensor = torch.tensor(testing_labels, dtype=torch.long).to(computation_device)

# ------------------------------------------------------------
# 2. Define neural network architecture
# ------------------------------------------------------------

class FortiSenseNeuralNetwork(nn.Module):
    """
    Fully connected feedforward neural network (MLP) for binary
    classification of network connections (normal vs attack).
    """

    def __init__(self, input_feature_count):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


neural_network_model = FortiSenseNeuralNetwork(
    input_feature_count=len(feature_column_names)
).to(computation_device)

# ------------------------------------------------------------
# 3. Configure training
# ------------------------------------------------------------

loss_function = nn.CrossEntropyLoss()
model_optimizer = optim.Adam(neural_network_model.parameters(), lr=0.001)

training_epochs = 10

print("[*] Starting neural network training...")
print(f"[+] Epochs: {training_epochs}")
print()

# ------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------

neural_network_model.train()

for epoch_index in range(training_epochs):
    model_optimizer.zero_grad()

    output_logits = neural_network_model(training_feature_tensor)
    training_loss = loss_function(output_logits, training_label_tensor)

    training_loss.backward()
    model_optimizer.step()

    print(f"Epoch {epoch_index + 1}/{training_epochs} - Loss: {training_loss.item():.4f}")

print()
print("[+] Neural network training complete.")
print()

# ------------------------------------------------------------
# 5. Evaluation on test set
# ------------------------------------------------------------

print("[*] Evaluating neural network on testing dataset...")

neural_network_model.eval()
with torch.no_grad():
    testing_output_logits = neural_network_model(testing_feature_tensor)
    predicted_label_tensor = torch.argmax(testing_output_logits, dim=1)

predicted_labels = predicted_label_tensor.cpu().numpy()
true_labels = testing_labels

accuracy_value = accuracy_score(true_labels, predicted_labels)
precision_value = precision_score(true_labels, predicted_labels)
recall_value = recall_score(true_labels, predicted_labels)
f1_value = f1_score(true_labels, predicted_labels)

print("=== Neural Network Evaluation Results ===")
print(f"Accuracy : {accuracy_value:.4f}")
print(f"Precision: {precision_value:.4f}")
print(f"Recall   : {recall_value:.4f}")
print(f"F1 score : {f1_value:.4f}")
print()

# ------------------------------------------------------------
# 6. Save trained model
# ------------------------------------------------------------

torch.save(neural_network_model.state_dict(), model_output_path)
print(f"[✓] Neural network model saved to: {model_output_path}")
print("[✓] FortiSense NN - Training and evaluation completed")
