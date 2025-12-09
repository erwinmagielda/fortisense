import os

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# FortiSense - Part III: Neural Network Model
#
# This script trains a fully connected feedforward neural
# network (MLP) on the NSL-KDD dataset for binary classification:
#   label = 0 -> normal
#   label = 1 -> attack
#
# High level workflow:
#   1) Load KDD training and testing datasets
#   2) Standardise input features
#   3) Define MLP architecture in PyTorch
#   4) Train on CPU or GPU (depending on availability)
#   5) Evaluate on test set using:
#        - accuracy
#        - precision
#        - recall
#        - F1 score
#   6) Save trained model parameters to disk
#   7) Expose metric dictionary for cross model comparison
#
# The resulting metrics are used in Part IV to compare the
# neural network against the classical ML baselines.
# ============================================================

# Select CUDA if available, otherwise fall back to CPU.
# This makes the script portable between machines with and
# without a GPU, without changing any training code.
computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] FortiSense NN - Initialising on device: {computation_device}")

# ------------------------------------------------------------
# Resolve project structure and create model directory
# ------------------------------------------------------------
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

# Feature columns are all attributes except the label and the
# human readable attack_type string.
feature_column_names = [
    col for col in training_dataframe.columns
    if col not in ("label", "attack_type")
]

training_features = training_dataframe[feature_column_names].values
testing_features = testing_dataframe[feature_column_names].values

# Binary labels
training_labels = training_dataframe["label"].values
testing_labels = testing_dataframe["label"].values

print(f"[+] Total features:   {len(feature_column_names)}")
print(f"[+] Training samples: {training_features.shape[0]}")
print(f"[+] Testing samples:  {testing_features.shape[0]}")
print()

print("[*] Applying feature scaling for neural network input...")

# StandardScaler normalises each feature dimension to zero mean
# and unit variance. This significantly stabilises neural network
# training compared with raw feature values.
feature_scaler = StandardScaler()
scaled_training_features = feature_scaler.fit_transform(training_features)
scaled_testing_features = feature_scaler.transform(testing_features)

# Convert scaled NumPy arrays to PyTorch tensors and move them
# to the selected computation device (CPU or GPU).
training_feature_tensor = torch.tensor(
    scaled_training_features, dtype=torch.float32
).to(computation_device)
testing_feature_tensor = torch.tensor(
    scaled_testing_features, dtype=torch.float32
).to(computation_device)

# Labels are stored as integer class indices for CrossEntropyLoss
training_label_tensor = torch.tensor(training_labels, dtype=torch.long).to(computation_device)
testing_label_tensor = torch.tensor(testing_labels, dtype=torch.long).to(computation_device)

# ------------------------------------------------------------
# 2. Define NN architecture
# ------------------------------------------------------------

class FortiSenseNeuralNetwork(nn.Module):
    """
    Fully connected feedforward neural network (MLP) for binary
    classification of network connections (normal vs attack).

    Architecture:
        input_dim -> 64 -> 32 -> 2 logits

    Activation:
        ReLU on the hidden layers. The final layer returns raw
        logits which are passed directly to CrossEntropyLoss.
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


# Instantiate the network with an input dimension that matches
# the number of features in the NSL KDD dataset.
neural_network_model = FortiSenseNeuralNetwork(
    input_feature_count=len(feature_column_names)
).to(computation_device)

# ------------------------------------------------------------
# 3. Configure training
# ------------------------------------------------------------

# CrossEntropyLoss:
#   - applies softmax internally
#   - expects integer class labels
loss_function = nn.CrossEntropyLoss()

# Adam is a standard adaptive optimiser that usually converges
# faster than vanilla SGD for this type of problem.
model_optimizer = optim.Adam(neural_network_model.parameters(), lr=0.001)

# Number of passes over the full training set.
# Here we train in full batch mode for simplicity. In a production
# IDS setting mini batches would be preferred.
training_epochs = 10

print("[*] Starting neural network training...")
print(f"[+] Epochs: {training_epochs}")
print()

# ------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------

neural_network_model.train()

for epoch_index in range(training_epochs):
    # Reset gradients from the previous step
    model_optimizer.zero_grad()

    # Forward pass on the entire training set
    output_logits = neural_network_model(training_feature_tensor)

    # Compute supervised loss against ground truth labels
    training_loss = loss_function(output_logits, training_label_tensor)

    # Backpropagate gradients through the network
    training_loss.backward()

    # Update parameters based on computed gradients
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
    # Forward pass on the test set without tracking gradients
    testing_output_logits = neural_network_model(testing_feature_tensor)

    # argmax over the class dimension returns predicted class index
    predicted_label_tensor = torch.argmax(testing_output_logits, dim=1)

# Move predictions back to CPU and convert to NumPy for use
# with scikit-learn metric functions.
predicted_labels = predicted_label_tensor.cpu().numpy()
true_labels = testing_labels

# Standard binary classification metrics
accuracy_value = accuracy_score(true_labels, predicted_labels)
precision_value = precision_score(true_labels, predicted_labels)
recall_value = recall_score(true_labels, predicted_labels)
f1_value = f1_score(true_labels, predicted_labels)

print("=== Neural Network Evaluation Results ===")
print(f"Accuracy : {accuracy_value:.4f}")
print(f"Precision: {precision_value:.4f}")
print(f"Recall   : {recall_value:.4f}")
print(f"F1-score : {f1_value:.4f}")
print()

# Expose metrics for Part IV comparison. Other modules can
# import fortisense_nn.py and directly use nn_metrics.
nn_metrics = {
    "model": "Neural Network",
    "accuracy": accuracy_value,
    "precision": precision_value,
    "recall": recall_value,
    "f1_score": f1_value,
}

# ------------------------------------------------------------
# 6. Save trained model
# ------------------------------------------------------------

# Only the state_dict (weights and biases) is saved here.
# The architecture itself is defined in FortiSenseNeuralNetwork
# and reconstructed at inference time.
torch.save(neural_network_model.state_dict(), model_output_path)

print(f"[✓] Neural network model saved to: {model_output_path}")
print("[✓] FortiSense NN - Training and evaluation completed")
