import os
import socket
import pickle

import pandas as pd

# ============================================================
# FortiSense - Part V: Real-Time IDS Client
#
# This client simulates live network traffic by streaming a
# subset of KDD-style test samples to the FortiSense IDS server.
#
# High level workflow:
#   1) Load KDDTest.csv from the data directory
#   2) Separate features from labels
#   3) Randomly select 50 rows to emulate live connections
#   4) Open a TCP connection to the IDS server
#   5) For each sampled row:
#        - serialize feature dictionary with pickle
#        - send over the socket
#        - receive prediction string ("normal" or "attack")
#        - compare with ground truth label
#   6) Print per sample result and overall online accuracy
#
# Note:
#   - This uses a very simple, unframed protocol and pickle,
#     which is acceptable only in a controlled lab environment.
# ============================================================

# ------------------------------------------------------------
# Resolve project structure relative to this script
# ------------------------------------------------------------
project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

# IDS server endpoint (must match the server configuration)
server_host = "127.0.0.1"
server_port = 5050

print("[*] FortiSense IDS Client - Loading test dataset...")

# Load KDD-style test split
testing_dataframe = pd.read_csv(testing_dataset_path)

# Keep a copy of the binary labels so we can compute accuracy
true_label_series = testing_dataframe["label"]

# Only feature columns are sent to the IDS server
# label and attack_type are removed so the server receives
# strictly numeric features plus any non label attributes
feature_dataframe = testing_dataframe.drop(columns=["label", "attack_type"])

print(f"[+] Test dataset loaded with {len(feature_dataframe)} total rows.\n")

# ------------------------------------------------------------
# Select a subset to simulate real time traffic
# ------------------------------------------------------------

# 50 samples is enough to demonstrate online performance
sample_count = 50

# Randomly sample rows with a fixed seed for reproducibility
sampled_dataframe = feature_dataframe.sample(n=sample_count, random_state=42)
sampled_indices = sampled_dataframe.index

print(f"[*] Selected {sample_count} random samples for real-time testing.")
print(f"[+] Sample indices: {list(sampled_indices[:10])} ...\n")

# ------------------------------------------------------------
# Connect to the IDS server
# ------------------------------------------------------------

print(f"[*] Connecting to IDS server at {server_host}:{server_port}...")

# create_connection handles underlying socket creation and connect
with socket.create_connection((server_host, server_port)) as client_socket:
    print("[+] Connected to IDS server.")
    print("[*] Sending sampled rows as simulated network traffic...\n")

    total_samples_sent = 0
    total_correct_predictions = 0

    # --------------------------------------------------------
    # Stream each sampled row as a single request
    # --------------------------------------------------------
    for position, row_index in enumerate(sampled_indices, start=1):
        # Extract this row and convert it to a simple dictionary.
        # This structure matches what the server expects to unpickle.
        sample_row_dictionary = sampled_dataframe.loc[row_index].to_dict()

        # Serialize dictionary into bytes using pickle
        # Warning: pickle is unsafe with untrusted sources.
        # This pattern is intended for isolated lab use only.
        payload_bytes = pickle.dumps(sample_row_dictionary)

        # Send the encoded payload to the server
        client_socket.send(payload_bytes)

        # Receive prediction from server.
        # The protocol is a simple text response such as "normal" or "attack".
        prediction_text = client_socket.recv(4096).decode().strip()

        # Ground truth label for this row (0 normal, 1 attack)
        true_label_value = int(true_label_series.loc[row_index])
        true_label_text = "normal" if true_label_value == 0 else "attack"

        # Compare received prediction with the ground truth
        is_correct_prediction = (prediction_text == true_label_text)

        total_samples_sent += 1
        if is_correct_prediction:
            total_correct_predictions += 1

        correctness_flag = "CORRECT" if is_correct_prediction else "WRONG"

        # Per sample status line
        print(
            f"Sample {position:02d} (row index {row_index}) - "
            f"True: {true_label_text:7s} | Predicted: {prediction_text:7s} "
            f"-> {correctness_flag}"
        )

    # --------------------------------------------------------
    # Compute simple online accuracy for this batch
    # --------------------------------------------------------
    if total_samples_sent > 0:
        accuracy_ratio = total_correct_predictions / total_samples_sent
    else:
        accuracy_ratio = 0.0

    print("\n=== FortiSense IDS - Online Evaluation Summary ===")
    print(f"Total samples sent           : {total_samples_sent}")
    print(f"Correct predictions          : {total_correct_predictions}")
    print(f"Online accuracy (50 samples) : {accuracy_ratio:.4f}\n")

    print("[✓] Finished sending samples. Closing client connection.")
