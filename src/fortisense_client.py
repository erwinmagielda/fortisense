import os
import socket
import pickle

import pandas as pd

# ============================================================
# FortiSense - Part 5: Real-Time IDS Client
#
# - Connects to the IDS server.
# - Loads samples from KDDTest.csv.
# - Selects 50 random rows to simulate live traffic.
# - Sends feature rows one by one to the server.
# - Receives predictions and compares them with true labels.
# - Prints per-sample result and overall accuracy.
# ============================================================

# Resolve project structure relative to this script
project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

server_host = "127.0.0.1"
server_port = 5050

print("[*] FortiSense IDS Client - Loading test dataset...")

testing_dataframe = pd.read_csv(testing_dataset_path)

# Keep a copy of labels so we can verify predictions later
true_label_series = testing_dataframe["label"]

# Only feature columns are sent to the IDS server
feature_dataframe = testing_dataframe.drop(columns=["label", "attack_type"])

print(f"[+] Test dataset loaded with {len(feature_dataframe)} total rows.")
print()

# Select 50 random samples for the real-time simulation
sample_count = 50
sampled_dataframe = feature_dataframe.sample(n=sample_count, random_state=42)
sampled_indices = sampled_dataframe.index

print(f"[*] Selected {sample_count} random samples for real-time testing.")
print(f"[+] Sample indices: {list(sampled_indices[:10])} ...")
print()

print(f"[*] Connecting to IDS server at {server_host}:{server_port}...")

with socket.create_connection((server_host, server_port)) as client_socket:
    print("[+] Connected to IDS server.")
    print("[*] Sending sampled rows as simulated network traffic...\n")

    total_samples_sent = 0
    total_correct_predictions = 0

    for position, row_index in enumerate(sampled_indices, start=1):
        # Convert this sample row to a dictionary of feature_name -> value
        sample_row_dictionary = sampled_dataframe.loc[row_index].to_dict()

        # Encode sample as bytes for transmission
        payload_bytes = pickle.dumps(sample_row_dictionary)
        client_socket.send(payload_bytes)

        # Receive prediction from server
        prediction_text = client_socket.recv(4096).decode().strip()

        # Ground truth label for this row (0 - normal, 1 - attack)
        true_label_value = int(true_label_series.loc[row_index])
        true_label_text = "normal" if true_label_value == 0 else "attack"

        is_correct_prediction = (prediction_text == true_label_text)

        total_samples_sent += 1
        if is_correct_prediction:
            total_correct_predictions += 1

        correctness_flag = "CORRECT" if is_correct_prediction else "WRONG"

        print(
            f"Sample {position:02d} (row index {row_index}) - "
            f"True: {true_label_text:7s} | Predicted: {prediction_text:7s} "
            f"-> {correctness_flag}"
        )

    if total_samples_sent > 0:
        accuracy_ratio = total_correct_predictions / total_samples_sent
    else:
        accuracy_ratio = 0.0

    print("\n=== FortiSense IDS - Online Evaluation Summary ===")
    print(f"Total samples sent      : {total_samples_sent}")
    print(f"Correct predictions     : {total_correct_predictions}")
    print(f"Online accuracy (50 samples): {accuracy_ratio:.4f}")
    print()

    print("[✓] Finished sending samples. Closing client connection.")
