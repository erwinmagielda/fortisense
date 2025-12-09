import os
import socket
import pickle

import joblib
import pandas as pd

# ============================================================
# FortiSense - Part V: Real-Time IDS Server
#
# 1. Loads the trained Random Forest model and feature scaler.
# 2. Listens on a TCP socket for incoming feature rows.
# 3. Each row is treated as one network connection.
# 4. Predicts "normal" or "attack" and sends the result back.
# ============================================================

project_root_directory = os.path.dirname(os.path.dirname(__file__))
model_directory = os.path.join(project_root_directory, "models")

random_forest_model_path = os.path.join(model_directory, "fortisense_random_forest.pkl")
feature_scaler_path = os.path.join(model_directory, "fortisense_feature_scaler.pkl")
feature_columns_path = os.path.join(model_directory, "fortisense_feature_columns.pkl")

print("[*] FortiSense IDS Server - Loading model and scaler...")

random_forest_classifier = joblib.load(random_forest_model_path)
feature_scaler = joblib.load(feature_scaler_path)
feature_column_names = joblib.load(feature_columns_path)

print("[+] Random Forest model loaded.")
print("[+] Feature scaler loaded.")
print("[+] Feature column list loaded.")
print()

HOST = "127.0.0.1"
PORT = 5050

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[*] FortiSense IDS Server running on {HOST}:{PORT}")
print("[*] Waiting for client connections...")
print()

while True:
    client_socket, client_address = server_socket.accept()
    print(f"[+] Client connected from {client_address}")

    try:
        while True:
            raw_data = client_socket.recv(4096)
            if not raw_data:
                print("[-] Client disconnected.")
                break

            # Decode incoming sample
            received_row = pickle.loads(raw_data)

            # Convert to DataFrame and enforce feature column order
            feature_dataframe = pd.DataFrame([received_row])[feature_column_names]

            # Scale using the same scaler as training
            scaled_features = feature_scaler.transform(feature_dataframe.values)

            # Predict using Random Forest
            predicted_label = random_forest_classifier.predict(scaled_features)[0]
            predicted_text = "normal" if predicted_label == 0 else "attack"

            print(f"[Prediction] {predicted_text}")

            # Send response back to client
            client_socket.send(predicted_text.encode())

    except Exception as exc:
        print(f"[!] Error while handling client {client_address}: {exc}")

    finally:
        client_socket.close()
        print("[*] Connection closed.")
        print("[*] Waiting for next client...\n")
