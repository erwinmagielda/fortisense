import os
import socket
import pickle

import joblib
import pandas as pd

# ============================================================
# FortiSense - Part V: Real-Time IDS Server
#
# This component exposes a TCP-based inference API
# that receives one feature row at a time and returns a binary
# prediction:
#       "normal"  or  "attack"
#
# High-level workflow:
#   1) Load trained Random Forest model + StandardScaler
#   2) Load feature column ordering to ensure correct alignment
#   3) Bind to a TCP socket and wait for incoming connections
#   4) For each received (pickled) feature dictionary:
#         - Convert to DataFrame
#         - Reorder columns to match training layout
#         - Apply saved scaler
#         - Predict using Random Forest
#         - Send textual result back over the socket
#
# Notes:
#   - This is a controlled, lab-only prototype. It uses pickle
#     and no message framing; in production environments neither
#     would be acceptable due to security and protocol risks.
#   - The server handles one client connection at a time for
#     demonstration simplicity.
# ============================================================

# ------------------------------------------------------------
# Resolve model paths
# ------------------------------------------------------------
project_root_directory = os.path.dirname(os.path.dirname(__file__))
model_directory = os.path.join(project_root_directory, "models")

random_forest_model_path = os.path.join(model_directory, "fortisense_random_forest.pkl")
feature_scaler_path = os.path.join(model_directory, "fortisense_feature_scaler.pkl")
feature_columns_path = os.path.join(model_directory, "fortisense_feature_columns.pkl")

print("[*] FortiSense IDS Server - Loading model and scaler...")

# Load Random Forest classifier, StandardScaler and the exact
# feature ordering used during training. Misaligned columns will
# break prediction, so enforcing consistent order is essential.
random_forest_classifier = joblib.load(random_forest_model_path)
feature_scaler = joblib.load(feature_scaler_path)
feature_column_names = joblib.load(feature_columns_path)

print("[+] Random Forest model loaded.")
print("[+] Feature scaler loaded.")
print("[+] Feature column list loaded.\n")

# ------------------------------------------------------------
# Configure network server
# ------------------------------------------------------------
# The server listens on localhost for simple demonstration
# purposes. Remote execution would require additional security
# layers (TLS, authentication, message framing, etc.).
HOST = "127.0.0.1"
PORT = 5050

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[*] FortiSense IDS Server running on {HOST}:{PORT}")
print("[*] Waiting for client connections...\n")

# ============================================================
# Main server loop
# Accepts one client at a time and processes a stream of
# feature rows until the client disconnects.
# ============================================================
while True:
    client_socket, client_address = server_socket.accept()
    print(f"[+] Client connected from {client_address}")

    try:
        while True:
            # ------------------------------------------------------------
            # Receive one pickled feature-row dictionary
            # ------------------------------------------------------------
            raw_data = client_socket.recv(4096)

            # An empty recv() means the client closed the connection
            if not raw_data:
                print("[-] Client disconnected.")
                break

            # WARNING: pickle is unsafe with untrusted inputs.
            # This server assumes a trusted environment (lab only).
            received_row = pickle.loads(raw_data)

            # ------------------------------------------------------------
            # Convert to DataFrame and enforce training-time column order
            # ------------------------------------------------------------
            # A list-of-dict inside DataFrame ensures predictable structure.
            # Selecting training_col_list ensures no missing/misordered fields.
            feature_dataframe = pd.DataFrame([received_row])[feature_column_names]

            # Apply the exact same scaling parameters used during training
            scaled_features = feature_scaler.transform(feature_dataframe.values)

            # ------------------------------------------------------------
            # Run the classifier
            # ------------------------------------------------------------
            predicted_label = random_forest_classifier.predict(scaled_features)[0]
            predicted_text = "normal" if predicted_label == 0 else "attack"

            print(f"[Prediction] {predicted_text}")

            # ------------------------------------------------------------
            # Send prediction result back to the client
            # ------------------------------------------------------------
            client_socket.send(predicted_text.encode())

    except Exception as exc:
        # Catch and log all unexpected errors without terminating the server
        print(f"[!] Error while handling client {client_address}: {exc}")

    finally:
        # Ensure socket closure even after exceptions
        client_socket.close()
        print("[*] Connection closed.")
        print("[*] Waiting for next client...\n")
