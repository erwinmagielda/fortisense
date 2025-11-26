"""
FortiSense – Part 5: Real-Time IDS Server
Listens for incoming network data and predicts normal/attack in real-time.
"""

import socket
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
with open("../models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler used during ML training
with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

HOST = "127.0.0.1"
PORT = 5050

# Create TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[*] FortiSense IDS Server running on {HOST}:{PORT}")

while True:
    client, addr = server_socket.accept()
    print(f"[+] Connection established: {addr}")

    while True:
        data = client.recv(4096)

        if not data:
            print("[-] Client disconnected.")
            break

        # Unpickle the incoming feature row
        received_row = pickle.loads(data)

        # Convert to DataFrame
        df = pd.DataFrame([received_row])

        # Scale using the same scaler
        X_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(X_scaled)[0]
        label = "normal" if prediction == 0 else "attack"

        print(f"[Prediction] {label}")

        client.send(label.encode())

    client.close()
