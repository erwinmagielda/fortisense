"""
FortiSense – Part 5: Real-Time IDS Client
Sends sample network data to the IDS server for classification.
"""

import socket
import pickle
import pandas as pd

HOST = "127.0.0.1"
PORT = 5050

# Load testing dataset
df = pd.read_csv("../data/KDDTest.csv")

# Drop label columns to match training features
df_features = df.drop(columns=["label", "attack_type"])

# Create connection
client = socket.create_connection((HOST, PORT))
print("[*] Connected to FortiSense IDS Server")

# Send first 20 rows as "real-time traffic"
for i in range(20):
    row_dict = df_features.iloc[i].to_dict()

    client.send(pickle.dumps(row_dict))

    result = client.recv(4096).decode()
    print(f"Row {i+1}: Prediction → {result}")

client.close()
