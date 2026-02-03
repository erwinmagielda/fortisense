"""
FortiSense IDS Server

Loads a trained model and exposes a simple TCP interface for real-time intrusion classification in a lab environment.
"""


import os
import pickle
import socket
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import pandas as pd


RECV_BYTES = 4096


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5050
    backlog: int = 1


def resolve_model_paths() -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(root_dir, "models")

    rf_path = os.path.join(model_dir, "fortisense_random_forest.pkl")
    scaler_path = os.path.join(model_dir, "fortisense_feature_scaler.pkl")
    cols_path = os.path.join(model_dir, "fortisense_feature_columns.pkl")
    return rf_path, scaler_path, cols_path


def load_artifacts(rf_path: str, scaler_path: str, cols_path: str):
    for p in (rf_path, scaler_path, cols_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing artefact: {p}")

    model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    cols: List[str] = joblib.load(cols_path)

    if not cols:
        raise RuntimeError("Feature column list is empty")

    return model, scaler, cols


def predict_one(model, scaler, cols: List[str], row: dict) -> str:
    df = pd.DataFrame([row])
    df = df[cols] 
    x = scaler.transform(df.values)
    y = int(model.predict(x)[0])
    return "normal" if y == 0 else "attack"


def main() -> int:
    cfg = ServerConfig()
    rf_path, scaler_path, cols_path = resolve_model_paths()

    print("[*] FortiSense server starting")
    try:
        model, scaler, cols = load_artifacts(rf_path, scaler_path, cols_path)
    except Exception as exc:
        print(f"[!] Server error: {exc}")
        return 1

    print(f"[+] Loaded model:  {os.path.basename(rf_path)}")
    print(f"[+] Loaded scaler: {os.path.basename(scaler_path)}")
    print(f"[+] Loaded cols:   {os.path.basename(cols_path)}")
    print()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((cfg.host, cfg.port))
        s.listen(cfg.backlog)

        print(f"[*] Listening on {cfg.host}:{cfg.port}")
        print("[*] Waiting for client...")
        print()

        while True:
            client, addr = s.accept()
            print(f"[+] Client connected: {addr}")

            with client:
                try:
                    while True:
                        raw = client.recv(RECV_BYTES)
                        if not raw:
                            print("[-] Client disconnected")
                            break

                        row = pickle.loads(raw)
                        pred = predict_one(model, scaler, cols, row)
                        client.send(pred.encode())
                        print(f"[+] Prediction: {pred}")
                except Exception as exc:
                    print(f"[!] Client error: {exc}")

            print("[*] Ready for next client")
            print()


if __name__ == "__main__":
    raise SystemExit(main())
