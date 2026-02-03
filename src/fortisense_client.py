"""
FortiSense Client

Simulates live network traffic by streaming feature rows to the IDS server and reporting online prediction accuracy.
"""


import os
import pickle
import socket
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5050
DEFAULT_SAMPLE_COUNT = 50
DEFAULT_RANDOM_SEED = 42
RECV_BYTES = 4096


@dataclass(frozen=True)
class ClientConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    sample_count: int = DEFAULT_SAMPLE_COUNT
    random_seed: int = DEFAULT_RANDOM_SEED


def resolve_paths() -> Tuple[str, str]:
    """Resolve project root and the default test dataset path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(root_dir, "data", "KDDTest.csv")
    return root_dir, dataset_path


def load_test_dataset(path: str) -> pd.DataFrame:
    """Load the test dataset and validate required columns."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Test dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"label", "attack_type"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {sorted(missing)}")

    return df


def build_sample(df: pd.DataFrame, sample_count: int, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Return sampled feature rows and the aligned ground truth labels."""
    labels = df["label"]
    features = df.drop(columns=["label", "attack_type"])

    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0")

    if sample_count > len(features):
        raise ValueError(f"sample_count ({sample_count}) exceeds dataset size ({len(features)})")

    sampled = features.sample(n=sample_count, random_state=seed)
    return sampled, labels


def encode_payload(row_dict: Dict) -> bytes:
    """Serialise a single sample into bytes."""
    return pickle.dumps(row_dict)


def decode_prediction(raw: bytes) -> str:
    """Decode the server response label."""
    return raw.decode(errors="replace").strip().lower()


def label_to_text(label_value: int) -> str:
    """Convert ground truth label to response tokens expected from the server."""
    return "normal" if int(label_value) == 0 else "attack"


def run_client(config: ClientConfig, dataset_path: str) -> int:
    print("[*] FortiSense client starting")
    print(f"[*] Dataset: {dataset_path}")
    print(f"[*] Target: {config.host}:{config.port}")
    print()

    df = load_test_dataset(dataset_path)
    sampled_df, true_labels = build_sample(df, config.sample_count, config.random_seed)

    indices = list(sampled_df.index)
    print(f"[+] Loaded rows: {len(df)}")
    print(f"[+] Streaming samples: {len(indices)} (seed {config.random_seed})")
    print()

    total = 0
    correct = 0

    with socket.create_connection((config.host, config.port)) as sock:
        print("[+] Connected")
        print()

        for i, idx in enumerate(indices, start=1):
            payload = encode_payload(sampled_df.loc[idx].to_dict())
            sock.send(payload)

            pred = decode_prediction(sock.recv(RECV_BYTES))
            truth = label_to_text(true_labels.loc[idx])

            ok = pred == truth
            total += 1
            correct += 1 if ok else 0

            flag = "OK" if ok else "MISS"
            print(f"{i:02d}) row={idx} truth={truth:<6} pred={pred:<6} {flag}")

    acc = (correct / total) if total else 0.0
    print()
    print("=== FortiSense Online Summary ===")
    print(f"Samples:   {total}")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {acc:.4f}")
    print()

    return 0


def main() -> int:
    _, dataset_path = resolve_paths()
    cfg = ClientConfig()
    try:
        return run_client(cfg, dataset_path)
    except Exception as exc:
        print(f"[!] Client error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
