"""
FortiSense NN

Trains a small neural network baseline for binary intrusion detection and exports metrics for cross-model comparison.
"""


import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


nn_metrics: Dict[str, Any] | None = None


@dataclass(frozen=True)
class Paths:
    root_dir: str
    data_dir: str
    models_dir: str
    train_csv: str
    test_csv: str
    nn_state: str
    nn_scaler: str
    nn_feature_cols: str
    metrics_json: str


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    hidden1: int = 64
    hidden2: int = 32
    seed: int = 42


def resolve_paths() -> Paths:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(root_dir, "data")
    models_dir = os.path.join(root_dir, "models")

    return Paths(
        root_dir=root_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        train_csv=os.path.join(data_dir, "KDDTrain.csv"),
        test_csv=os.path.join(data_dir, "KDDTest.csv"),
        nn_state=os.path.join(models_dir, "fortisense_nn.pth"),
        nn_scaler=os.path.join(models_dir, "fortisense_nn_scaler.pkl"),
        nn_feature_cols=os.path.join(models_dir, "fortisense_nn_feature_columns.pkl"),
        metrics_json=os.path.join(models_dir, "fortisense_metrics_nn.json"),
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)

    required = {"label", "attack_type"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {sorted(missing)}")

    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[List[str], Any, Any]:
    cols = [c for c in df.columns if c not in ("label", "attack_type")]
    x = df[cols].values
    y = df["label"].values
    return cols, x, y


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "model": "Neural Network",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2)


def load_saved_metrics(metrics_json: str) -> Dict[str, Any] | None:
    if not os.path.isfile(metrics_json):
        return None
    try:
        with open(metrics_json, "r", encoding="utf-8") as h:
            obj = json.load(h)
        return obj.get("nn_metrics")
    except Exception:
        return None


class FortiSenseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x): 
        return self.net(x)


def main() -> int:
    global nn_metrics

    cfg = TrainConfig()
    set_seed(cfg.seed)

    p = resolve_paths()
    os.makedirs(p.models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[*] FortiSense NN: loading datasets")
    print(f"[*] Device: {device}")
    try:
        train_df = load_dataset(p.train_csv)
        test_df = load_dataset(p.test_csv)
    except Exception as exc:
        print(f"[!] NN error: {exc}")
        return 1

    feature_cols, x_train, y_train = split_features_labels(train_df)
    _, x_test, y_test = split_features_labels(test_df)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    x_train_t = torch.tensor(x_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    x_test_t = torch.tensor(x_test_s, dtype=torch.float32, device=device)

    model = FortiSenseMLP(len(feature_cols), cfg.hidden1, cfg.hidden2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"[+] Train: rows={len(train_df)} features={len(feature_cols)} epochs={cfg.epochs}")
    print()

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad()
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch:02d}/{cfg.epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x_test_t), dim=1).detach().cpu().numpy()

    nn_metrics = compute_metrics(y_test, pred)

    print()
    print("=== FortiSense NN Summary ===")
    print(
        f"accuracy={nn_metrics['accuracy']:.4f} "
        f"precision={nn_metrics['precision']:.4f} "
        f"recall={nn_metrics['recall']:.4f} "
        f"f1={nn_metrics['f1_score']:.4f}"
    )
    print()

    print("[*] Saving artefacts")
    torch.save(model.state_dict(), p.nn_state)


    import joblib

    joblib.dump(scaler, p.nn_scaler)
    joblib.dump(feature_cols, p.nn_feature_cols)

    save_json(p.metrics_json, {"nn_metrics": nn_metrics})

    print(f"[+] Saved: {p.nn_state}")
    print(f"[+] Saved: {p.nn_scaler}")
    print(f"[+] Saved: {p.nn_feature_cols}")
    print(f"[+] Saved: {p.metrics_json}")
    print()

    return 0


if __name__ != "__main__":
    _p = resolve_paths()
    nn_metrics = load_saved_metrics(_p.metrics_json)

if __name__ == "__main__":
    raise SystemExit(main())
