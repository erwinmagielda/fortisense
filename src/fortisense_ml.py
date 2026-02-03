"""
FortiSense ML

Trains and evaluates classical ML models (Random Forest, Linear SVM) and persists artefacts for offline comparison and live IDS use.
"""


import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Exported for fortisense_compare.py compatibility
rf_metrics: Dict[str, Any] | None = None
svm_metrics: Dict[str, Any] | None = None


@dataclass(frozen=True)
class Paths:
    root_dir: str
    data_dir: str
    models_dir: str
    train_csv: str
    test_csv: str
    rf_model: str
    svm_model: str
    scaler: str
    feature_cols: str
    metrics_json: str


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
        rf_model=os.path.join(models_dir, "fortisense_random_forest.pkl"),
        svm_model=os.path.join(models_dir, "fortisense_linear_svm.pkl"),
        scaler=os.path.join(models_dir, "fortisense_feature_scaler.pkl"),
        feature_cols=os.path.join(models_dir, "fortisense_feature_columns.pkl"),
        metrics_json=os.path.join(models_dir, "fortisense_metrics_ml.json"),
    )


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


def compute_metrics(y_true, y_pred, model_name: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2)


def load_saved_metrics(metrics_json: str) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    if not os.path.isfile(metrics_json):
        return None, None
    try:
        with open(metrics_json, "r", encoding="utf-8") as h:
            obj = json.load(h)
        return obj.get("rf_metrics"), obj.get("svm_metrics")
    except Exception:
        return None, None


def main() -> int:
    global rf_metrics, svm_metrics

    p = resolve_paths()
    os.makedirs(p.models_dir, exist_ok=True)

    print("[*] FortiSense ML: loading datasets")
    try:
        train_df = load_dataset(p.train_csv)
        test_df = load_dataset(p.test_csv)
    except Exception as exc:
        print(f"[!] ML error: {exc}")
        return 1

    feature_cols, x_train, y_train = split_features_labels(train_df)
    _, x_test, y_test = split_features_labels(test_df)

    print(f"[+] Train: rows={len(train_df)} features={len(feature_cols)}")
    print(f"[+] Test:  rows={len(test_df)}")
    print()

    print("[*] Training Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_metrics = compute_metrics(y_test, rf_pred, "Random Forest")
    print(f"[+] RF accuracy={rf_metrics['accuracy']:.4f} f1={rf_metrics['f1_score']:.4f}")

    print("[*] Training Linear SVM (scaled features)")
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    svm = LinearSVC(random_state=42, max_iter=10000)
    svm.fit(x_train_s, y_train)
    svm_pred = svm.predict(x_test_s)
    svm_metrics = compute_metrics(y_test, svm_pred, "Linear SVM")
    print(f"[+] SVM accuracy={svm_metrics['accuracy']:.4f} f1={svm_metrics['f1_score']:.4f}")

    print()
    print("=== FortiSense ML Summary ===")
    df = pd.DataFrame([rf_metrics, svm_metrics])[["model", "accuracy", "precision", "recall", "f1_score"]]
    print(df)
    print()

    print("[*] Saving artefacts")
    joblib.dump(rf, p.rf_model)
    joblib.dump(svm, p.svm_model)
    joblib.dump(scaler, p.scaler)
    joblib.dump(feature_cols, p.feature_cols)

    save_json(
        p.metrics_json,
        {
            "rf_metrics": rf_metrics,
            "svm_metrics": svm_metrics,
        },
    )

    print(f"[+] Saved: {p.rf_model}")
    print(f"[+] Saved: {p.svm_model}")
    print(f"[+] Saved: {p.scaler}")
    print(f"[+] Saved: {p.feature_cols}")
    print(f"[+] Saved: {p.metrics_json}")
    print()

    return 0


if __name__ != "__main__":
    _p = resolve_paths()
    _rf, _svm = load_saved_metrics(_p.metrics_json)
    rf_metrics = _rf
    svm_metrics = _svm

if __name__ == "__main__":
    raise SystemExit(main())
