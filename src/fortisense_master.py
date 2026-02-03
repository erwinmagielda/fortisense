"""
FortiSense Master

Orchestrates the full IDS pipeline and provides a single interactive entry point for demos and evaluation.
"""


import os
import subprocess
import sys
from typing import Dict, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PYTHON_EXE = sys.executable


STAGES: Dict[str, Tuple[str, str]] = {
    "1": ("EDA", os.path.join(SCRIPT_DIR, "fortisense_eda.py")),
    "2": ("Classical ML (RF + Linear SVM)", os.path.join(SCRIPT_DIR, "fortisense_ml.py")),
    "3": ("Neural Network (PyTorch)", os.path.join(SCRIPT_DIR, "fortisense_nn.py")),
    "4": ("Model Comparison", os.path.join(SCRIPT_DIR, "fortisense_compare.py")),
    "5": ("IDS Server", os.path.join(SCRIPT_DIR, "fortisense_server.py")),
}


def run_stage(name: str, path: str) -> int:
    if not os.path.isfile(path):
        print(f"[!] Missing stage file: {path}")
        return 1

    print()
    print(f"[*] Running: {name}")
    print("=" * 60)

    try:
        completed = subprocess.run([PYTHON_EXE, path], cwd=ROOT_DIR, check=False)
        rc = int(completed.returncode or 0)
    except KeyboardInterrupt:
        print("\n[!] Cancelled")
        rc = 130
    except Exception as exc:
        print(f"[!] Failed to launch stage: {exc}")
        rc = 1

    print("=" * 60)
    print(f"[+] Finished: {name} (exit code: {rc})")
    print()
    return rc


def print_menu() -> None:
    print("=" * 46)
    print("                 FortiSense")
    print("=" * 46)
    print("1) EDA")
    print("2) Train ML (RF + Linear SVM)")
    print("3) Train NN (PyTorch)")
    print("4) Compare Models")
    print("5) Start IDS Server")
    print("6) Exit")
    print()


def main() -> int:
    while True:
        print_menu()
        try:
            choice = input("Select an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting FortiSense...")
            return 0

        if choice == "6":
            print("Exiting FortiSense...")
            return 0

        if choice in STAGES:
            name, path = STAGES[choice]
            run_stage(name, path)
            continue

        print("[!] Invalid selection")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
