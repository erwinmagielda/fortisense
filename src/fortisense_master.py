import os
import sys
import subprocess

# ============================================================
# FortiSense - Master Orchestrator
#
# This script drives the entire FortiSense pipeline end-to-end:
#
#   Part I   → fortisense_eda.py
#              Exploratory Data Analysis on a KDD-style set
#
#   Part II  → fortisense_ml.py
#              Classical ML models (Random Forest + Linear SVM)
#
#   Part III → fortisense_nn.py
#              Neural network model in PyTorch
#
#   Part IV  → fortisense_compare.py
#              Consolidated model comparison (RF vs SVM vs NN)
#
#   Part V   → fortisense_server.py
#              Real-time IDS server based on the trained model
#
# For each step, the user is asked whether to run it. This allows:
#   - selective re-execution of stages
#   - skipping expensive training if artefacts already exist
#
# The master orchestrator keeps the individual scripts decoupled
# but provides a single entry point for demos and marking.
# ============================================================


def run_module(script_name, description):
    """
    Run a FortiSense script as a separate Python process.

    Parameters
    ----------
    script_name : str
        File name of the script to execute (e.g. "fortisense_ml.py").
    description : str
        Human readable description used for status messages.

    Notes
    -----
    - Uses subprocess.run with check=True to raise an exception
      if the child process exits with a non-zero status code.
    - On failure, the entire pipeline terminates with exit code 1.
    """
    script_directory = os.path.dirname(__file__)
    script_path = os.path.join(script_directory, script_name)

    print(f"\n[*] FortiSense Master - Running {description} ({script_name})...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"[+] Completed: {description}")
    except subprocess.CalledProcessError as error:
        print(f"[!] Error while running {description}: {error}")
        sys.exit(1)


def confirm_step(description):
    """
    Prompt the user to confirm execution of a pipeline step.

    Parameters
    ----------
    description : str
        Human readable description of the step about to run.

    Returns
    -------
    bool
        True if the user responded with 'y' (case insensitive),
        False for any other input.
    """
    print("\n-----------------------------------------------")
    print(f"Next step: {description}")
    choice = input("Run this step now? (Y/N): ").strip().lower()
    return choice == "y"


def main():
    print("===============================================")
    print(" FortiSense - AI-based IDS Prototype ")
    print("===============================================\n")

    # --------------------------------------------------------
    # 1. EDA
    # --------------------------------------------------------
    if confirm_step("Part I - Exploratory Data Analysis"):
        run_module("fortisense_eda.py", "Part I - Exploratory Data Analysis")
    else:
        print("[+] Skipped: Part I - Exploratory Data Analysis")

    # --------------------------------------------------------
    # 2. Classical ML Models (Random Forest + Linear SVM)
    # --------------------------------------------------------
    if confirm_step("Part II - Classical Machine Learning Models (RF + SVM)"):
        run_module(
            "fortisense_ml.py",
            "Part II - Classical Machine Learning Models (RF + SVM)",
        )
    else:
        print("[+] Skipped: Part II - Classical Machine Learning Models (RF + SVM)")

    # --------------------------------------------------------
    # 3. Neural Network (PyTorch)
    # --------------------------------------------------------
    if confirm_step("Part III - Neural Network Model"):
        run_module("fortisense_nn.py", "Part III - Neural Network Model")
    else:
        print("[+] Skipped: Part III - Neural Network Model")

    # --------------------------------------------------------
    # 4. Model comparison (RF vs SVM vs NN)
    # --------------------------------------------------------
    if confirm_step("Part IV - Model Comparison (RF vs SVM vs NN)"):
        run_module(
            "fortisense_compare.py",
            "Part IV - Model Comparison (RF vs SVM vs NN)",
        )
    else:
        print("[+] Skipped: Part IV - Model Comparison (RF vs SVM vs NN)")

    print("\n[✓] FortiSense Master - Core pipeline finished (with possible skips).")
    print("[*] At this point, any completed training steps have saved models")
    print("    into the 'models' directory.\n")

    # --------------------------------------------------------
    # 5. Start the IDS server (optional)
    # --------------------------------------------------------
    user_choice = input("Start FortiSense IDS server now? (Y/N): ").strip().lower()

    if user_choice == "y":
        print("\n[*] Starting FortiSense IDS Server (Part V - Real-Time IDS Prototype)...")
        print("[*] The server will run in this terminal.")
        print("[*] Open a second terminal and run:")
        print("    python fortisense_client.py")
        print()

        script_directory = os.path.dirname(__file__)
        server_script_path = os.path.join(script_directory, "fortisense_server.py")

        try:
            # This call blocks and keeps the server running in the
            # current terminal until the process is interrupted.
            subprocess.run([sys.executable, server_script_path], check=True)
        except subprocess.CalledProcessError as error:
            print(f"[!] Error while running IDS server: {error}")
            sys.exit(1)
    else:
        print("\n[+] Skipping IDS server start.")
        print("    You can start it manually later with:")
        print("    python fortisense_server.py")
        print()

    print("[✓] FortiSense Master - Finished.")


if __name__ == "__main__":
    main()
