import os
import sys
import subprocess

# ============================================================
# FortiSense - Master Orchestrator
#
# Runs the full project pipeline in order:
#   1. EDA (fortisense_eda.py)
#   2. Classical ML Models (fortisense_ml.py)
#   3. NN (fortisense_nn.py)
#   4. Model comparison (fortisense_compare.py)
#   5. Optionally starts the IDS server (fortisense_server.py)
#
# The script pauses before each step and asks to proceed.
# ============================================================


def run_module(script_name, description):
    """
    Run a Python module as a separate process and print
    clear status messages around it.
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
    Ask the user whether to run the given pipeline step.
    Returns True if the user answered 'y', False otherwise.
    """
    print("\n-----------------------------------------------")
    print(f"Next step: {description}")
    choice = input("Run this step now? (Y/N): ").strip().lower()
    return choice == "y"


def main():
    print("===============================================")
    print(" FortiSense - AI-based IDS Prototype ")
    print("===============================================\n")

    # 1. EDA
    if confirm_step("Part I - Exploratory Data Analysis"):
        run_module("fortisense_eda.py", "Part 1 - Exploratory Data Analysis")
    else:
        print("[+] Skipped: Part I - Exploratory Data Analysis")

    # 2. Classical ML Modlels (Random Forest + Linear SVM)
    if confirm_step("Part II - Classical ML Models (Random Forest + Linear SVM)"):
        run_module("fortisense_ml.py", "Part 2 - Classical ML Models")
    else:
        print("[+] Skipped: Part II - Classical ML Models")

    # 3. Neural Network (PyTorch)
    if confirm_step("Part III - Neural Network Model (PyTorch)"):
        run_module("fortisense_nn.py", "Part III - Neural Network Model")
    else:
        print("[+] Skipped: Part III - Neural Network Model")

    # 4. Model comparison (RF vs SVM vs NN)
    if confirm_step("Part IV - Model Comparison (RF vs SVM vs NN)"):
        run_module("fortisense_compare.py", "Part IV - Model Comparison")
    else:
        print("[+] Skipped: Part IV - Model Comparison")

    print("\n[✓] FortiSense Master - Core pipeline finished (with possible skips).")
    print("[*] At this point, any completed training steps have saved models")
    print("    into the 'models' directory.\n")

    # 5. Ask whether to start the IDS server
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
            # This will block and keep the server running
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
