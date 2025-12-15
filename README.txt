FortiSense is an AI-driven Intrusion Detection System (IDS) prototype implemented fully in Python. It combines exploratory data analysis, classical machine learning models, a neural network classifier, and a simple real-time IDS client/server prototype.

Python 3.11 is recommended due to compatibility issues with some dependencies (notably Seaborn) on newer Python versions.

Dependencies listed in dependencies.txt
pip install -r dependencies.txt

Run the master script to execute the full pipeline:
python fortisense_master.py

At the last step of pipeline, run client independently:
python fortisense_client.py