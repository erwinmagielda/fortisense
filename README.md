# FortiSense

Machine learning intrusion detection system prototype for security research and controlled lab environments.

## What It Is
FortiSense is an end-to-end IDS pipeline that evaluates multiple machine learning models on a KDD-style dataset and exposes a simple real-time inference interface. It combines offline analysis with a live classification demo to make model behaviour observable beyond static metrics.

## Why It Exists
Intrusion detection projects often stop at offline evaluation. FortiSense extends that work by reusing trained models in a live IDS setting, making trade-offs, false positives, and limitations visible in practice rather than only on paper.

## Pipeline Overview
1. Exploratory data analysis of network traffic features  
2. Classical ML training and evaluation (Random Forest, Linear SVM)  
3. Neural network baseline (PyTorch MLP)  
4. Unified model comparison using standard metrics  
5. Real-time IDS server and client for live inference  

## Project Structure
```
fortisense/
├── src/
│   ├── fortisense_master.py
│   ├── fortisense_eda.py
│   ├── fortisense_ml.py
│   ├── fortisense_nn.py
│   ├── fortisense_compare.py
│   ├── fortisense_server.py
│   └── fortisense_client.py
│
├── data/          # Datasets (ignored)
├── models/        # Trained artefacts and metrics (ignored)
├── README.md
└── .gitignore
```

## Usage
Run the interactive orchestrator:

```bash
python src/fortisense_master.py
```

The master menu allows you to:
- run EDA
- train models
- compare results
- start the IDS server

In a second terminal, start the client:

```bash
python src/fortisense_client.py
```

## Metrics
Models are evaluated using:
- accuracy
- precision
- recall
- F1-score

F1-score is used as the primary comparison metric due to class imbalance.

## Security Notes
The real-time IDS uses pickle and a minimal TCP protocol. This is unsafe by design and intended only for isolated lab environments.

## Status
Prototype / research project.

## Licence
MIT
