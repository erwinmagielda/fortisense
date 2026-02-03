# FortiSense

Machine learning intrusion detection system prototype for security research and controlled lab environments.

## What It Is
FortiSense is an end-to-end intrusion detection pipeline that evaluates multiple machine learning models on a KDD-style dataset and exposes a simple real-time inference interface. The project connects offline analysis to live classification to make model behaviour observable beyond static metrics.

## Why It Exists
Intrusion detection work often stops at offline evaluation. FortiSense exists to extend that workflow into a live IDS context, making model trade-offs, false positives, and operational limitations visible in practice.

## Pipeline Overview
1. Exploratory data analysis of network traffic features  
2. Classical machine learning training and evaluation  
3. Neural network baseline modelling  
4. Unified model comparison using standard metrics  
5. Real-time intrusion detection inference  

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

The menu allows you to:
- Run exploratory analysis  
- Train machine learning models  
- Compare evaluation results  
- Start the IDS server  

In a second terminal, start the client:

```bash
python src/fortisense_client.py
```

## Metrics
Models are evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

F1-score is used as the primary comparison metric due to class imbalance.

## Security Notes
The real-time IDS uses pickle and a minimal TCP protocol. This design is intentionally unsafe and intended only for isolated lab environments.

## Status
Prototype research project.

## Licence
MIT
