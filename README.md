# Privacy-Preserving Federated Learning for Intrusion Detection Systems

## Overview

This project implements a **Federated Learning-based Intrusion Detection System (IDS)** with **Differential Privacy** for secure and privacy-preserving collaborative model training.

The system allows multiple clients to train local models on their own network traffic data while sharing only model updates with a central server. Differential Privacy mechanisms are applied to protect sensitive information during the training process.

This work was developed as part of a **dissertation research project** on privacy-preserving machine learning for cybersecurity.

---

## Dataset

The experiments use the **CICIDS2017 dataset**, which contains labeled network traffic for multiple types of cyber attacks.

Dataset source:
https://www.unb.ca/cic/datasets/ids-2017.html

Example traffic types included:

* DDoS
* PortScan
* Web Attacks
* Infiltration
* Benign traffic

Due to GitHub size limits, the dataset files are **not included in this repository**.

---

## Features

* Federated Learning model training
* Differential Privacy integration
* Privacy budget (ε) tracking
* ROC-AUC performance evaluation
* Confusion Matrix visualization
* Communication cost analysis

---

## Project Structure

```
Federated-learning/
│
├── federated learning.py
├── Federated leraining copy.py
├── README.md
├── .gitignore
└── results/
```

---

## Requirements

Install the required Python libraries:

```
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## Running the Project

Run the federated learning simulation:

```
python "federated learning.py"
```

The program will:

1. Load and preprocess the dataset
2. Simulate multiple federated clients
3. Train local models
4. Aggregate updates on the server
5. Apply differential privacy noise
6. Generate performance metrics

---

## Evaluation Metrics

The system evaluates the IDS using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Additionally, the project visualizes:

* Communication cost per federated round
* Differential privacy budget (ε)

---

## Research Contribution

This project demonstrates how **Federated Learning combined with Differential Privacy** can improve privacy protection in distributed intrusion detection systems while maintaining strong detection performance.

---

## Author

Kelvin E. M. Morkel
MSc / BSc Research Project
Federated Learning for Network Intrusion Detection

---
