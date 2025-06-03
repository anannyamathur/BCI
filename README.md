# Connecting Brains and Interfaces: Real-Time EEG-Based Stress Detection via Spiking Neural Networks

This repository provides the reference implementation of our **Late-Breaking Work (LBW)** paper accepted at [ACM COMPASS 2025](https://compass.acm.org/). 

> **Note**: The corresponding ACM Digital Library entry will be updated here once the paper is officially published.

This project presents a novel approach for efficient, real-time classification of brain signals using Spiking Neural Networks (SNNs). Designed for low-power brain-machine interface applications, the architecture processes EEG data in just a few time steps.

The core of the system is a multi-layer SNN model that generates spike-based predictions, which are then combined using an ensemble of:

- Support Vector Classifier (SVC) and

- XGBoost decision trees.

Performance was benchmarked against a modified Symmetric Convolutional and Adversarial Neural Network (SCANN):

- 3-class EEG task (1500 ms / 300 timesteps):
Ours: 43% vs SCANN: 38%

- 2-class EEG task (500 ms / 100 timesteps):
Ours: 60% vs SCANN: 50%

> This is a reference implementation accompanying our LBW paper at ACM COMPASS 2025. The repository is still evolving.

## Dataset 




## Citation
 Amit Kumar, Anannya Mathur, and Deepak Joshi. 2025. Connecting Brains
 and Interfaces: Real-Time EEG-Based Stress Detection via Spiking Neural
 Networks.InACMSIGCAS/SIGCHIConferenceonComputingandSustainable
 Societies (COMPASS ’25), July 22–25, 2025, Toronto, ON, Canada. ACM, New
 York, NY, USA, 6 pages. https://doi.org/10.1145/3715335.3736311
