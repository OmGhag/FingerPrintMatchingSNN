# FingerPrintMatchingSNN

A project implementing fingerprint matching using Spiking Neural Networks (SNNs) for efficient and biologically-inspired pattern recognition.

This repository explores advanced biometric authentication through SNNs, leveraging neuromorphic computing principles for real-time fingerprint analysis.


- **Efficient**: Utilizes spiking neurons for low-power, event-driven processing.
- **Accurate**: Achieves high matching precision on standard datasets.
- **Scalable**: Modular design for integration into larger biometric systems.

## ➤ Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)

## ➤ Installation

1. Clone the repo
   ```sh
   git clone https://github.com/OmGhag/FingerPrintMatchingSNN.git
   ```
2. Create a Python virtual environment (Python 3.8+ required)
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
   Key libraries include: NumPy, PyTorch, snnTorch (for SNN simulation), OpenCV (for image processing).

## ➤ Getting Started

To get started quickly, follow these steps in order:

1. Run the data preparation script to set up the dataset:
   ```sh
   python data.py
   ```
2. Open and run all cells in `preprocess.ipynb` to preprocess the fingerprint data (e.g., minutiae extraction and feature engineering).
3. Finally, open and run all cells in `train.ipynb` to train the SNN model on the preprocessed data.

For detailed configuration, see [config.yaml](config.yaml) if available.

## ➤ Usage

### Data Preparation
Run the script to load and organize the fingerprint dataset:
```sh
python data.py
```
This handles downloading or loading sample data into the appropriate directories.

### Preprocessing
Use the Jupyter notebook for data cleaning and feature extraction:
- Launch Jupyter: `jupyter notebook preprocess.ipynb`
- Execute all cells sequentially. This includes steps like image enhancement, minutiae detection using OpenCV, and normalization for SNN input.

### Training the Model
Train the SNN using the dedicated notebook:
- Launch Jupyter: `jupyter notebook train.ipynb`
- Execute all cells. This covers model architecture setup (Leaky Integrate-and-Fire neurons with STDP), training loop, and evaluation.

### Key Features
- **Minutiae Extraction**: Ridge endings and bifurcations detected via OpenCV.
- **SNN Architecture**: Leaky Integrate-and-Fire neurons with STDP learning.
- **Evaluation Metrics**: FAR/FRR, EER on benchmark datasets.

After training, saved models can be used for inference in custom scripts.

