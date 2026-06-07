# Radar-Based Human Activity Recognition
### Comparative Study of SVM, CNN, and KNN on Radar Spectrograms

<p align="center">
  <img src="https://img.shields.io/badge/Research-ICCIS%202025-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Institution-BITS%20Pilani%2C%20Goa-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

---

## Overview

This repository contains the implementation of our research paper:

> **"Efficient Human Activity Recognition from Radar Spectrograms: Comparative Study of SVM, CNN, and KNN for Tactical Edge Deployment"**
>
> *Shri Manasa S, Sarvani Sruthi Chundi*
> Presented at the **7th International Conference on Communication and Intelligent Systems (ICCIS 2025)**
> BITS Pilani, K K Birla Goa Campus — September 26–27, 2025

The study investigates machine learning approaches for recognizing human activities from radar spectrogram data using Fourier-transformed features, with a focus on deployment feasibility at the tactical edge.

---

## Research Highlights

-  **Radar spectrograms** as input — privacy-preserving and environment-robust
-  **Fourier Transform (FFT)** for frequency-domain feature extraction
-  **Three-model comparative analysis** — SVM, KNN, and CNN
-  **Tactical edge deployment** focus — lightweight and efficient inference
-  **PostgreSQL** backend for scalable image data management

---

## Methodology

```
Radar Spectrogram Images (PostgreSQL)
            │
            ▼
   Preprocessing & Grayscale Conversion
            │
            ▼
   2D Fast Fourier Transform (FFT)
   Log-Magnitude Spectrum Extraction
            │
            ▼
   ┌─────────────────────────────┐
   │     Feature Matrix (64×64)  │
   └─────────────────────────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
   SVM     KNN     CNN
    │       │       │
    └───────┴───────┘
            │
            ▼
   Accuracy Comparison & Evaluation
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Classical ML | Scikit-learn (SVM, KNN) |
| Signal Processing | SciPy (FFT), OpenCV |
| Data Storage | PostgreSQL + psycopg2 |
| Visualization | Matplotlib, Seaborn |
| Image Processing | Pillow, NumPy |

---

## Project Structure

```
radar-har-classifier/
├── human_activity_classifier.py   # Main classifier pipeline
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment variable template
└── outputs/                       # Generated plots (gitignored)
    ├── model_comparison.png
    └── cnn_training_history.png
```

---

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/shrimanasa/radar-har-classifier.git
cd radar-har-classifier
```

**2. Install dependencies**
```bash
pip install tensorflow scikit-learn psycopg2-binary scipy opencv-python pillow matplotlib seaborn numpy
```

**3. Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

**4. Run the classifier**
```bash
python human_activity_classifier.py
```

---

## Results

The classifier outputs:
- Per-model accuracy scores (SVM, KNN, CNN)
- Model comparison bar chart
- CNN training & validation curves
- Debug spectrogram visualizations

---

## Citation

If you use this work, please cite:

```
Shri Manasa S, Sarvani Sruthi Chundi,
"Efficient Human Activity Recognition from Radar Spectrograms:
Comparative Study of SVM, CNN, and KNN for Tactical Edge Deployment",
ICCIS 2025, BITS Pilani, Goa Campus, September 2025.
```

---

<p align="center">
  <i>Built with curiosity. Presented with purpose.</i>
</p>
