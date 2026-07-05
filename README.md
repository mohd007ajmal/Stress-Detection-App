<div align="center">

# 🎙️ Speech Stress Detection using Attention-Based BiLSTM

### Deep Learning-based Speech Stress Detection using MFCC, Delta Features and Attention-Based Bidirectional LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-3.50-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

# 📖 Overview

Speech carries rich information about a person's emotional and psychological state. This project presents a **deep learning-based speech stress detection system** capable of classifying whether a speaker is under stress using only a short voice recording.

The system extracts handcrafted acoustic features from speech and feeds them into an **Attention-Based Bidirectional Long Short-Term Memory (Attention-BiLSTM)** network, enabling the model to focus on the most informative temporal regions of the speech signal.

A lightweight **Gradio** interface provides real-time predictions through a simple web application.

---

# ✨ Features

- 🎤 Real-time voice recording
- 🧠 Deep Learning based stress classification
- 🎯 Attention-Based BiLSTM architecture
- 📊 MFCC Feature Extraction
- 📈 Delta & Delta-Delta MFCC
- 🔊 Zero Crossing Rate (ZCR)
- 📉 Root Mean Square (RMS) Energy
- ⚡ Interactive Gradio Web Interface
- 📦 Easy local deployment

---

# 🏛️ System Architecture

```text
Voice Recording
       │
       ▼
Audio Preprocessing
       │
       ▼
Feature Extraction
(MFCC + ΔMFCC + Δ²MFCC + ZCR + RMS)
       │
       ▼
Feature Scaling
       │
       ▼
Attention-Based BiLSTM
       │
       ▼
Sigmoid Classification
       │
       ▼
Stress / No Stress
```

---

# 🧠 Feature Extraction

The model extracts the following acoustic features from every speech sample.

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| MFCC (40)          | Captures speech spectral characteristics |
| Delta MFCC         | Captures temporal variations             |
| Delta-Delta MFCC   | Captures acceleration of speech dynamics |
| Zero Crossing Rate | Measures frequency changes               |
| RMS Energy         | Measures speech intensity                |

Total Feature Dimension:

```
122 Features × 174 Time Steps
```

---

# 🤖 Model

The proposed model consists of

- Attention Layer
- Bidirectional LSTM
- Dense Layer
- Sigmoid Output Layer

The attention mechanism enables the network to automatically focus on the most informative portions of the speech sequence before classification.

---

# 📂 Project Structure

```text
Speech-Stress-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── final_stress_model.h5
├── stress_scaler.pkl
│
└── Optimized_Speech_Stress_Detection_Using_MFCC_Delta_Features_and_Attention_Based_BiLSTM.ipynb
```

---

# 📊 Dataset

This project was developed using the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

**Official Dataset**

https://doi.org/10.5281/zenodo.1188976

**Citation**

> Livingstone, S. R., & Russo, F. A. (2018). _The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)._ Zenodo. https://doi.org/10.5281/zenodo.1188976

### Dataset Preparation

RAVDESS is an **emotion recognition dataset**, not a dedicated stress dataset.

For this project, the emotion labels were mapped into **binary stress-related classes (Stress / No Stress)** for research purposes.

The dataset is **not included** in this repository due to licensing and storage limitations.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Speech-Stress-Detection.git
```

Move into the project

```bash
cd Speech-Stress-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

The Gradio interface will automatically launch in your browser.

---

# 💻 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Joblib
- Gradio

---

# 🔄 Prediction Pipeline

```text
Speech Recording
      │
      ▼
Load Audio
      │
      ▼
Normalize Audio
      │
      ▼
MFCC Extraction
      │
      ▼
Delta Features
      │
      ▼
Feature Scaling
      │
      ▼
Attention-BiLSTM
      │
      ▼
Prediction
```

---


# 🔮 Future Improvements

- Multi-level stress prediction
- Explainable AI (XAI)
- Mobile deployment
- Cloud API integration
- Background noise robustness
- Cross-language speech support
- Larger stress-specific datasets

---

# ⚠️ Disclaimer

This project is intended solely for **research and educational purposes**.

It is **not** a medical diagnostic tool and should not be used for clinical or psychological diagnosis.

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Ajmal M**

B.Tech – Computer Science & Engineering

Machine Learning • Deep Learning • Artificial Intelligence

GitHub: https://github.com/YOUR_USERNAME
