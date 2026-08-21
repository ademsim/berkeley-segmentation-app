# 🖼️ Berkeley Image Segmentation with VGG16 Transfer Learning

This repository contains an interactive web application built with **Streamlit** that performs pixel-level image segmentation using a custom-trained **VGG16 Transfer Learning** model. The project leverages **Keras 3** with the **PyTorch backend** and automatically retrieves trained model weights from Google Drive.

---

## 🚀 Features

* **Interactive Web Interface:** Upload any standard image (`.jpg`, `.jpeg`, `.png`) and view the original image alongside its generated segmentation mask side-by-side.
* **Transfer Learning Architecture:** Utilizes pre-trained VGG16 weights for robust feature extraction combined with an encoder-decoder segmentation structure.
* **Cloud-Optimized Model Loading:** Bypasses GitHub's file size limits by securely downloading the `.keras` model weights on-the-fly using `gdown` and caching them efficiently via Streamlit.
* **Cross-Backend Stability:** Configured with Keras 3 and PyTorch backend for high compatibility across local environments and cloud runtimes.

---

## 🛠️ Tech Stack

* **Python** (3.11+)
* **Streamlit** (Web Framework)
* **TensorFlow / Keras 3** (Deep Learning)
* **PyTorch** (Model Backend)
* **NumPy & Pillow** (Image Processing & Matrix Operations)
* **gdown** (Google Drive Integration)

---

## 📂 Project Structure

```text
berkeley-segmentation-app/
│
├── app.py                # Main Streamlit application script
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version configuration
└── README.md             # Project documentation

Installation & Local Execution
To run this application locally on your machine, follow these steps:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/berkeley-segmentation-app.git
   cd berkeley-segmentation-app

2. **Install the dependencies::**
  ```bash
  pip install -r requirements.txt
