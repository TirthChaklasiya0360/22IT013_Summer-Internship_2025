# 22IT013_Summer-Internship_2025
# CNN-Driven Analysis of Handwritten Hindi Digits

## 📌 Overview
This project focuses on *Handwritten Hindi Digit Recognition* using *Deep Learning* techniques. We implement a *Convolutional Neural Network (CNN)* and a *Deep Feed-Forward Neural Network (DFFNN)* to classify handwritten digits (0–9) from the *MNIST Hindi Digit Dataset*.

The project demonstrates the effectiveness of deep learning in solving real-world OCR (Optical Character Recognition) problems for Indian scripts.

---

## 🚀 Features
- Handwritten digit recognition (0–9) in *Hindi numerals*
- Implemented using *CNN*
- *Data Preprocessing & Augmentation* (scaling, translation, rotation)
- *Optimizers*: RMSprop & Adam
- Achieved *high accuracy* on the Hindi MNIST dataset
- Potential applications in:
  - Postal automation
  - Document digitization
  - Data entry automation

---

## 📂 Dataset
- *Dataset Used*: [MNIST Hindi Digit Dataset](https://www.kaggle.com/datasets/adarshpalikonda/hindi-mnist-dataset)
- *Size*: ~20,000 samples
- *Image Format*: 28×28 grayscale images
- *Classes*: 10 (Digits 0–9 in Hindi script)

---

## ⚙ Methodology
1. *Data Preprocessing*
   - Normalization (pixel values between 0–1)
   - Data Augmentation (rotation up to 15°, scaling, translation)

2. *Model Architecture*
   - *Convolutional Layers* for feature extraction
   - *Pooling Layers* (Max & Average Pooling) for dimensionality reduction
   - *Fully Connected Layers* for classification
   - *Softmax Classifier* for final digit prediction

3. *Optimizers Used*
   - *RMSprop* – reduces oscillations, improves convergence
   - *Adam* – adaptive learning rate for efficient training

---

## 📊 Results
- CNN achieved *state-of-the-art accuracy* on handwritten Hindi digit recognition
- Outperformed traditional ML models (KNN, SVM, MLP)
- Robust to variations in handwriting styles and orientations

---

## 🖥 Tech Stack
- *Language*: Python
- *Libraries/Frameworks*:
  - TensorFlow / Keras
  - NumPy
  - Matplotlib

---

## 🔮 Future Scope
- Extend recognition to *complete Devanagari characters*
- Real-time recognition system for *mobile & web applications*
- Integration with *postal automation and document scanning systems*

---

## 📜 Authors
- Tirth Chaklasiya

Department of Information Technology, Chandubhai S. Patel Institute of Technology, CHARUSAT, Gujarat, India.

---
