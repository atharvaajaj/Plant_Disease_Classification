# 🌿 Plant Disease Detection using Deep Learning

### 🚀 Live Demo  
👉 **[Try the Web App on Hugging Face Spaces](https://huggingface.co/spaces/Atharva046/Plant_Disease_Classification_Using_DL)**

---

## 🧠 Overview

The **Plant Disease Detection System** is a deep learning–powered web application developed as part of the **AICTE Internship Program**.  
It helps farmers and researchers identify plant diseases early by analyzing leaf images using a **Convolutional Neural Network (CNN)**.  
This project promotes **AI-driven sustainable agriculture** by offering an efficient and scalable disease classification solution.

---

## 🌾 Dataset Information

- 📦 **Dataset Source:** [Kaggle – New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- 📸 Contains over 87,000 images of **healthy and diseased plant leaves**  
- 🪴 Covers multiple plant species (Tomato, Corn, Potato, etc.)  
- 🧹 Includes labeled folders for supervised training  

---

## 🔗 Model Access

You can download the trained CNN model from Google Drive:  
📥 [Plant Disease CNN Model (.keras)](https://drive.google.com/file/d/1UVKP-Ihap3GI03P2kHzLTO5BP_HTtcyK/view?usp=sharing)

---

## 🎯 Objectives

- Detect and classify plant diseases using deep learning  
- Provide a real-time, user-friendly web app interface  
- Support sustainable agriculture through intelligent crop monitoring  

---

## 🧩 Problem Statement

Traditional methods of plant disease detection are:  
- Time-consuming  
- Require expert agronomists  
- Not scalable for large agricultural fields  

🌱 This system automates detection using CNNs, improving efficiency and reducing dependency on manual observation.

---

## 🧰 Tools and Technologies

| Category | Tools |
|-----------|-------|
| Programming | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV, NumPy |
| Web Framework | Streamlit |
| Visualization | Matplotlib, Pandas |
| Deployment | Hugging Face Spaces |

---

## ⚙️ Methodology

### 1️⃣ Data Collection  
Plant leaf images were collected from the Kaggle dataset mentioned above.

### 2️⃣ Preprocessing  
- Image resizing and normalization  
- Data augmentation (rotation, flipping, brightness)  

### 3️⃣ Model Training  
A CNN model was trained for disease classification using Keras and TensorFlow.

### 4️⃣ Evaluation  
Evaluated using accuracy, precision, recall, and confusion matrix.

### 5️⃣ Deployment  
The trained model was deployed on **Hugging Face Spaces** using **Streamlit**.

---

## 🧮 Model Architecture

- **Input Layer:** 128×128×3 RGB image  
- **Convolutional + Pooling Layers:** Extract features from leaf texture  
- **Dropout Layers:** Reduce overfitting  
- **Dense Layers:** Learn non-linear patterns  
- **Softmax Output:** Multi-class disease classification  

---

## 🧠 Architecture Diagram

Below is a visual representation of the system workflow — from image input to cloud deployment:

*(Replace with your actual uploaded image filename if different.)*

---

## 🖥️ Streamlit Web App Features

- 🌿 Upload any plant leaf image  
- 🧠 Get instant CNN-based disease predictions  
- 📊 Displays confidence score for each prediction  
- 🎨 Clean, responsive, and minimal interface  

---

## 🌱 Impact on Sustainable Agriculture

- Early detection reduces crop losses  
- Low-cost and scalable AI solution for rural farming  
- Supports smart agriculture initiatives  

---

## 🔮 Future Enhancements

- Expand support for additional crop types and diseases  
- Integrate IoT-based live leaf monitoring  
- Deploy mobile version for offline use  

---

## 🧾 Project Structure

```
├── app.py                    # Streamlit app entry point
├── models/
│   └── plant_disease_model.keras
├── requirements.txt           # Python dependencies
├── dataset/                   # (optional) training dataset
└── README.md
```

---

## 👨‍💻 Author

**Atharva Joshi**  
🎓 AICTE Student ID: STU6767e244a3d251734861380  
💡 Focus Areas: Deep Learning | Computer Vision | AI for Agriculture  
🔗 GitHub: [@atharvaajaj](https://github.com/atharvaajaj)  
🔗 Hugging Face: [@Atharva046](https://huggingface.co/Atharva046)

---

## 📚 References

- [TensorFlow](https://www.tensorflow.org/)  
- [OpenCV](https://opencv.org/)  
- [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- [Hugging Face Spaces](https://huggingface.co/spaces)


