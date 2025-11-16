# 🌿 Medicinal Plant Disease Identification and Segmentation using Deep Learning

## 🧩 Overview
This project focuses on **identifying and segmenting diseases in medicinal plant leaves** using advanced **deep learning** and **computer vision** techniques.  
The goal is to assist farmers, researchers, and botanists in early detection of plant diseases for effective crop management and preservation of medicinal plant quality.

---       

## 🚀 Key Features
- 🔍 **Image Classification:** Predicts disease type from a plant leaf image with high accuracy.  
- 🧠 **Deep Learning Models:** Implemented **ResNet50** and **EfficientNet** achieving **90%+ accuracy**.  
- 🎯 **Segmentation Module:** Highlights the infected leaf regions using OpenCV-based segmentation.  
- 🌐 **Flask Web App:** Simple web interface where users can upload images and get instant predictions.  
- 📊 **Model Evaluation:** Includes confusion matrix, classification report, and visualization of predictions.

---

## 🧠 Tech Stack
| Category | Technologies |
|-----------|---------------|
| **Languages** | Python |
| **Frameworks** | TensorFlow / Keras, Flask |
| **Libraries** | OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn |
| **Deployment** | Flask Web Application |
| **Tools** | Google Colab / Jupyter Notebook |

---

## ⚙️ Workflow
1. **Dataset Collection:** Gathered medicinal plant leaf images (healthy and diseased).  
2. **Data Preprocessing:** Applied image resizing, normalization, and augmentation.  
3. **Model Training:** Used **ResNet50** and **EfficientNet** CNN architectures for classification.  
4. **Segmentation:** Isolated disease-affected regions using OpenCV image masks.  
5. **Evaluation:** Computed accuracy, precision, recall, and F1-score.  
6. **Deployment:** Integrated the trained model into a Flask web app for real-time prediction.

---

## 🧪 Results
- ✅ **Test Accuracy:** 90%+  
- ✅ **Robust Segmentation:** Visual localization of infected areas  
- ✅ **Generalization:** Performs well on unseen images  

Example Workflow:

| Step | Description | Example |
|------|--------------|---------|
| 🖼️ Input | User uploads a medicinal leaf image | ![input](images/sample_input.jpg) |
| ⚙️ Processing | Model classifies and segments diseased area | ![processing](images/sample_segmentation.jpg) |
| 📈 Output | Displays predicted disease name and confidence | ![output](images/sample_output.jpg) |



