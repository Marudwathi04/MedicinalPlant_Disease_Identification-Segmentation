import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from utils1.model import ResNet9
from torchvision import transforms
from ultralytics import YOLO

# Disease classification model setup
disease_classes = [
    'Bay Laurel_scab', 'Bay Laurel', 'Bay Laurel_rust', 'Bay Laurel___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Turmeric___Cercospora_leaf_spot Gray_leaf_spot',
    'Turmeric___Common_rust_', 'Turmeric', 'Turmeric___healthy', 'Wild Grape',
    'Wild Grape___Esca_(Black_Measles)', 'Wild Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Wild Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Willow',
    'Willow___healthy', 'Pepper', 'Pepper___healthy', 'Bitter Leaf___Early_blight',
    'Bitter Leaf', 'Bitter Leaf___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Ground ivy', 'Ground ivy___healthy',
    'Lindens', 'Wild Nightshade___Early_blight', 'Wild Nightshade',
    'Wild Nightshade___Leaf_Mold', 'Wild Nightshade___Septoria_leaf_spot',
    'Wild Nightshade___Spider_mites Two-spotted_spider_mite', 'Wild Nightshade___Target_Spot',
    'Wild Nightshade___Tomato_Yellow_Leaf_Curl_Virus', 'Wild Nightshade___Tomato_mosaic_virus',
    'Wild Nightshade___healthy'
]

disease_model_path = 'plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# YOLO model setup
yolo_model = YOLO("best (1).pt")

# Remedies dictionary with predefined disease names
remedies = {
    "Bay Laurel": {
        "disease": "Scab",
        "remedy": "Ensure proper air circulation and prune infected branches."
    },
    "Pepper": {
        "disease": "Anthracnose",
        "remedy": "Use fungicides containing copper and avoid overhead watering."
    },
    "Turmeric": {
        "disease": "Cercospora Leaf Spot",
        "remedy": "Use disease-free rhizomes and apply neem-based products."
    },
    "Wild Nightshade": {
        "disease": "Bacterial Spot",
        "remedy": "Remove infected leaves and use organic insecticides."
    },
    "Wild Grape": {
        "disease": "Black Measles",
        "remedy": "Prune and destroy infected vines, and ensure proper vineyard sanitation."
    },
    "Willow": {
        "disease": "Rust",
        "remedy": "Cut and remove infected branches, and avoid waterlogging."
    },
    "Lindens": {
        "disease": "Bacterial Spot",
        "remedy": "Rotate Lindens with non-solanaceous crops for at least two years to reduce disease build-up in the soil."
    },
    "Ground ivy": {
        "disease": "Leaf Scorch",
        "remedy": "Regularly remove and destroy infected plant debris to reduce the pathogen's overwintering sites. This is crucial for limiting disease spread."
    },
    "Bitter Leaf": {
        "disease": "Bacterial Blight",
        "remedy": "Use copper-based products as preventive measures and remove infected leaves."
    }
}

# Helper functions
def predict_disease(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    img_t = transform(img)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

def detect_objects(image_path, model=yolo_model):
    results = model(image_path, conf=0.5)
    original_image = cv2.imread(image_path)
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Streamlit App
st.title("Plant Disease and Object Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded file temporarily for YOLO processing
    temp_image_path = "temp_image.jpg"
    img.save(temp_image_path)

    # Disease classification
    st.subheader("Disease Classification Result")
    disease_prediction = predict_disease(img)
    st.write(f"**Predicted Leaf:** {disease_prediction}")

    # Show predefined disease and remedy
    base_disease = disease_prediction.split('___')[0]
    if base_disease in remedies:
        disease_info = remedies[base_disease]
        st.write(f"**Disease Name:** {disease_info['disease']}")
        st.subheader("Suggested Remedy")
        st.write(disease_info['remedy'])
    else:
        st.write("No predefined disease or remedy available for this plant.")

    # YOLO detection
    st.subheader("Disease Segmentation Result")
    detected_image = detect_objects(temp_image_path)
    st.image(detected_image, caption="YOLO Detection", use_container_width=True)
