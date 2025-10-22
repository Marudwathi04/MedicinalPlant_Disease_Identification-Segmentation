import streamlit as st
import torch
from PIL import Image
import numpy as np
from utils1.model import ResNet9
from torchvision import transforms
from ultralytics import YOLO
import io

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

@st.cache_resource
def load_disease_model():
    model_path = 'plant_disease_model.pth'
    model = ResNet9(3, len(disease_classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_yolo_model():
    model = YOLO("best (1).pt")
    return model

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
def predict_disease(img, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    img_t = transform(img)
    img_u = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        yb = model(img_u)
        _, preds = torch.max(yb, dim=1)
        return disease_classes[preds[0].item()]

def detect_objects(pil_image, model):
    results = model(pil_image, conf=0.5)
    # The plot() method returns a BGR numpy array with detections
    annotated_image_bgr = results[0].plot()
    # Convert BGR to RGB for display in Streamlit
    annotated_image_rgb = annotated_image_bgr[..., ::-1]
    return annotated_image_rgb

# Streamlit App
st.title("Plant Disease and Object Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load models
    disease_model = load_disease_model()
    yolo_model = load_yolo_model()

    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Disease classification
    st.subheader("Disease Classification Result")
    disease_prediction = predict_disease(img, disease_model)
    st.write(f"**Predicted Leaf:** {disease_prediction}")

    # Show predefined disease and remedy
    # Handle both '___' and '_' as separators for base plant name
    if '___' in disease_prediction:
        base_disease = disease_prediction.split('___')[0]
    else:
        base_disease = disease_prediction.split('_')[0]

    if base_disease in remedies:
        disease_info = remedies[base_disease]
        st.write(f"**Disease Name:** {disease_info['disease']}")
        st.subheader("Suggested Remedy")
        st.write(disease_info['remedy'])

    # YOLO detection
    st.subheader("Disease Segmentation Result")
    detected_image = detect_objects(img, yolo_model)
    st.image(detected_image, caption="YOLO Detection", use_container_width=True)
