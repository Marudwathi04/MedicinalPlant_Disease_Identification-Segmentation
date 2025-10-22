import torch
from utils1.model import ResNet9
from PIL import Image
from torchvision import transforms
# Load the model
disease_classes = ['Bay Laurel_scab',
    'Bay Laurel', 'Bay Laurel_rust', 'Bay Laurel___healthy',
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

def load_model(model_path, num_classes):
    """Loads the ResNet9 model."""
    model = ResNet9(3, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(img_path, model):
    """Predicts the disease for a single image."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path)
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0) # Add batch dimension
    with torch.no_grad():
        yb = model(img_u)
        _, preds = torch.max(yb, dim=1)
        prediction = disease_classes[preds[0].item()]
        return prediction

if __name__ == '__main__':
    disease_model_path = 'plant_disease_model.pth'
    disease_model = load_model(disease_model_path, len(disease_classes))

    # Example usage:
    # Create a dummy image or replace "path/to/your/image.jpg" with an actual image path
    try:
        img_path = "Pepper.jpg" # Make sure this image exists
        prediction = predict_image(img_path, disease_model)
        print(f"Image: {img_path}")
        print(f"Prediction: {prediction}")
    except FileNotFoundError:
        print(f"Error: Make sure an image file exists at 'Pepper.jpg' to run this example.")