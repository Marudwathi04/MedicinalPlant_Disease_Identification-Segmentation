import torch
from utils1.model import ResNet9
from PIL import Image
import io
from torchvision import transforms
# Load the model
disease_classes = ['Bay Laurel_scab',
                   'Bay Laurel',
                   'Bay Laurel_rust',
                   'Bay Laurel___healthy',
                   'Blueberry___healthy',
                   'Neem___Powdery_mildew',
                   'Neem___healthy',
                   'Turmeric___Cercospora_leaf_spot Gray_leaf_spot',
                   'Turmeric___Common_rust_',
                   'Turmeric',
                   'Turmeric___healthy',
                   'Wild Grape',
                   'Wild Grape___Esca_(Black_Measles)',
                   'Wild Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Wild Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Willow',
                   'Willow___healthy',
                   'Pepper',
                   'Pepper___healthy',
                   'basil___Early_blight',
                   'basil___Late_blight',
                   'basil___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'aloevera___Leaf_scorch',
                   'aloevera___healthy',
                   'Wild Nightshade___Bacterial_spot',
                   'Wild Nightshade___Early_blight',
                   'Wild Nightshade',
                   'Wild Nightshade___Leaf_Mold',
                   'Wild Nightshade___Septoria_leaf_spot',
                   'Wild Nightshade___Spider_mites Two-spotted_spider_mite',
                   'Wild Nightshade___Target_Spot',
                   'Wild Nightshade___Tomato_Yellow_Leaf_Curl_Virus',
                   'Wild Nightshade___Tomato_mosaic_virus',
                   'Wild Nightshade___healthy']



disease_model_path = 'plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


model = torch.load(disease_model_path, map_location=torch.device('cpu'))

def predict_image(img_path, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path)
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

img = "Pepper.jpg"
prediction = predict_image(img)

print(prediction)