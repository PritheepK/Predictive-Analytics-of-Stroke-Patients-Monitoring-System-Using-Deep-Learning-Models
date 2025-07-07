import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from timm.models import create_model
import matplotlib.pyplot as plt

# Define your CustomConvNeXt class here
class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNeXt, self).__init__()
        self.model = create_model('convnext_small', pretrained=True)
        self.model.head = nn.Identity()  # Remove original head
        self.pool = nn.AdaptiveAvgPool2d(1)  # Adaptive pooling to get fixed-size output
        self.fc = nn.Linear(self.model.num_features, num_classes)  # New head for classification

    def forward(self, x):
        x = self.model(x)  # Forward through ConvNeXt
        x = self.pool(x)   # Apply adaptive pooling
        x = torch.flatten(x, 1)  # Flatten for linear layer
        x = self.fc(x)     # Forward through new head
        return x

# Load the trained model
model = CustomConvNeXt(num_classes=5)  # Replace 5 with your actual number of classes
model.load_state_dict(torch.load('convnext_model.pth'))
model.eval()

# Define the transformation (ensure this matches your training transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class names for mapping indices to names
class_names = ['Atrial Fibrillation (AFib)', 'Enlarged Heart (Cardiomegaly)', 
               'Heart Attack (Myocardial Infarction)', 'Heart Block (Arrhythmia related)', 'Normal']

# Function to make predictions
def predict(image_path, model, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)  # Compute probabilities
        _, predicted = torch.max(outputs.data, 1)
        predicted_class_name = class_names[predicted.item()]
        predicted_probability = probs[0, predicted.item()].item()
        return predicted.item(), predicted_class_name, predicted_probability

# Function to visualize the image
def visualize_prediction(image_path, class_name, probability):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Predicted: {class_name} (Probability: {probability:.2f})")
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'hb.jpg'  # Replace with the path to your test image
predicted_class_index, predicted_class_name, predicted_probability = predict(image_path, model, transform, class_names)

print(f'Predicted class index: {predicted_class_index}, Class name: {predicted_class_name}, Probability: {predicted_probability:.4f}')
visualize_prediction(image_path, predicted_class_name, predicted_probability)

# Optionally, save the image with the predicted class label
output_image_path = f'predicted_{predicted_class_name.replace(" ", "_").lower()}.jpg'
image = Image.open(image_path)
image.save(output_image_path)
print(f"Predicted image saved as {output_image_path}")
