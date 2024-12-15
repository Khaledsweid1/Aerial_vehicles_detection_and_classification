import torch
from torchvision import models, transforms

# 1. Load the trained ResNet-18 model
resnet18 = models.resnet18(weights=None)  # Set pretrained=False to use custom weights
num_features = resnet18.fc.in_features
resnet18.fc = torch.nn.Linear(num_features, 5)  # Modify the final layer for 5 classes
resnet18.load_state_dict(torch.load('best_model10epochs.pth', map_location=torch.device('cpu')))  # Map weights to CPU
resnet18.eval()  # Set the model to evaluation mode

# 2. Define the image transformation (resize, normalize) as done during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 as expected by ResNet
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


# 3. Function to process and classify an image
def classify_image10(image):
    # Apply transformations to the image (resize, tensor, normalize)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move model and image to the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18.to(device)
    image_tensor = image_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        output = resnet18(image_tensor)
        _, predicted_class = torch.max(output, 1)

    # Class names (updated with your actual class names)
    class_names = ['Airplanes', 'Drones', 'Fighterjets', 'Helicopters', 'UAVs']
    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name