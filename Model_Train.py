import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from timm.models import create_model
from sklearn.metrics import classification_report

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

def main():
    # Step 1: Set Up Your Custom Dataset
    data_dir = r'F:\ABDUL\ABDUL 2024\HEART_STROK\dataset'  # Change this to your dataset path

    # Define transformations for the training and validation sets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Step 2: Define the Custom ConvNeXt Model
    model = CustomConvNeXt(num_classes=len(train_dataset.classes))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Step 3: Set Up Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Step 4: Validation and Classification Report
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    print(report)

    # Save classification report to a text file
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    # Save the trained model
    torch.save(model.state_dict(), 'convnext_model.pth')

    print("Model saved as 'convnext_model.pth' and classification report saved as 'classification_report.txt'.")

if __name__ == "__main__":
    main()
