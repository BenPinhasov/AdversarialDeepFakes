import PIL.Image
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.datasets import YourDataset  # Replace with your dataset class
from torchvision.datasets import DatasetFolder, ImageFolder
from PIL import Image as pil_image
from tqdm import tqdm


# def main():
#     weights = ResNet50_Weights.DEFAULT
#     model = resnet50(weights=weights)
#     model.eval()

def loader(path):
    image = np.load(path) / 2
    return image


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.xai_extension = '_xai.jpg'

    def __getitem__(self, idx):
        image_path, label = self.imgs[idx]
        xai_path = image_path.replace(self.xai_extension, '.jpg')

        image = self.loader(image_path)
        xai_map = self.loader(xai_path)

        if self.transform is not None:
            image = self.transform(image)
            xai_map = self.transform(xai_map)

        return image, xai_map, label


class CustomResNet50(nn.Module):
    def __init__(self, weights):
        super(CustomResNet50, self).__init__()

        # Load the pre-trained ResNet-50 model
        self.resnet50 = resnet50(weights=weights)

        # Remove the final classification layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Add a custom classification layer
        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 512),  # Concatenated feature vectors from 2 images
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)  # Output vector with 2 values (fake and attacked)
        )

    def forward(self, x1, x2):
        # Forward pass for the two input images
        x1 = self.resnet50(x1)
        x2 = self.resnet50(x2)

        # Flatten and concatenate the feature vectors
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        concatenated_features = torch.cat((x1, x2), dim=1)

        # Forward pass through the custom classification layer
        output = self.fc(concatenated_features)
        return output


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


# From chatgpt
def example_for_train_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = 'Datasets/dataset_custom_resent50/EfficientNetB4ST/GuidedBackprop/'
    # data = np.load(data_path).astype("float16")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # dataset = DatasetFolder(root=data_path, loader=loader, extensions=['npy'], transform=transform)
    dataset = CustomImageFolder(root=data_path, transform=transform)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    # Step 1: Load the pre-trained ResNet-50 model
    weights = ResNet50_Weights.DEFAULT
    # model = CustomResNet50(weights=weights)
    model = resnet50(weights=weights)
    model.fc = nn.Linear(2048, 2)
    model = model.to(device)
    # model.resnet50 = model.resnet50.train(False)
    # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Step 4: Load your dataset and create data loaders
    # Replace YourDataset with your actual dataset class and adjust data augmentation/transforms as needed

    # Replace YourDataset with your actual dataset class and set appropriate batch size
    # train_dataset = YourDataset(root='path_to_training_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    activation_function = nn.Softmax(dim=1)
    # Training loop
    num_epochs = 50  # You can adjust this
    epoch_pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_accuracy = 0.0

        batch_pbar = tqdm(total=len(train_loader))
        model = model.train()
        for images, xais, labels in train_loader:
            # images = images.to(device)
            xais = xais.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # outputs = model(images.float(), xais.float())
            outputs = model(xais.float())
            outputs = activation_function(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_pbar.update(1)
        print('testing the model')
        model = model.eval()
        with torch.no_grad():
            for val_images, val_xais, val_labels in test_loader:
                # val_images = val_images.to(device)
                val_xais = val_xais.to(device)
                val_labels = val_labels.to(device)
                # val_outputs = model(val_images.float(), val_xais.float())
                val_outputs = model(val_xais.float())
                val_outputs = activation_function(val_outputs)
                accuracy = calculate_accuracy(val_outputs, val_labels)
                total_accuracy += accuracy

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(test_loader)

        epoch_pbar.update(1)
        batch_pbar.close()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy * 100:.2f}%")
    epoch_pbar.close()

    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_fake_real_classifier1.pth')


if __name__ == '__main__':
    example_for_train_resnet()
