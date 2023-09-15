from torchvision.models import resnet50, ResNet50_Weights


def main():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()


# From chatgpt
def example_for_train_resnet():
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision.datasets import YourDataset  # Replace with your dataset class

    # Step 1: Load the pre-trained ResNet-50 model
    resnet50 = models.resnet50(pretrained=True)

    # Step 2: Modify the classifier (fully connected) layer
    num_classes = 2  # Fake or real
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, num_classes)

    # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

    # Step 4: Load your dataset and create data loaders
    # Replace YourDataset with your actual dataset class and adjust data augmentation/transforms as needed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Replace YourDataset with your actual dataset class and set appropriate batch size
    train_dataset = YourDataset(root='path_to_training_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 10  # You can adjust this
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model
    torch.save(resnet50.state_dict(), 'resnet50_fake_real_classifier.pth')


if __name__ == '__main__':
    main()
