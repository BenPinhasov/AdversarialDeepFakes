import os

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
from torch.utils.tensorboard import SummaryWriter
from dataset.transform import ImageXaiFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime as dt

# def main():
#     weights = ResNet50_Weights.DEFAULT
#     model = resnet50(weights=weights)
#     model.eval()

def loader(path):
    image = np.load(path) / 2
    return image


class CustomResNet50(nn.Module):
    def __init__(self, weights, dropout=False):
        super(CustomResNet50, self).__init__()

        # Load the pre-trained ResNet-50 model
        self.resnet50 = resnet50(weights=weights)

        # Remove the final classification layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        if dropout:
        # Add a custom classification layer
            self.fc = nn.Sequential(
                nn.Linear(2048 * 2, 512),  # Concatenated feature vectors from 2 images
                nn.BatchNorm3d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 2)  # Output vector with 2 values (fake and attacked)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2048 * 2, 512),  # Concatenated feature vectors from 2 images
                nn.ReLU(inplace=True),
                nn.Linear(512, 2)
            )

    def forward(self, x1, x2):
        # Forward pass for the two input images
        batch_size = x1.shape[0]
        output = self.resnet50(torch.vstack([x1, x2]))
        # x2 = self.resnet50(x2)
        x1 = output[:batch_size]
        x2 = output[batch_size:]

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
    _, labels = torch.max(labels, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


# From chatgpt
def example_for_train_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_original_crops_path = r'newDataset\Train\Frames\original\xception\original'
    train_original_xai_path = r'newDataset\Train\Frames\original\xception\GuidedBackprop'
    train_attacked_path = r'newDataset\Train\Frames\attacked\Deepfakes\xception\original'
    train_attacked_xai_path = r'newDataset\Train\Frames\attacked\Deepfakes\xception\GuidedBackprop'
    validation_original_crops_path = r'newDataset\Validation\Frames\original\xception\original'
    validation_original_xai_path = r'newDataset\Validation\Frames\original\xception\GuidedBackprop'
    validation_attacked_path = r'newDataset\Validation\Frames\attacked\Deepfakes\xception\original'
    validation_attacked_xai_path = r'newDataset\Validation\Frames\attacked\Deepfakes\xception\GuidedBackprop'

    num_epochs = 100
    lr = 0.1
    batch_size = 16
    dropout = False
    time = dt.datetime.now().strftime('%b%d_%H-%M-%S')
    summery_path = f'runs/{time}_lr{lr}_batch{batch_size}_dropout{dropout}'
    summery_writer = SummaryWriter(log_dir=summery_path)
    # data = np.load(data_path).astype("float16")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # dataset = DatasetFolder(root=data_path, loader=loader, extensions=['npy'], transform=transform)
    train_dataset = ImageXaiFolder(
        original_path=train_original_crops_path,
        original_xai_path=train_original_xai_path,
        attacked_path=train_attacked_path,
        attacked_xai_path=train_attacked_xai_path,
        transform=transform)
    validation_dataset = ImageXaiFolder(
        original_path=validation_original_crops_path,
        original_xai_path=validation_original_xai_path,
        attacked_path=validation_attacked_path,
        attacked_xai_path=validation_attacked_xai_path,
        transform=transform)
    # total_len = len(dataset)
    # train_len = int(total_len * 0.7)
    # valid_len = int(total_len * 0.2)
    # test_len = total_len - train_len - valid_len
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
    #                                                                            [train_len, valid_len, test_len])
    # train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    # Step 1: Load the pre-trained ResNet-50 model
    weights = ResNet50_Weights.DEFAULT
    model = CustomResNet50(weights=weights, dropout=dropout)
    # model = resnet50(weights=weights)
    # model.fc = nn.Linear(2048, 2)
    model = model.to(device)
    # model.resnet50 = model.resnet50.train(False)
    # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, min_lr=0.0001)
    # Step 4: Load your dataset and create data loaders
    # Replace YourDataset with your actual dataset class and adjust data augmentation/transforms as needed

    # Replace YourDataset with your actual dataset class and set appropriate batch size
    # train_dataset = YourDataset(root='path_to_training_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    activation_function = nn.Softmax(dim=1)
    # Training loop
    best_val_acc = 0
    best_model = None

    epoch_pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_accuracy = 0.0
        train_accuracy = 0.0
        batch_pbar = tqdm(total=len(train_loader))
        model = model.train()
        for images, xais, labels in train_loader:
            images = images.to(device)
            xais = xais.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.float(), xais.float())
            # outputs = model(xais.float())
            # outputs = activation_function(outputs)
            # outputs = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            train_accuracy += calculate_accuracy(activation_function(outputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_pbar.update(1)

        avg_loss = running_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        summery_writer.add_scalar('Loss/train', avg_loss, epoch)
        summery_writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)

        print('validation the model')
        val_running_loss = 0.0
        model = model.eval()
        with torch.no_grad():
            for val_images, val_xais, val_labels in validation_loader:
                val_images = val_images.to(device)
                val_xais = val_xais.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_images.float(), val_xais.float())
                # val_outputs = model(val_xais.float())
                # val_outputs = activation_function(val_outputs)
                val_loss = criterion(val_outputs, val_labels)
                accuracy = calculate_accuracy(activation_function(val_outputs), val_labels)
                total_accuracy += accuracy
                val_running_loss += val_loss.item()
        val_avg_loss = val_running_loss / len(validation_loader)
        avg_accuracy = total_accuracy / len(validation_loader)
        # scheduler.step(val_avg_loss)
        summery_writer.add_scalar('Accuracy/validation', avg_accuracy, epoch)
        summery_writer.add_scalar('Loss/validation', val_avg_loss, epoch)
        summery_writer.add_scalar('LR/validation', optimizer.param_groups[0]['lr'], epoch)
        epoch_pbar.update(1)
        batch_pbar.close()
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_model = model.state_dict()
            torch.save(best_model, summery_path+'/best_model.pth')
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy * 100:.2f}%")
    epoch_pbar.close()
    if best_model is not None:
        torch.save(best_model, summery_path+'/best_model.pth')
    torch.save(best_model, summery_path+'/last_model.pth')
    print('testing the model')
    # test the model with test dataset
    total_accuracy = 0
    model = model.eval()
    with torch.no_grad():
        for test_images, test_xais, test_labels in test_loader:
            # val_images = val_images.to(device)
            test_xais = test_xais.to(device)
            test_labels = test_labels.to(device)
            # val_outputs = model(val_images.float(), val_xais.float())
            test_outputs = model(test_xais.float())
            test_outputs = activation_function(test_outputs)
            accuracy = calculate_accuracy(test_outputs, test_labels)
            total_accuracy += accuracy

    avg_accuracy = total_accuracy / len(validation_loader)
    print(f"Test Accuracy: {avg_accuracy * 100:.2f}%")
    summery_writer.add_scalar('Accuracy/test', avg_accuracy)
    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_fake_real_classifier1.pth')


class Classifier(nn.Module):
    def train___init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # self.fc1 = nn.Linear(input_dim, 256)  # Adjust the hidden layer size as needed
        self.fc1 = nn.Linear(input_dim, num_classes)  # Adjust the hidden layer size as needed
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(256, 128)  # You can adjust the size of hidden layers
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        return x


def training_using_clip():
    import clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    data_path = 'Datasets/new_dataset_resent50/EfficientNetB4ST/IntegratedGradients/'
    summery_writer = SummaryWriter('clip_logs')
    classifier = Classifier(512, 2).to(device)
    dataset = CustomImageFolder(root=data_path, transform=preprocess)
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    valid_len = int(total_len * 0.2)
    test_len = total_len - train_len - valid_len
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_len, valid_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 50
    epoch_pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        print('\ntraining the model')
        batch_pbar = tqdm(total=len(train_loader))
        classifier.train()
        for images, xais, labels in train_loader:
            xais = xais.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                embeddings = model.encode_image(xais).type(torch.float32)
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            batch_pbar.update(1)
        batch_pbar.close()
        summery_writer.add_scalar('Loss/train', loss, epoch)
        # Validation
        classifier.eval()
        print('\nvalidation the model')
        validation_bar = tqdm(total=len(validation_loader))
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, xais, labels in validation_loader:
                xais = xais.to(device)
                labels = labels.to(device)
                embeddings = model.encode_image(xais).type(torch.float32)
                outputs = classifier(embeddings)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                validation_bar.update(1)
        validation_bar.close()

        accuracy = 100 * total_correct / total_samples
        print(f'\nEpoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%')
        summery_writer.add_scalar('Accuracy/validation', accuracy, epoch)
        epoch_pbar.update(1)
    epoch_pbar.close()
    print('\ntesting the model')
    # test the model with test dataset
    total_accuracy = 0
    classifier.eval()
    total_correct = 0
    total_samples = 0
    testing_bar = tqdm(total=len(test_loader))
    with torch.no_grad():
        for images, xais, labels in test_loader:
            xais = xais.to(device)
            labels = labels.to(device)
            embeddings = model.encode_image(xais).type(torch.float32)
            outputs = classifier(embeddings)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            testing_bar.update(1)
        total_accuracy += 100 * total_correct / total_samples

    avg_accuracy = total_accuracy / len(test_loader)
    print(f"Test Accuracy: {avg_accuracy * 100:.2f}%")
    summery_writer.add_scalar('Accuracy/test', avg_accuracy)
    testing_bar.close()


if __name__ == '__main__':
    example_for_train_resnet()
    # training_using_clip()
