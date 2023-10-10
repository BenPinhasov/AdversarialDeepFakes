import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm

from dataset.transform import ImageXaiFolder, SiameseDataset
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


# Define a Siamese Network using a pre-trained ResNet-50 as the base
class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward_one(self, x):
        output = self.base_network(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# Define a contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# Load a pre-trained ResNet-18 model
pretrained_resnet = resnet50(pretrained=True)
pretrained_resnet.fc = nn.Identity()  # Remove the fully connected layer

# Create Siamese Network using the pre-trained ResNet as the base
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
siamese_net = SiameseNetwork(pretrained_resnet).to(device)

# Define a contrastive loss
criterion = ContrastiveLoss()

# Define optimizer
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

# Load and preprocess your dataset
# You will need to prepare your dataset and data loaders
# as per your specific requirements.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_path = 'Datasets/dataset_custom_resent50/EfficientNetB4ST/IntegratedGradients/'
dataset = SiameseDataset(data_path, transform=transform)
total_len = len(dataset)
train_len = int(total_len * 0.7)
valid_len = int(total_len * 0.2)
test_len = total_len - train_len - valid_len
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                           [train_len, valid_len, test_len])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training loop
num_epochs = 10  # Adjust as needed
logs = SummaryWriter()
epoch_bar = tqdm(range(num_epochs))
for epoch in range(num_epochs):
    print('\n training')
    siamese_net.train()
    total_loss = 0.0
    train_bar = tqdm(train_loader)
    for batch_idx, (input1, input2, label) in enumerate(train_loader):
        optimizer.zero_grad()
        input1 = input1.to(device)
        input2 = input2.to(device)
        label = label.to(device)
        output1, output2 = siamese_net(input1, input2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_bar.update(1)
    train_bar.close()
    logs.add_scalar('Loss/train', total_loss / len(train_dataset), epoch)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {total_loss / len(train_dataset):.4f}")

    # Validation loop within the epoch
    print('\n validation')
    validation_bar = tqdm(validation_loader)
    siamese_net.eval()
    with torch.no_grad():
        val_losses = []
        val_predictions = []
        val_labels = []
        total_accuracy = 0
        for val_batch_idx, (val_input1, val_input2, val_label) in enumerate(validation_loader):
            val_label = val_label.to(device)
            val_input1 = val_input1.to(device)
            val_input2 = val_input2.to(device)
            val_output1, val_output2 = siamese_net(val_input1, val_input2)
            val_loss = criterion(val_output1, val_output2, val_label)
            val_losses.append(val_loss.item())
            # val_prediction = val_output1 - val_output2
            val_distances = torch.norm(val_output1 - val_output2, p=2, dim=1)
            val_predictions = [1 if p < 0.5 else 0 for p in val_distances]
            accuracy = accuracy_score(val_label.tolist(), val_predictions)
            # val_predictions.extend((val_output1 - val_output2).tolist())
            # val_labels.extend(val_label.tolist())
            total_accuracy += accuracy
            validation_bar.update(1)
            # accuracy = accuracy_score(val_labels, [1 if p < 0.5 else 0 for p in val_prediction])

        avg_accuracy = total_accuracy / len(validation_loader)

        val_accuracy = accuracy_score(val_labels, [1 if p < 0.5 else 0 for p in val_predictions])
        logs.add_scalar('Loss/val', sum(val_losses) / len(val_losses), epoch)
        print(f"Validation Loss: {avg_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    validation_bar.close()
    epoch_bar.update(1)
epoch_bar.close()

print('\n testing')
test_bar = tqdm(test_loader)
# Testing loop (outside the training loop)
siamese_net.eval()
with torch.no_grad():
    test_losses = []
    test_predictions = []
    test_labels = []

    for test_batch_idx, (test_input1, test_input2, test_label) in enumerate(test_loader):
        test_input1 = test_input1.to(device)
        test_input2 = test_input2.to(device)
        test_label = test_label.to(device)
        test_output1, test_output2 = siamese_net(test_input1, test_input2)
        test_loss = criterion(test_output1, test_output2, test_label)
        test_losses.append(test_loss.item())
        test_predictions.extend(test_output1 - test_output2)
        test_labels.extend(test_label)
        test_bar.update(1)

    test_accuracy = accuracy_score(test_labels, [1 if p < 0.5 else 0 for p in test_predictions])
    print(f"Test Loss: {sum(test_losses) / len(test_losses):.4f}, Test Accuracy: {test_accuracy:.4f}")
test_bar.close()

# Save or use the trained siamese_net for your classification task
