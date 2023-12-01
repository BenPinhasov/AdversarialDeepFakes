import os

from tqdm import tqdm

from xai_classification import CustomResNet50, calculate_accuracy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.transform import ImageXaiFolder


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_dir = 'runs'
    test_original_crops_path = r'newDataset\Test\Frames\original\xception\original'
    test_original_xai_path = r'newDataset\Test\Frames\original\xception\GuidedBackprop'
    test_attacked_path = r'newDataset\Test\Frames\attacked\Deepfakes\xception\original'
    test_attacked_xai_path = r'newDataset\Test\Frames\attacked\Deepfakes\xception\GuidedBackprop'

    runs = os.listdir(runs_dir)
    run_bar = tqdm(total=len(runs))
    for run in runs:
        if run.find('True') != -1:  # check if dropout is true
            model = CustomResNet50(2, dropout=True)
        else:
            model = CustomResNet50(2)
        model.to(device)
        working_path = os.path.join(runs_dir, run)
        f = open(os.path.join(working_path, 'best_model_acc.txt'), 'w')
        model.load_state_dict(torch.load(os.path.join(working_path, 'best_model.pth')))
        model.eval()
        activation_function = nn.Softmax(dim=1)

        test_dataset = ImageXaiFolder(original_path=test_original_crops_path,
                                      original_xai_path=test_original_xai_path,
                                      attacked_path=test_attacked_path,
                                      attacked_xai_path=test_attacked_xai_path,
                                      transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        total_accuracy = 0.0
        batch_bar = tqdm(total=len(test_loader))
        with torch.no_grad():
            for test_images, test_xais, test_labels in test_loader:
                test_images = test_images.to(device)
                test_xais = test_xais.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model(test_images.float(), test_xais.float())
                # test_outputs = model(test_xais.float())
                # test_outputs = activation_function(test_outputs)
                accuracy = calculate_accuracy(activation_function(test_outputs), test_labels)
                total_accuracy += accuracy
                batch_bar.update(1)
        batch_bar.close()
        avg_accuracy = total_accuracy / len(test_loader)
        print(f'Run: {run}, Accuracy: {avg_accuracy}')
        f.write('Accuracy: ' + str(avg_accuracy))
        f.close()
        run_bar.update(1)
    run_bar.close()

    pass


if __name__ == '__main__':
    main()
