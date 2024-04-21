import os

from torchmetrics import ROC
from tqdm import tqdm
from xai_classification import CustomResNet50, calculate_accuracy, CustomViT, CustomClip
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
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

    runs_main_dir = 'runs_resnet50_nograd'
    detector_types = ['EfficientNetB4ST', 'xception']
    attack_methods = ['iterative_fgsm']  # ['square', 'apgd-ce', 'black_box', 'ifgs']
    xai_methods_model = ['GuidedBackprop', 'InputXGradient', 'IntegratedGradients', 'Saliency']
    xai_methods_dataset = ['GuidedBackprop', 'InputXGradient', 'IntegratedGradients', 'Saliency']
    black_xai = False
    black_img = False
    rocs = {}
    for xai_method in xai_methods_model:
        xai_method_dataset = xai_method
        for detector_type in detector_types:
            # for xai_method_dataset in xai_methods_dataset:
            for test_detector_type in detector_types:
                for attack_method in attack_methods:
                    print(
                        f'Testing {detector_type} with {xai_method} and attack: {attack_method} on dataset: {test_detector_type}_{xai_method_dataset}')
                    runs_dir = f'{runs_main_dir}/{detector_type}/{xai_method}'
                    test_original_crops_path = rf'newDataset\Test\Frames\original\{test_detector_type}\original'
                    test_original_xai_path = rf'newDataset\Test\Frames\original\{test_detector_type}\{xai_method_dataset}'
                    test_attacked_path = rf'newDataset\Test\Frames\attacked\{attack_method}\Deepfakes\{test_detector_type}\original'
                    test_attacked_xai_path = rf'newDataset\Test\Frames\attacked\{attack_method}\Deepfakes\{test_detector_type}\{xai_method_dataset}'

                    runs = os.listdir(runs_dir)
                    run_bar = tqdm(total=len(runs))
                    if runs_dir.find('vit') != -1:
                        model = CustomViT()
                    elif runs_dir.find('clip') != -1:
                        model = CustomClip()
                    for run in runs:
                        if runs_dir.find('resnet50') != -1:
                            weights = ResNet50_Weights.DEFAULT
                            if run.find('True') != -1:  # check if dropout is true
                                model = CustomResNet50(weights=weights, dropout=True)
                            else:
                                model = CustomResNet50(weights=weights)
                        model.to(device)
                        working_path = os.path.join(runs_dir, run)
                        f = open(os.path.join(working_path,
                                              f'best_model_acc_{test_detector_type}_{xai_method_dataset}_{attack_method}_blackxai_{black_xai}.txt'),
                                 'w')
                        model.load_state_dict(torch.load(os.path.join(working_path, 'best_model.pth')))
                        model.eval()
                        activation_function = nn.Softmax(dim=1)

                        test_dataset = ImageXaiFolder(original_path=test_original_crops_path,
                                                      original_xai_path=test_original_xai_path,
                                                      attacked_path=test_attacked_path,
                                                      attacked_xai_path=test_attacked_xai_path,
                                                      transform=transform,
                                                      black_xai=black_xai,
                                                      black_img=black_img)
                        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
                        total_accuracy = 0.0
                        batch_bar = tqdm(total=len(test_loader))
                        concatenated_preds = torch.empty(0).to(device)
                        concatenated_labels = torch.empty(0).to(device)
                        r = ROC(task="multiclass", num_classes=2)
                        with torch.no_grad():
                            for test_images, test_xais, test_labels in test_loader:
                                test_images = test_images.to(device)
                                test_xais = test_xais.to(device)
                                test_labels = test_labels.to(device)
                                test_outputs = model(test_images.float(), test_xais.float())
                                concatenated_preds = torch.cat((concatenated_preds, activation_function(test_outputs)))
                                concatenated_labels = torch.cat((concatenated_labels, test_labels))
                                # test_outputs = model(test_xais.float())
                                # test_outputs = activation_function(test_outputs)
                                accuracy = calculate_accuracy(activation_function(test_outputs), test_labels)
                                total_accuracy += accuracy
                                batch_bar.update(1)
                        fpr, tpr, threshold = r(concatenated_preds,
                                                torch.argmax(concatenated_labels.squeeze().to(torch.long), dim=1))
                        rocs[f'{detector_type}_{xai_method}_{attack_method}'] = (r, fpr, tpr, threshold)
                        batch_bar.close()
                        avg_accuracy = total_accuracy / len(test_loader)
                        print(f'Run: {run}, Accuracy: {avg_accuracy}')
                        f.write('Accuracy: ' + str(avg_accuracy))
                        f.close()
                        run_bar.update(1)
                    run_bar.close()
    with open(f'{runs_main_dir}_rocs.pkl', 'wb') as f:
        import pickle
        pickle.dump(rocs, f)

    pass


if __name__ == '__main__':
    main()
