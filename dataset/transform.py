"""

Author: Andreas Rössler
"""
import os
import random

from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from itertools import combinations
from PIL import Image

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        transforms.functional.normalize(tensor, self.mean, self.std, inplace=True)
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub(m).div(s)
        #     # The normalize code -> t.sub_(m).div_(s)
        return tensor


def get_transformer(face_policy: str, patch_size: int, net_normalizer: transforms.Normalize, train: bool):
    # Transformers and traindb
    if face_policy == 'scale':
        # The loader crops the face isotropically then scales to a square of size patch_size_load
        loading_transformations = [
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
            A.Resize(height=patch_size, width=patch_size, always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    elif face_policy == 'tight':
        # The loader crops the face tightly without any scaling
        loading_transformations = [
            A.LongestMaxSize(max_size=patch_size, always_apply=True),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    else:
        raise ValueError('Unknown value for face_policy: {}'.format(face_policy))

    if train:
        aug_transformations = [
            A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
                ]),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
                A.ImageCompression(quality_lower=50, quality_upper=99),
            ], )
        ]
    else:
        aug_transformations = []

    # Common final transformations
    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        ToTensorV2(),
    ]
    transf = A.Compose(
        loading_transformations + downsample_train_transformations + aug_transformations + final_transformations)
    return transf


EfficientNetB4ST_default_data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'to_tensor': transforms.Compose([
        transforms.ToTensor()
    ]),
    'normalize': transforms.Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'unnormalize': transforms.Compose([
        UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #     "to_tensor": A.Compose([
    #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ToTensorV2()
    #     ]),
    #     "un_normalize": A.Compose([
    #         A.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    #         A.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    #     ])
}

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),

    # Added these transforms for attack
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'unnormalize': transforms.Compose([
        UnNormalize([0.5] * 3, [0.5] * 3)
    ])
}

"""

Author: Honggu Liu
"""
mesonet_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),

    # Added these transforms for attack
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'unnormalize': transforms.Compose([
        UnNormalize([0.5] * 3, [0.5] * 3)
    ])
}


class ImageXaiFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.xai_extension = '_xai.jpg'
        self.image_files = []
        self.xai_map_files = []
        for subfolder in os.listdir(root):
            image_subfolder = os.path.join(root, subfolder)
            self.image_files.extend(
                [os.path.join(image_subfolder, f) for f in os.listdir(image_subfolder) if not f.endswith('_xai.jpg')])
            self.xai_map_files.extend(
                [os.path.join(image_subfolder, f) for f in os.listdir(image_subfolder) if f.endswith('_xai.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # image_path, label = self.imgs[idx]
        image_path = self.image_files[idx]
        # xai_path = image_path.replace(self.xai_extension, '.jpg')
        xai_path = image_path.split('.')[0] + self.xai_extension
        label = 1 if "attacked" in image_path else 0

        image = self.loader(image_path)
        xai_map = self.loader(xai_path)

        if self.transform is not None:
            image = self.transform(image)
            xai_map = self.transform(xai_map)

        return image, xai_map, label


class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = self.get_image_paths()
        # self.pairs, self.labels = self.create_pairs()
        self.pairs = self.create_pairs()
    def get_image_paths(self):
        image_paths = []
        for cls in self.classes:
            if cls == 'fake':
                continue
            class_dir = os.path.join(self.root_dir, cls)
            class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('_xai.jpg')]
            image_paths.extend(class_images)
        return image_paths

    def create_pairs(self):
        # pairs = []
        # labels = []
        # for i in range(len(self.image_paths)):
        #     for j in range(i + 1, len(self.image_paths)):
        #         img1, img2 = self.image_paths[i], self.image_paths[j]
        #         label = 1 if self.class_to_index[os.path.basename(os.path.dirname(img1))] == self.class_to_index[os.path.basename(os.path.dirname(img2))] else 0
        #         pairs.append((img1, img2))
        #         labels.append(label)
        # return pairs, labels
        # pairs = combinations(self.image_paths, 2)
        random_pairs = [(random.choice(self.image_paths), random.choice(self.image_paths)) for _ in range(100000)]
        return random_pairs
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = 1 if self.class_to_index[os.path.basename(os.path.dirname(img1_path))] == self.class_to_index[os.path.basename(os.path.dirname(img2_path))] else 0
        img1 = Image.open(img1_path) # Convert to grayscale
        img2 = Image.open(img2_path) # Convert to grayscale

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = float(label)  # Convert label to a float

        return img1, img2, label