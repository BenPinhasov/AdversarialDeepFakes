"""
Mesonet, need to name it classifier so that model loading works.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision


class myMeso4(nn.Module):
    """
    Pytorch Implemention of Meso4 like the tf implemention
    Autor: Ben Pinhasov
    Date: Aug 19, 2023
    """

    def __init__(self, num_classes=2):
        super(myMeso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding='same', bias=True)
        # self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

        self.conv2 = nn.Conv2d(8, 8, 5, padding='same', bias=True)
        # self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(8)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

        self.conv3 = nn.Conv2d(8, 16, 5, padding='same', bias=True)
        # self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

        self.conv4 = nn.Conv2d(16, 16, 5, padding='same', bias=True)
        # self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(16)
        self.maxpooling4 = nn.MaxPool2d(kernel_size=(4, 4))

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)  # (8, 256, 256)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 128, 128)

        x = self.conv2(x)  # (8, 128, 128)
        x = nn.ReLU()(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = nn.ReLU()(x)
        x = self.bn3(x)
        x = self.maxpooling3(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = nn.ReLU()(x)
        x = self.bn4(x)
        x = self.maxpooling4(x)  # (16, 8, 8)

        x = self.flatten(x)  # (Batch, 16*8*8)
        x = self.dropout1(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Meso4(nn.Module):
    """
    Pytorch Implemention of Meso4
    Autor: Honggu Liu
    Date: July 4, 2019
    """

    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        # flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input):
        x = self.conv1(input)  # (8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 128, 128)

        x = self.conv2(x)  # (8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)  # (16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MesoInception4(nn.Module):
    """
    Pytorch Implemention of MesoInception4
    Author: Honggu Liu
    Date: July 7, 2019
    """

    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        # InceptionLayer1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    # InceptionLayer
    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, input):
        x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
        x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

        x = self.conv1(x)  # (Batch, 16, 64 ,64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

        x = self.conv2(x)  # (Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.keras import layers
    from classifiers import Meso4

    pytorch_model = myMeso4()
    classifier = Meso4()
    classifier.load(r'C:\Users\Ben.pinhasov\PycharmProjects\MesoNet\weights\Meso4_DF.h5')
    for i, tf_layer in enumerate(classifier.model.layers):
        if isinstance(tf_layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(tf_layer, tf.keras.layers.MaxPool2D):
            continue
        if isinstance(tf_layer, tf.keras.layers.Flatten):
            continue
        if isinstance(tf_layer, tf.keras.layers.Dropout):
            continue
        if isinstance(tf_layer, tf.keras.layers.LeakyReLU):
            continue
        if isinstance(tf_layer, tf.keras.layers.ReLU):
            continue
        # Get corresponding PyTorch layer
        pt_layer = list(pytorch_model.modules())[i]

        # Transfer weights
        tf_weights = tf_layer.get_weights()[0]
        source_axes = range(tf_weights.ndim)
        dest_axes = range(tf_weights.ndim - 1, -1, -1)
        tf_weights = np.moveaxis(tf_weights, source_axes, dest_axes)

        pt_layer.weight.data = torch.from_numpy(tf_weights)
        if pt_layer.bias is not None:
            pt_layer.bias.data = torch.from_numpy(tf_layer.get_weights()[1])
    print('')
