import cv2

from network.models import model_selection
from dataset.transform import xception_default_data_transforms, mesonet_default_data_transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import dlib
import torch

torch.cuda.empty_cache()
import torch.nn as nn
import os
import sys
from PIL import Image as pil_image
import json
from network.models import model_selection
from captum.attr import IntegratedGradients, InputXGradient, GuidedBackprop, Saliency, visualization
from matplotlib import pyplot as plt
import numpy as np


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, model_type, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we invoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape)
    :param model_type: string of the model type
    :param cuda: Bool if we need use cuda or not
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily cast to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if model_type == "xception":
        preprocess = xception_default_data_transforms['test']
    elif model_type == "meso":
        preprocess = mesonet_default_data_transforms['test']
    else:
        raise "Model type not supported"

    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, model_type, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param model_type: string of the model type
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, model_type, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)  # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def tensor_to_numpy(tensor):
    img = tensor.squeeze().T.cpu().detach().numpy()  # get normalized
    img = np.swapaxes(img, 1, 0)
    return img


def image_to_display(img, label=None, confidence=None):
    # img -= img.min()
    # img /= img.max()
    # img = (img * 255).squeeze().T.cpu().detach().numpy()  # get normalized
    # img = np.swapaxes(img, 1, 0).astype(dtype=np.uint8)
    # img = tensor_to_numpy(img).astype(dtype=np.uint8) * 255
    if label and confidence:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (5, 15)
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        img = cv2.putText(img.copy(), f'Classification: {label} {round(confidence, 3)}', org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
    return img


def display_images(xai_image, face):
    rows_rgb, cols_rgb, channels = face.shape
    rows_gray, cols_gray, _ = xai_image.shape

    rows_comb = max(rows_rgb, rows_gray)
    cols_comb = cols_rgb + cols_gray
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

    comb[:rows_rgb, :cols_rgb] = face
    comb[:rows_gray, cols_rgb:] = xai_image

    cv2.imshow('xai_wb', comb)
    cv2.waitKey(1)
    return comb


def main():
    video_path = 'Data/Trump_Deepfake.mp4'
    model_type = 'xception'
    model_path = 'faceforensics++_models_subset/face_detection/xception/all_c23.p'
    cuda = True
    xai_method = 'Saliency'  # IntegratedGradients, InputXGradient, GuidedBackprop, Saliency
    xai_methods = ['GuidedBackprop', 'Saliency', 'InputXGradient', 'IntegratedGradients']
    target = 0
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 10

    for xai_method in xai_methods:
        try:
            # fix video path string and get video file name (video_fn)
            video_path = video_path.replace('\\', '/') if '\\' in video_path else video_path
            video_fn = video_path.split('/')[-1].split('.')[0]
            # create video writer
            # writer = cv2.VideoWriter(video_fn + '_xai_' + xai_method + '.avi', fourcc, fps, (299, 598)[::-1])
            writer = cv2.VideoWriter(video_fn + '_xai_' + xai_method + '.avi', fourcc, fps, (600, 600)[::-1])
            # create frame reader
            reader = cv2.VideoCapture(video_path)

            # create face detector
            face_detector = dlib.get_frontal_face_detector()

            # load model
            if model_path is not None:
                if not cuda:
                    model = torch.load(model_path, map_location="cpu")
                else:
                    if model_type == 'meso':
                        model = model_selection(model_type, 2)[0]
                        weights = torch.load(model_path)
                        model.load_state_dict(weights)
                    elif model_type == 'xception':
                        model = torch.load(model_path)
                    else:
                        raise f"{model_type} not supported"
                print('Model found in {}'.format(model_path))
            else:
                raise 'No model found'
            if cuda:
                print("Converting mode to cuda")
                model = model.cuda()
                for param in model.parameters():
                    param.requires_grad = True
                print("Converted to cuda")

            # create xai
            if xai_method == 'IntegratedGradients':
                xai = IntegratedGradients(model)
            elif xai_method == 'InputXGradient':
                xai = InputXGradient(model)
            elif xai_method == 'GuidedBackprop':
                xai = GuidedBackprop(model)
            elif xai_method == 'Saliency':
                xai = Saliency(model)
            else:
                raise 'No xai method chosen'
            while reader.isOpened():
                ret, image = reader.read()
                if not ret:
                    break
                # get image size
                height, width = image.shape[:2]
                # detect face
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_detector(gray, 1)
                if len(faces):
                    # take only the biggest face
                    face = faces[0]
                    x, y, size = get_boundingbox(face, width, height)
                    cropped_face = image[y:y + size, x:x + size, :]

                    preprocessed_image = preprocess_image(cropped_face, model_type, cuda)

                    output = model(preprocessed_image)
                    output = nn.Softmax(dim=1)(output)
                    confidence, prediction = torch.max(output, 1)
                    confidence = float(confidence[0])
                    pred = int(prediction.cpu().numpy())
                    label = 'fake' if pred == 1 else 'real'

                    # face_to_disp = image_to_display(preprocessed_image, label, confidence)
                    # face_to_disp = image_to_display(preprocessed_image)
                    face_to_disp = tensor_to_numpy(preprocessed_image)
                    if xai_method == 'IntegratedGradients':
                        xai_img = xai.attribute(preprocessed_image, target=target, internal_batch_size=1)
                    else:
                        xai_img = xai.attribute(preprocessed_image, target=target)
                    # xai_img = image_to_display(xai_img)
                    xai_img = tensor_to_numpy(xai_img)
                    # fig, ax = visualization.visualize_image_attr_multiple(xai_img, face_to_disp,
                    #                                                       methods=['heat_map', 'original_image'],
                    #                                                       signs=['absolute_value', 'absolute_value'],
                    #                                                       show_colorbar=True)
                    fig, ax = visualization.visualize_image_attr(xai_img, face_to_disp, method='blended_heat_map',
                                                                 sign='absolute_value', show_colorbar=True,
                                                                 use_pyplot=False)
                    canvas = FigureCanvas(fig)
                    canvas.draw()
                    buf = canvas.buffer_rgba()
                    img_to_disp = image_to_display(np.asarray(buf)[:,:,:3], label=label, confidence=confidence)
                    cv2.imshow(xai_method, img_to_disp)
                    cv2.waitKey(1)
                    # xai_gray = cv2.cvtColor(xai_img, cv2.COLOR_BGR2GRAY)
                    # (thresh, blackAndWhiteImage) = cv2.threshold(xai_gray, 127, 255, cv2.THRESH_BINARY)
                    # TODO: Apply color map https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
                    # blackAndWhiteImage = cv2.applyColorMap(xai_gray, cv2.COLORMAP_HOT)
                    # comb_image = display_images(blackAndWhiteImage, cv2.cvtColor(face_to_disp, cv2.COLOR_RGB2BGR))
                    writer.write(img_to_disp)
                    pass
        except:
            import traceback
            traceback.print_exc()
            pass
        writer.release()
    pass


if __name__ == '__main__':
    main()
