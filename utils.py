import torch
import cv2
import numpy as np
from PIL import Image as pil_image
import torch.nn as nn
from torchvision import transforms

from dataset.transform import *


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


def preprocess_image(image, model_type, cuda=True, legacy=False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if not legacy:
        # only conver to tensor here,
        # other transforms -> resize, normalize differentiable done in predict_from_model func
        # same for meso, xception
        if model_type == 'meso' or model_type == 'xception':
            preprocess = xception_default_data_transforms['to_tensor']
            preprocessed_image = preprocess(pil_image.fromarray(image))
        elif model_type == 'EfficientNetB4ST':
            # preprocess = EfficientNetB4ST_default_data_transforms['to_tensor']
            # normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # preprocess = get_transformer('scale', 224, normalizer, train=False)
            # preprocessed_image = preprocess(image=image)['image']
            preprocess = EfficientNetB4ST_default_data_transforms['to_tensor']
            preprocessed_image = preprocess(pil_image.fromarray(image))

    else:
        if model_type == "xception":
            preprocess = xception_default_data_transforms['test']
            preprocessed_image = preprocess(pil_image.fromarray(image))

        elif model_type == "meso":
            preprocess = mesonet_default_data_transforms['test']
            preprocessed_image = preprocess(pil_image.fromarray(image))

        elif model_type == 'EfficientNetB4ST':
            # normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # preprocess = get_transformer('scale', 224, normalizer, train=False)
            # preprocessed_image = preprocess(image=image)['image']
            preprocess = EfficientNetB4ST_default_data_transforms['test']
            preprocessed_image = preprocess(pil_image.fromarray(image))

    preprocessed_image = preprocessed_image.unsqueeze(0)

    if cuda:
        preprocessed_image = preprocessed_image.cuda()

    preprocessed_image.requires_grad = True
    return preprocessed_image


def un_preprocess_image(image, size):
    """
    Tensor to PIL image and RGB to BGR
    """

    image.detach()
    new_image = image.squeeze(0)
    new_image = new_image.detach().cpu()

    undo_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size)
    ])

    new_image = undo_transform(new_image)
    new_image = np.array(new_image)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    return new_image


def un_preprocess_image_batch(images_batch, size):
    """
    Tensor to PIL image and RGB to BGR
    """
    image_list = []
    for i in range(images_batch.shape[0]):
        image = images_batch[i]
        image.detach()
        new_image = image.squeeze(0)
        new_image = new_image.detach().cpu()

        undo_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size)
        ])

        new_image = undo_transform(new_image)
        new_image = np.array(new_image)

        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        image_list.append(new_image)
    return image_list


def check_attacked(preprocessed_image, xai_map, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: torch tenosr (bs, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = attacked, 0 = real), output probs, logits
    """

    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    normalized_image = transform(preprocessed_image)
    normalized_xai = transform(xai_map)
    logits = model(normalized_image.float().cuda(), normalized_xai.float().cuda())
    output = post_function(logits)
    _, prediction = torch.max(output, 1)  # argmax
    prediction = float(prediction.cpu().numpy())
    output = output.detach().cpu().numpy().tolist()
    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits


def check_attacked_batch(preprocessed_images, xai_maps, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of input images in batch. Performs resizing and normalization before feeding in images.

    :param preprocessed_images: torch tensor containing preprocessed images (batch_size, c, h, w)
    :param xai_maps: torch tensor containing XAI maps (batch_size, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: boolean specifying whether to move tensors to CUDA
    :return: predictions (1 = attacked, 0 = real), output probs, logits
    """
    # Differentiable resizing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    normalized_images = torch.stack([transform(image) for image in preprocessed_images])
    normalized_xais = torch.stack([transform(xai_map) for xai_map in xai_maps])

    if cuda:
        normalized_images = normalized_images.cuda()
        normalized_xais = normalized_xais.cuda()

    # Model prediction
    logits = model(normalized_images.float(), normalized_xais.float())
    output = post_function(logits)
    _, predictions = torch.max(output, 1)  # argmax
    predictions = predictions.detach()
    output = output.detach()

    return predictions, output, logits


def predict_with_model_legacy(image, model, model_type, post_function=nn.Softmax(dim=1),
                              cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, model_type, cuda, legacy=True)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)
    if model_type == 'EfficientNetB4ST':
        fake_pred = output[0][1].item()
        real_pred = 1 - fake_pred
        output = np.array([real_pred, fake_pred])
        prediction = float(np.argmax(output))
        output = [output.tolist()]
    # Cast to desired
    else:
        _, prediction = torch.max(output, 1)  # argmax
        prediction = float(prediction.cpu().numpy())
        output = output.detach().cpu().numpy().tolist()
    return int(prediction), output


def predict_with_model_legacy_batch(images, model, model_type, post_function=nn.Softmax(dim=1),
                                    cuda=True):
    """
    Predicts the label of input images in batch. Preprocesses the input images and
    casts them to cuda if required.

    :param images: numpy array of images
    :param model: torch model with linear layer at the end
    :param model_type: string specifying the model type
    :param preprocess_func: function to preprocess the images in batch
    :param post_function: e.g., softmax
    :param cuda: boolean specifying whether to move tensors to CUDA
    :return: predictions (1 = fake, 0 = real) and corresponding outputs
    """
    batch_size = len(images)
    preprocessed_images = preprocess_image_batch(images, model_type, cuda, legacy=True)

    # Model prediction
    output = model(preprocessed_images)
    output = post_function(output)

    predictions = []
    outputs = []

    for i in range(batch_size):
        if model_type == 'EfficientNetB4ST':
            fake_pred = output[i][1].item()
            real_pred = 1 - fake_pred
            output_i = np.array([real_pred, fake_pred])
            prediction = float(np.argmax(output_i))
            output_i = [output_i.tolist()]
        else:
            _, prediction = torch.max(output[i], 0)  # argmax
            prediction = float(prediction.cpu().numpy())
            output_i = output[i].detach().cpu().numpy().tolist()

        predictions.append(int(prediction))
        outputs.append(output_i)

    return predictions, outputs


def calculate_xai_map(cropped_face, model, model_type, xai_calculator, xai_method, cuda=True):
    preprocessed_image = preprocess_image(cropped_face, model_type)
    prediction, output = predict_with_model_legacy(cropped_face, model, model_type, post_function=nn.Softmax(dim=1),
                                                   cuda=cuda)
    if xai_method == 'IntegratedGradients':
        xai_img = xai_calculator.attribute(preprocessed_image, target=prediction, internal_batch_size=1)
    else:
        xai_img = xai_calculator.attribute(preprocessed_image, target=prediction)
    return xai_img


def preprocess_image_batch(images, model_type, cuda=True, legacy=False):
    """
    Preprocesses the images such that they can be fed into our network.
    During this process, we convert each image to a PIL image and apply
    preprocessing transformations specific to each model type.

    :param images: numpy array of images in opencv form (i.e., BGR and of shape [batch_size, height, width, channels])
    :param model_type: string specifying the model type
    :param cuda: boolean specifying whether to move tensors to CUDA
    :param legacy: boolean specifying whether to use legacy preprocessing
    :return: pytorch tensor of shape [batch_size, 3, image_size, image_size], possibly casted to CUDA
    """
    batch_size = len(images)
    preprocessed_images = []

    for i in range(batch_size):
        image = images[i]

        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not legacy:
            if model_type == 'meso' or model_type == 'xception':
                preprocess = xception_default_data_transforms['to_tensor']
                preprocessed_image = preprocess(pil_image.fromarray(image))
            elif model_type == 'EfficientNetB4ST':
                preprocess = EfficientNetB4ST_default_data_transforms['to_tensor']
                preprocessed_image = preprocess(pil_image.fromarray(image))
        else:
            if model_type == "xception":
                preprocess = xception_default_data_transforms['test']
                preprocessed_image = preprocess(pil_image.fromarray(image))
            elif model_type == "meso":
                preprocess = mesonet_default_data_transforms['test']
                preprocessed_image = preprocess(pil_image.fromarray(image))
            elif model_type == 'EfficientNetB4ST':
                preprocess = EfficientNetB4ST_default_data_transforms['test']
                preprocessed_image = preprocess(pil_image.fromarray(image))

        preprocessed_images.append(preprocessed_image)

    preprocessed_images = torch.stack(preprocessed_images)

    if cuda:
        preprocessed_images = preprocessed_images.cuda()

    preprocessed_images.requires_grad = True
    return preprocessed_images


def calculate_xai_map_batch(cropped_faces, model, model_type, xai_calculator, xai_method, cuda=True):
    preprocessed_images = preprocess_image_batch(cropped_faces, model_type, cuda=cuda)

    predictions, outputs = predict_with_model_legacy_batch(cropped_faces, model, model_type,
                                                           post_function=nn.Softmax(dim=1),
                                                           cuda=cuda)
    if xai_method == 'IntegratedGradients':
        xai_imgs = xai_calculator.attribute(preprocessed_images, target=predictions, internal_batch_size=1)
    else:
        xai_imgs = xai_calculator.attribute(preprocessed_images, target=predictions)
    return xai_imgs
