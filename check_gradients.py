from dataset.transform import EfficientNetB4ST_default_data_transforms
import dlib
from network.models import model_selection
import torch
from dataset.transform import get_transformer
from torchvision import transforms
import cv2
import torch.nn.functional as F


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


def print_grad(grad):
    print(grad)


def main():
    model_path = 'models/EfficientNetB4ST_FFPP/bestval.pth'
    video_path = 'Datasets/manipulated_sequences/Deepfakes/c23/videos/033_097.mp4'
    # Face detector
    face_detector = dlib.get_frontal_face_detector()
    model = model_selection('EfficientNetB4ST', 2)
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = True

    for name, layer in model.named_modules():
        layer.register_backward_hook(lambda module, grad_input, grad_output: print_grad(grad_output))

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = get_transformer('scale', 224, normalizer, train=False)
    target = torch.tensor([[0]])
    reader = cv2.VideoCapture(video_path)
    while reader.isOpened():
        ret, image = reader.read()
        if ret:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            preprocessed_image = preprocess(image=cropped_face)['image']
            preprocessed_image = preprocessed_image.unsqueeze(0)
            preprocessed_image.requires_grad = True
            output = model(preprocessed_image)
            model.zero_grad()
            loss = F.nll_loss(output[0], target[0])
            loss.backward()

    print("Converted to cuda")


if __name__ == '__main__':
    main()